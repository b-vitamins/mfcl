"""Step execution helpers for the trainer."""

from __future__ import annotations

import time
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Any, Dict, Iterator, TYPE_CHECKING

from collections.abc import Sized as _Sized

import torch
import torch.distributed as dist

from mfcl.metrics.meter import SmoothedValue
from mfcl.runtime.budget import BudgetTracker
from mfcl.utils.dist import get_world_size, is_main_process, reduce_dict

from .budget import BudgetEnforcer
from .telemetry import TelemetryEpochState, TelemetryManager, StepTelemetryResult
from .logging import log_exception

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from torch.utils.data import DataLoader
    from mfcl.telemetry.timers import StepTimer
    from .trainer import Trainer


@dataclass
class EpochState:
    epoch: int
    iterator: Iterator[Any]
    total: int
    loss_meter: SmoothedValue
    time_meter: SmoothedValue
    throughput_meter: SmoothedValue
    last_lr: float
    sum_loss: float
    count: int
    samples_seen: int
    epoch_time: float
    data_time_total: float
    compute_time_total: float
    h2d_bytes_total: float
    prev_end: float
    step: int
    telemetry: TelemetryEpochState
    max_steps_override: int
    budget: BudgetTracker | None


@dataclass
class StepContext:
    batch: Dict[str, Any]
    data_start: float
    data_elapsed: float
    compute_start: float
    tokens_this_step: int
    timer: "StepTimer | None"


class StepExecutor:
    """Execute training steps while delegating telemetry and budget logic."""

    def __init__(
        self,
        trainer: "Trainer",
        *,
        telemetry: TelemetryManager,
        budget: BudgetEnforcer,
    ) -> None:
        self._trainer = trainer
        self._telemetry = telemetry
        self._budget = budget

    def start_epoch(
        self,
        epoch: int,
        loader: "DataLoader | Iterable[Any]",
    ) -> EpochState:
        trainer = self._trainer
        trainer.method.train()
        loss_meter = SmoothedValue(window=50)
        time_meter = SmoothedValue(window=50)
        throughput_meter = SmoothedValue(window=50)

        total = len(loader) if isinstance(loader, _Sized) else 0
        try:
            last_lr = float(trainer.optimizer.param_groups[0]["lr"])
        except Exception:
            last_lr = 0.0
        trainer.optimizer.zero_grad(set_to_none=True)

        if is_main_process():
            trainer.console.epoch_start(epoch, total)

        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            try:
                sampler.set_epoch(epoch)
            except Exception as exc:
                log_exception("sampler.set_epoch", exc, epoch=epoch)
        batch_sampler = getattr(loader, "batch_sampler", None)
        if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
            try:
                batch_sampler.set_epoch(epoch)
            except Exception as exc:
                log_exception("batch_sampler.set_epoch", exc, epoch=epoch)

        iterator = iter(loader)
        prev_end = time.time()

        telemetry_state = self._telemetry.build_epoch_state()

        return EpochState(
            epoch=epoch,
            iterator=iterator,
            total=total,
            loss_meter=loss_meter,
            time_meter=time_meter,
            throughput_meter=throughput_meter,
            last_lr=last_lr,
            sum_loss=0.0,
            count=0,
            samples_seen=0,
            epoch_time=0.0,
            data_time_total=0.0,
            compute_time_total=0.0,
            h2d_bytes_total=0.0,
            prev_end=prev_end,
            step=0,
            telemetry=telemetry_state,
            max_steps_override=trainer._max_steps_override,
            budget=trainer.budget_tracker,
        )

    def prepare_step(self, epoch: int, state: EpochState) -> StepContext | None:
        trainer = self._trainer
        budget_tracker = state.budget
        if budget_tracker is not None and self._budget.sync_stop_signal(
            budget_tracker.should_stop()
        ):
            return None

        data_start = state.prev_end
        try:
            batch = next(state.iterator)
        except StopIteration:
            return None

        batch = trainer.to_device(batch)
        after_move = time.time()
        data_elapsed = after_move - data_start
        state.data_time_total += data_elapsed
        state.h2d_bytes_total += float(trainer._last_move_bytes)

        timer = state.telemetry.timer
        if trainer._gpu_augmentor is not None:
            batch = trainer._gpu_augmentor(batch)

        tokens_this_step = 0
        if budget_tracker is not None:
            result = self._budget.evaluate_step_budget(batch)
            tokens_this_step = result.tokens_this_step
            if result.should_stop:
                return None

        step_index = state.step + 1
        global_step = trainer._global_step + 1

        self._telemetry.on_prepare_step(
            state.telemetry,
            epoch=epoch,
            step_index=step_index,
            global_step=global_step,
            data_elapsed=data_elapsed,
        )

        compute_start = time.time()

        return StepContext(
            batch=batch,
            data_start=data_start,
            data_elapsed=data_elapsed,
            compute_start=compute_start,
            tokens_this_step=tokens_this_step,
            timer=timer,
        )

    def execute_step(
        self,
        epoch: int,
        state: EpochState,
        step_ctx: StepContext,
    ) -> tuple[Dict[str, Any], torch.Tensor]:
        trainer = self._trainer
        timer = step_ctx.timer
        forward_ctx = timer.range_forward() if timer is not None else nullcontext()
        try:
            with forward_ctx:
                with trainer.scaler.autocast():
                    stats = trainer.method(step_ctx.batch)
                if "loss" not in stats:
                    raise KeyError("Method step() must return dict with key 'loss'")
                loss = stats["loss"]
                if trainer.topr_monitor is not None:
                    try:
                        trainer._maybe_update_topr(
                            epoch=epoch,
                            global_step=trainer._global_step + 1,
                        )
                    except Exception as exc:
                        log_exception(
                            "topr_monitor.update",
                            exc,
                            epoch=epoch,
                            step=trainer._global_step + 1,
                        )
        finally:
            self._telemetry.on_forward_complete(
                state.telemetry,
                epoch=epoch,
                global_step=trainer._global_step + 1,
            )

        return stats, loss

    def backward_and_update(
        self,
        epoch: int,
        state: EpochState,
        step_ctx: StepContext,
        stats: Dict[str, Any],
        loss: torch.Tensor,
    ) -> float:
        trainer = self._trainer
        finite = torch.isfinite(loss.detach())
        finite_flag = bool(finite.item()) if finite.dim() == 0 else bool(finite.all().item())
        if trainer.stability_sentry is not None:
            trainer.stability_sentry.check_loss(
                loss,
                batch=step_ctx.batch,
                optimizer=trainer.optimizer,
                scaler=trainer.scaler,
                step=trainer._global_step + 1,
                epoch=epoch,
            )
        if not finite_flag:
            trainer.console.newline()
            raise RuntimeError("Loss exploded (non-finite scalar encountered)")

        loss_to_backward = loss / trainer.accum_steps
        timer = step_ctx.timer
        backward_ctx = timer.range_backward() if timer is not None else nullcontext()
        with backward_ctx:
            trainer.scaler.scale(loss_to_backward).backward()

        loss_scalar = float(loss.detach().to(torch.float32).item())
        state.sum_loss += loss_scalar
        state.count += 1

        do_step = ((state.step + 1) % trainer.accum_steps) == 0
        if do_step:
            optimizer_ctx = timer.range_optimizer() if timer is not None else nullcontext()
            with optimizer_ctx:
                state.last_lr = trainer._apply_optimizer_step(
                    trainer.accum_steps,
                    epoch=epoch,
                    global_step=trainer._global_step + 1,
                    batch=step_ctx.batch,
                )

        return loss_scalar

    def finalize_step(
        self,
        epoch: int,
        state: EpochState,
        step_ctx: StepContext,
        stats: Dict[str, Any],
        loss: torch.Tensor,
        loss_scalar: float,
    ) -> bool:
        trainer = self._trainer
        compute_end = time.time()
        compute_elapsed = compute_end - step_ctx.compute_start
        state.compute_time_total += compute_elapsed
        state.loss_meter.update(loss_scalar)

        hook_metrics = self._stats_to_floats(stats)
        hook_metrics.setdefault("loss", loss_scalar)
        hook_metrics.setdefault("lr", float(state.last_lr))
        hook_metrics.setdefault("data_time", step_ctx.data_elapsed)
        hook_metrics.setdefault("compute_time", compute_elapsed)

        telemetry_result = self._telemetry.on_step_finalize(
            state.telemetry,
            epoch=epoch,
            state=state,
            step_ctx=step_ctx,
            hook_metrics=hook_metrics,
            stats=stats,
            loss=loss,
            loss_scalar=loss_scalar,
        )

        dt = telemetry_result.step_end - step_ctx.data_start
        state.time_meter.update(dt)
        batch_size = self._infer_batch_size(step_ctx.batch)
        ips_value = 0.0
        if batch_size > 0 and dt > 0:
            ips_value = batch_size / dt
            state.throughput_meter.update(ips_value)
            state.samples_seen += batch_size
        state.epoch_time += dt

        self._telemetry.finalize_timer(
            state.telemetry,
            step_time=dt,
            ips=ips_value,
        )

        state.prev_end = telemetry_result.step_end
        state.step += 1

        budget_tracker = state.budget
        if budget_tracker is not None:
            budget_tracker.update(
                step_samples=step_ctx.tokens_this_step,
                step_wall_ms=dt * 1000.0,
                comm_bytes=int(telemetry_result.comm_bytes_total),
                energy_Wh=telemetry_result.energy_delta_wh,
            )
            if self._budget.sync_stop_signal(budget_tracker.should_stop()):
                return True

        if state.max_steps_override and trainer._global_step >= state.max_steps_override:
            return True

        return False

    def finish_epoch(self, epoch: int, state: EpochState) -> Dict[str, float]:
        trainer = self._trainer
        if state.count > 0 and (state.count % trainer.accum_steps) != 0:
            state.last_lr = trainer._apply_optimizer_step(
                state.count % trainer.accum_steps,
                epoch=epoch,
                global_step=trainer._global_step,
                batch=None,
            )

        reduce_map = {
            "sum_loss": torch.tensor(state.sum_loss, device=trainer.device),
            "count": torch.tensor(state.count, device=trainer.device),
        }
        if get_world_size() > 1:
            reduce_map = reduce_dict(reduce_map, op="sum")  # type: ignore[assignment]
        epoch_loss = (
            reduce_map["sum_loss"] / (reduce_map["count"] + 1e-12)
        ).item()

        global_samples = float(state.samples_seen)
        global_epoch_time = float(state.epoch_time)

        energy_info = self._telemetry.compute_epoch_energy(state.telemetry)
        epoch_energy_wh_value = energy_info["epoch_energy_wh"]
        epoch_energy_cost = energy_info["epoch_energy_cost"]
        energy_epoch_j = energy_info["epoch_energy_j"]
        energy_per_image_j = (
            energy_epoch_j / global_samples if global_samples > 0 else 0.0
        )

        if get_world_size() > 1 and dist.is_available() and dist.is_initialized():
            try:
                samples_tensor = torch.tensor(
                    global_samples, device=trainer.device, dtype=torch.float64
                )
                time_tensor = torch.tensor(
                    global_epoch_time, device=trainer.device, dtype=torch.float64
                )
                dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)
                global_samples = float(samples_tensor.item())
                global_epoch_time = float(time_tensor.item())
            except Exception as exc:
                log_exception(
                    "distributed epoch metrics reduction",
                    exc,
                    epoch=epoch,
                )

        metrics: Dict[str, float] = {
            "loss": float(epoch_loss),
            "lr": float(state.last_lr),
            "data_time": float(state.data_time_total / max(1, state.count)),
            "compute_time": float(state.compute_time_total / max(1, state.count)),
            "h2d_bytes": float(state.h2d_bytes_total),
            "samples_per_s": (
                float(global_samples / max(global_epoch_time, 1e-6))
                if global_epoch_time > 0
                else 0.0
            ),
            "epoch_time": float(global_epoch_time),
            "epoch_energy_wh": float(epoch_energy_wh_value),
            "epoch_energy_cost": float(epoch_energy_cost),
            "epoch_energy_j": float(energy_epoch_j),
            "energy_per_image_J": float(energy_per_image_j),
            "energy_epoch_cost_usd": float(epoch_energy_cost),
        }

        if is_main_process():
            trainer.console.newline()
            summary_metrics = {
                "loss": epoch_loss,
                "lr": float(state.last_lr),
                "time_per_batch": state.time_meter.global_avg,
            }
            if state.epoch_time > 0 and state.samples_seen > 0:
                summary_metrics["imgs_per_sec"] = (
                    state.samples_seen / state.epoch_time
                )
            if global_epoch_time > 0 and global_samples > 0:
                summary_metrics["global_imgs_per_sec"] = (
                    global_samples / global_epoch_time
                )
            if state.count > 0:
                summary_metrics["data_time"] = (
                    state.data_time_total / state.count
                )
                summary_metrics["compute_time"] = (
                    state.compute_time_total / state.count
                )
                summary_metrics["h2d_mb_per_step"] = (
                    (state.h2d_bytes_total / state.count) / (1024 ** 2)
                )
            if state.telemetry.energy_monitor is not None:
                summary_metrics["energy_epoch_Wh"] = epoch_energy_wh_value
                summary_metrics["energy_per_image_J"] = energy_per_image_j
                if epoch_energy_cost > 0:
                    summary_metrics["energy_epoch_cost_usd"] = epoch_energy_cost
            trainer.console.summary(epoch, summary_metrics)

        return metrics

    @staticmethod
    def _stats_to_floats(stats: Dict[str, Any]) -> Dict[str, float]:
        flat: Dict[str, float] = {}
        for key, value in stats.items():
            if torch.is_tensor(value):
                if value.numel() == 1:
                    flat[key] = float(value.detach().to(torch.float32).item())
            else:
                try:
                    flat[key] = float(value)
                except (TypeError, ValueError):
                    continue
        return flat

    @staticmethod
    def _infer_batch_size(batch: Any) -> int:
        if torch.is_tensor(batch):
            return int(batch.shape[0]) if batch.ndim > 0 else 0
        if isinstance(batch, dict):
            idx = batch.get("index")
            if torch.is_tensor(idx) and idx.ndim > 0:
                return int(idx.shape[0])
            for value in batch.values():
                if torch.is_tensor(value) and value.ndim > 0:
                    return int(value.shape[0])
                if isinstance(value, list) and value:
                    first = value[0]
                    if torch.is_tensor(first) and first.ndim > 0:
                        return int(first.shape[0])
        if isinstance(batch, (list, tuple)) and batch:
            first = batch[0]
            if torch.is_tensor(first) and first.ndim > 0:
                return int(first.shape[0])
            if isinstance(first, (list, tuple)):
                return len(batch)
        return 0


__all__ = ["StepExecutor", "EpochState", "StepContext"]

