"""Lean trainer for SSL methods with AMP, accumulation and clean logs."""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from typing import Any, Dict, Optional, Iterable, Callable
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LRScheduler as SchedulerBase  # type: ignore
else:
    from torch.optim.lr_scheduler import _LRScheduler as SchedulerBase  # type: ignore

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

from mfcl.engines.hooks import Hook, HookList
from typing import Protocol
from collections.abc import Iterable as _CIterable
from mfcl.telemetry.comms_logger import get_comms_logger
from mfcl.telemetry.timers import StepTimer
from mfcl.telemetry.memory import MemoryMonitor
from mfcl.telemetry.power import PowerMonitor

# Global handle for telemetry integrations (e.g., comms logging) to access the
# active trainer instance without introducing import cycles.
CURRENT_TRAINER: "Trainer | None" = None
from mfcl.utils.amp import AmpScaler
from mfcl.utils.checkpoint import save_checkpoint, load_checkpoint
from mfcl.utils.consolemonitor import ConsoleMonitor
from mfcl.utils.dist import (
    barrier,
    get_world_size,
    is_main_process,
    reduce_dict,
    unwrap_ddp,
)
from mfcl.metrics.meter import SmoothedValue


class _TrainableMethod(Protocol):
    def to(self, device: torch.device) -> "_TrainableMethod": ...
    def train(self, mode: bool = ...) -> "_TrainableMethod": ...
    def parameters(self) -> _CIterable[nn.Parameter]: ...
    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = ...) -> Any: ...
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]: ...
    def step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]: ...
    def on_optimizer_step(self) -> None: ...


class Trainer:
    """Lean trainer for SSL methods."""

    def __init__(
        self,
        method: _TrainableMethod,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[SchedulerBase] = None,
        console: Optional[ConsoleMonitor] = None,
        device: Optional[torch.device] = None,
        scaler: Optional[AmpScaler] = None,
        hooks: Optional[Hook | HookList] = None,
        save_dir: Optional[str] = None,
        keep_k: int = 3,
        log_interval: int = 1,
        accum_steps: int = 1,
        clip_grad: Optional[float] = None,
        scheduler_step_on: str = "batch",
        guard_grad_nan: bool = False,
        channels_last_inputs: bool = False,
        gpu_augmentor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        timer: Optional[StepTimer] = None,
        memory_monitor: Optional[MemoryMonitor] = None,
        energy_monitor: Optional[PowerMonitor] = None,
    ) -> None:
        """Construct the trainer.

        Args:
            method: mfcl.methods.BaseMethod subclass with .step() and optional hooks.
            optimizer: Torch optimizer for method parameters.
            scheduler: Optional LR scheduler.
            console: ConsoleMonitor for live, clean logging.
            device: Compute device. If None, infer from CUDA availability.
            scaler: AMP scaler wrapper. If None, create default.
            hooks: Optional extra hooks (single or composite).
            save_dir: Directory for checkpoints. If None, disables saving.
            keep_k: How many checkpoints to retain when saving.
            log_interval: Print/update frequency in steps.
            accum_steps: Gradient accumulation steps per optimizer update (>=1).
            clip_grad: Max norm for gradient clipping (None disables).
            scheduler_step_on: Step scheduler per 'batch' or per 'epoch'.
            guard_grad_nan: If True, raise when gradients contain NaN or Inf.
            gpu_augmentor: Optional callable applied to batches after to_device to
                populate view tensors (used for GPU-side augmentation).
            timer: Optional StepTimer for telemetry logging.
        """
        self.method: _TrainableMethod = method
        self._method_impl = unwrap_ddp(method)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.console = console or ConsoleMonitor()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.scaler = scaler or AmpScaler(enabled=torch.cuda.is_available())
        self.hooks: HookList = (
            hooks if isinstance(hooks, HookList) else HookList([hooks] if hooks else [])
        )
        self.save_dir = save_dir
        self.keep_k = int(keep_k)
        self.log_interval = int(log_interval)
        if accum_steps < 1:
            raise ValueError("accum_steps must be >= 1")
        self.accum_steps = int(accum_steps)
        self.clip_grad = clip_grad
        if scheduler_step_on not in {"batch", "epoch"}:
            raise ValueError("scheduler_step_on must be 'batch' or 'epoch'")
        self.scheduler_step_on = scheduler_step_on
        self.guard_grad_nan = bool(guard_grad_nan)
        self.channels_last_inputs = bool(channels_last_inputs)
        self._gpu_augmentor = gpu_augmentor
        self.step_timer = timer
        self.memory_monitor = memory_monitor
        self.energy_monitor = energy_monitor
        try:
            env_steps = int(os.environ.get("MFCL_OOM_SEARCH_MAX_STEPS", "0"))
        except Exception:
            env_steps = 0
        self._max_steps_override = max(0, env_steps)

        base_method = unwrap_ddp(self.method)
        base_method.to(self.device)
        if self.channels_last_inputs:
            try:
                base_method.to(memory_format=torch.channels_last)
            except (TypeError, RuntimeError):
                pass
        # Global step counts processed micro-batches across the lifetime of the trainer
        self._global_step = 0
        self._last_move_bytes = 0

    def fit(
        self,
        train_loader: DataLoader | Iterable[Any],
        val_loader: Optional[DataLoader | Iterable[Any]] = None,
        epochs: int = 200,
        resume_path: Optional[str] = None,
        eval_every: int = 1,
        save_every: int = 1,
    ) -> None:
        """Train for N epochs.

        Args:
            train_loader: SSL train dataloader producing dict batches.
            val_loader: Optional labeled loader for periodic linear eval.
            epochs: Number of epochs to run.
            resume_path: If provided, load optimizer/scheduler/scaler/method and epoch.
            eval_every: Frequency in epochs to trigger on_eval_end hook with metrics dict.
            save_every: Frequency in epochs to save checkpoints.
        """
        global CURRENT_TRAINER
        prev_trainer = CURRENT_TRAINER
        CURRENT_TRAINER = self
        try:
            start_epoch = 1
            if resume_path and os.path.exists(resume_path):
                state = load_checkpoint(resume_path, strict=False)
                if state:
                    start_epoch = max(1, int(state.get("epoch", 0)) + 1)
                    self._restore_checkpoint_state(state)

            self.method.train()
            try:
                fn: Any = getattr(self._method_impl, "on_train_start", None)
                if callable(fn):
                    fn()
            except Exception:
                pass

            world_size = get_world_size()
            train_state = {
                "epoch": start_epoch,
                "global_step": self._global_step,
                "device": str(self.device),
                "world_size": world_size,
                "accum_steps": self.accum_steps,
                "save_dir": self.save_dir,
            }
            self.hooks.on_train_start(train_state)

            for epoch in range(start_epoch, epochs + 1):
                train_state["epoch"] = epoch
                train_state["global_step"] = self._global_step
                self.hooks.on_epoch_start(epoch, train_state)
                epoch_metrics = self.train_one_epoch(epoch, train_loader)
                # Ensure downstream hooks receive the epoch number alongside metrics so
                # they can align their internal schedules with the trainer state.
                if "epoch" not in epoch_metrics:
                    epoch_metrics = dict(epoch_metrics)
                    epoch_metrics["epoch"] = epoch
                train_state["global_step"] = self._global_step

                # Save checkpoint
                if is_main_process() and self.save_dir and (epoch % save_every == 0):
                    os.makedirs(self.save_dir, exist_ok=True)
                    ckpt = {
                        "epoch": epoch,
                        "global_step": self._global_step,
                        "method": self.method.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                        "scaler": self.scaler.state_dict() if self.scaler else None,
                        "metrics": epoch_metrics,
                    }
                    path = os.path.join(self.save_dir, f"ckpt_ep{epoch:04d}.pt")
                    save_checkpoint(path, ckpt, keep_k=self.keep_k, make_latest=True)
                    self.hooks.on_checkpoint(path, ckpt)

                # Eval hook
                if epoch % eval_every == 0:
                    self.hooks.on_eval_end(epoch_metrics)

                if self.scheduler and self.scheduler_step_on == "epoch":
                    self.scheduler.step()

                if self._max_steps_override and self._global_step >= self._max_steps_override:
                    break

            barrier()
        finally:
            CURRENT_TRAINER = prev_trainer

    def train_one_epoch(
        self, epoch: int, loader: DataLoader | Iterable[Any]
    ) -> Dict[str, float]:
        """Run one training epoch.

        Returns:
            Dict of averaged metrics for the epoch (includes 'loss' and 'lr').
        """
        self.method.train()
        loss_meter = SmoothedValue(window=50)
        time_meter = SmoothedValue(window=50)
        throughput_meter = SmoothedValue(window=50)

        from collections.abc import Sized as _Sized
        total = len(loader) if isinstance(loader, _Sized) else 0
        try:
            last_lr = float(self.optimizer.param_groups[0]["lr"])
        except Exception:
            last_lr = 0.0
        self.optimizer.zero_grad(set_to_none=True)
        # t0 = time.time()

        # Running sums for epoch-level reduction
        sum_loss = 0.0
        count = 0
        samples_seen = 0
        epoch_time = 0.0

        if is_main_process():
            self.console.epoch_start(epoch, total)

        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            try:
                sampler.set_epoch(epoch)
            except Exception:
                pass
        batch_sampler = getattr(loader, "batch_sampler", None)
        if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
            try:
                batch_sampler.set_epoch(epoch)
            except Exception:
                pass

        iterator = iter(loader)
        step = 0
        prev_end = time.time()
        data_time_total = 0.0
        compute_time_total = 0.0
        h2d_bytes_total = 0.0

        comms_logger = get_comms_logger()
        memory_monitor = self.memory_monitor
        energy_monitor = self.energy_monitor
        epoch_energy_start_wh = 0.0
        epoch_energy_start_j = 0.0
        if energy_monitor is not None:
            epoch_energy_start_wh, epoch_energy_start_j = energy_monitor.get_totals()
        max_steps_override = self._max_steps_override

        while True:
            data_start = prev_end
            try:
                batch = next(iterator)
            except StopIteration:
                break
            batch = self.to_device(batch)
            after_move = time.time()
            data_elapsed = after_move - data_start
            data_time_total += data_elapsed
            h2d_bytes_total += float(self._last_move_bytes)
            timer = self.step_timer
            if memory_monitor is not None:
                memory_monitor.update_step_context(
                    epoch=epoch,
                    step_index=step + 1,
                    global_step=self._global_step + 1,
                )
            if energy_monitor is not None:
                energy_monitor.update_step_context(
                    epoch=epoch,
                    step_index=step + 1,
                    global_step=self._global_step + 1,
                )
            if timer is not None:
                timer.begin_step(
                    epoch=epoch,
                    step_index=step + 1,
                    global_step=self._global_step + 1,
                )
                timer.record_data(data_elapsed)
            if comms_logger is not None:
                comms_logger.begin_step(
                    epoch=epoch,
                    step_index=step + 1,
                    global_step=self._global_step + 1,
                    timer=timer,
                )
            if self._gpu_augmentor is not None:
                batch = self._gpu_augmentor(batch)
            compute_start = time.time()
            forward_ctx = timer.range_forward() if timer is not None else nullcontext()
            with forward_ctx:
                with self.scaler.autocast():
                    stats = self.method(batch)
                if "loss" not in stats:
                    raise KeyError("Method step() must return dict with key 'loss'")
                loss = stats["loss"]
            finite = torch.isfinite(loss.detach())
            if finite.dim() == 0:
                finite_flag = bool(finite.item())
            else:
                finite_flag = bool(finite.all().item())
            if not finite_flag:
                self.console.newline()
                raise RuntimeError("Loss exploded (non-finite scalar encountered)")

            loss_to_backward = loss / self.accum_steps
            backward_ctx = timer.range_backward() if timer is not None else nullcontext()
            with backward_ctx:
                self.scaler.scale(loss_to_backward).backward()

            # Update running sums for epoch averages
            sum_loss += float(loss.detach().to(torch.float32).item())
            count += 1

            do_step = ((step + 1) % self.accum_steps) == 0
            if do_step:
                optimizer_ctx = (
                    timer.range_optimizer() if timer is not None else nullcontext()
                )
                with optimizer_ctx:
                    last_lr = self._apply_optimizer_step(self.accum_steps)

            # Update meters and console
            compute_end = time.time()
            compute_elapsed = compute_end - compute_start
            compute_time_total += compute_elapsed
            loss_meter.update(float(loss.detach().to(torch.float32).item()))

            misc_ctx = timer.range_misc() if timer is not None else nullcontext()
            with misc_ctx:
                self._global_step += 1
                hook_metrics = self._stats_to_floats(stats)
                hook_metrics.setdefault(
                    "loss", float(loss.detach().to(torch.float32).item())
                )
                hook_metrics.setdefault("lr", float(last_lr))
                hook_metrics.setdefault("data_time", data_elapsed)
                hook_metrics.setdefault("compute_time", compute_elapsed)
                self.hooks.on_batch_end(self._global_step, hook_metrics)

                if memory_monitor is not None:
                    memory_monitor.record_step_snapshot(
                        epoch=epoch,
                        global_step=self._global_step,
                    )

                should_tail = (total > 0 and step == total - 1)
                if is_main_process() and (
                    step % self.log_interval == 0 or should_tail
                ):
                    metrics: Dict[str, float] = {
                        "loss": loss_meter.global_avg,
                        "lr": float(last_lr),
                    }
                    if throughput_meter.count > 0:
                        metrics["ips"] = throughput_meter.global_avg
                    for k in (
                        "pos_sim",
                        "neg_sim_mean",
                        "cos_sim",
                        "diag_mean",
                        "offdiag_mean",
                        "mse",
                        "std_mean",
                        "cov_offdiag",
                    ):
                        v = stats.get(k)
                        if v is not None:
                            try:
                                metrics[k] = float(v.detach().to(torch.float32).item())
                            except Exception:
                                continue
                    self.console.live(
                        epoch,
                        step + 1,
                        total,
                        metrics,
                        metric_order=(
                            "loss",
                            "lr",
                            "ips",
                            "pos_sim",
                            "neg_sim_mean",
                        ),
                    )
                step_end = time.time()

            dt = step_end - data_start
            time_meter.update(dt)
            batch_size = self._infer_batch_size(batch)
            if batch_size > 0 and dt > 0:
                throughput_meter.update(batch_size / dt)
                samples_seen += batch_size
            epoch_time += dt
            if timer is not None:
                ips_value = (batch_size / dt) if (batch_size > 0 and dt > 0) else 0.0
                timer.end_step(step_time_s=dt, ips=ips_value)
            if comms_logger is not None:
                comms_logger.end_step()

            prev_end = step_end
            step += 1

            if max_steps_override and self._global_step >= max_steps_override:
                break

        # Flush gradients if the final micro-batch count does not evenly divide the
        # accumulation factor. Without this step, the trailing micro-batches would
        # never update the model parameters and would leak into the next epoch.
        if count > 0 and (count % self.accum_steps) != 0:
            last_lr = self._apply_optimizer_step(count % self.accum_steps)

        # Epoch-level reduce (loss)
        reduce_map = {
            "sum_loss": torch.tensor(sum_loss, device=self.device),
            "count": torch.tensor(count, device=self.device),
        }
        if get_world_size() > 1:
            reduce_map = reduce_dict(reduce_map, op="sum")  # type: ignore[assignment]
        epoch_loss = (reduce_map["sum_loss"] / (reduce_map["count"] + 1e-12)).item()

        global_samples = float(samples_seen)
        global_epoch_time = float(epoch_time)
        epoch_energy_wh_value = 0.0
        epoch_energy_per_image_j = 0.0
        epoch_energy_j_value = 0.0
        epoch_energy_cost = 0.0
        if energy_monitor is not None:
            total_wh, total_j = energy_monitor.get_totals()
            energy_epoch_wh = max(0.0, total_wh - epoch_energy_start_wh)
            energy_epoch_j = max(0.0, total_j - epoch_energy_start_j)
            epoch_energy_wh_value = energy_epoch_wh
            epoch_energy_j_value = energy_epoch_j
            epoch_energy_cost = energy_monitor.get_epoch_cost(energy_epoch_wh)
        if get_world_size() > 1 and dist.is_available() and dist.is_initialized():
            try:
                samples_tensor = torch.tensor(
                    global_samples, device=self.device, dtype=torch.float64
                )
                time_tensor = torch.tensor(
                    global_epoch_time, device=self.device, dtype=torch.float64
                )
                dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)
                global_samples = float(samples_tensor.item())
                global_epoch_time = float(time_tensor.item())
            except Exception:
                pass

        if energy_monitor is not None:
            epoch_energy_per_image_j = (
                epoch_energy_j_value / global_samples if global_samples > 0 else 0.0
            )

        if is_main_process():
            self.console.newline()
            summary_metrics = {
                "loss": epoch_loss,
                "lr": float(last_lr),
                "time_per_batch": time_meter.global_avg,
            }
            if epoch_time > 0 and samples_seen > 0:
                summary_metrics["imgs_per_sec"] = samples_seen / epoch_time
            if global_epoch_time > 0 and global_samples > 0:
                summary_metrics["global_imgs_per_sec"] = global_samples / global_epoch_time
            if count > 0:
                summary_metrics["data_time"] = data_time_total / count
                summary_metrics["compute_time"] = compute_time_total / count
                summary_metrics["h2d_mb_per_step"] = (h2d_bytes_total / count) / (1024 ** 2)
            if energy_monitor is not None:
                summary_metrics["energy_epoch_Wh"] = epoch_energy_wh_value
                summary_metrics["energy_per_image_J"] = epoch_energy_per_image_j
                if epoch_energy_cost > 0:
                    summary_metrics["energy_epoch_cost_usd"] = epoch_energy_cost
            self.console.summary(epoch, summary_metrics)

        return {
            "loss": float(epoch_loss),
            "lr": float(last_lr),
            "time_per_batch": float(time_meter.global_avg),
            "imgs_per_sec": float(samples_seen / epoch_time) if epoch_time > 0 else 0.0,
            "global_imgs_per_sec": float(global_samples / global_epoch_time)
            if global_epoch_time > 0
            else 0.0,
            "data_time": data_time_total / count if count > 0 else 0.0,
            "compute_time": compute_time_total / count if count > 0 else 0.0,
            "h2d_mb_per_step": (h2d_bytes_total / count) / (1024 ** 2)
            if count > 0
            else 0.0,
            "energy_epoch_Wh": epoch_energy_wh_value,
            "energy_per_image_J": epoch_energy_per_image_j,
            "energy_epoch_cost_usd": epoch_energy_cost,
        }

    def _apply_optimizer_step(self, micro_batches_in_step: int) -> float:
        """Apply an optimizer step handling AMP, clipping and scheduling."""
        # Unscale before optional gradient clipping to avoid scaling issues
        self.scaler.unscale_(self.optimizer)
        if micro_batches_in_step <= 0:
            raise ValueError("micro_batches_in_step must be positive")
        if micro_batches_in_step != self.accum_steps:
            # Scale gradients so the effective average matches the actual micro-batch
            # count for this optimizer step. Without this adjustment the gradients
            # from a short accumulation window (e.g. epoch flush) would be scaled
            # down by ``micro_batches_in_step / self.accum_steps``.
            scale = self.accum_steps / float(micro_batches_in_step)
            for group in self.optimizer.param_groups:
                for p in group.get("params", []):
                    if p.grad is not None:
                        p.grad.detach().mul_(scale)
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(
                self.method.parameters(), max_norm=float(self.clip_grad)
            )
        if self.guard_grad_nan:
            total_norm = 0.0
            for group in self.optimizer.param_groups:
                for p in group.get("params", []):
                    if p.grad is not None:
                        v = p.grad.detach()
                        if torch.any(torch.isnan(v)) or torch.any(torch.isinf(v)):
                            raise RuntimeError("Detected NaN/Inf in gradients")
                        total_norm += float(v.norm(2).detach().item() ** 2)
            # total_norm retained for potential debugging/logging hooks.
        # Apply optimizer step via scaler; detect if the step was skipped due to inf/NaN
        prev_scale: float | None = None
        if hasattr(self.scaler, "scaler") and self.scaler.scaler is not None:
            try:
                prev_scale = float(self.scaler.scaler.get_scale())
            except Exception:
                prev_scale = None
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        method_impl = self._method_impl
        if hasattr(method_impl, "on_optimizer_step"):
            try:
                method_impl.on_optimizer_step()
            except Exception:
                pass
        if self.scheduler and self.scheduler_step_on == "batch":
            # Only step LR scheduler when the optimizer performed an update.
            did_step = True
            if prev_scale is not None and hasattr(self.scaler, "scaler") and self.scaler.scaler is not None:
                try:
                    new_scale = float(self.scaler.scaler.get_scale())
                    # If the scale decreased, GradScaler skipped optimizer.step() due to inf/NaN.
                    if new_scale < float(prev_scale):
                        did_step = False
                except Exception:
                    did_step = True
            if did_step:
                self.scheduler.step()
        return self.optimizer.param_groups[0]["lr"]

    @staticmethod
    def _stats_to_floats(stats: Dict[str, Any]) -> Dict[str, float]:
        """Convert scalar-like entries in ``stats`` to floats for hooks/loggers."""
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

    def to_device(self, obj: Any) -> Any:
        """Recursively move tensors in obj to self.device."""
        bytes_moved = 0

        def _move(item: Any) -> Any:
            nonlocal bytes_moved
            if torch.is_tensor(item):
                if item.device != self.device:
                    bytes_moved += item.element_size() * item.numel()
                tensor = item.to(self.device, non_blocking=True)
                if (
                    self.channels_last_inputs
                    and tensor.ndim == 4
                    and tensor.is_floating_point()
                ):
                    tensor = tensor.contiguous(memory_format=torch.channels_last)
                return tensor
            if isinstance(item, dict):
                return {k: _move(v) for k, v in item.items()}
            if isinstance(item, (list, tuple)):
                seq = [_move(x) for x in item]
                if isinstance(item, tuple):
                    try:
                        if hasattr(item, "_fields"):
                            return type(item)(*seq)
                        return type(item)(seq)
                    except TypeError:
                        try:
                            return type(item)(*seq)
                        except TypeError:
                            return tuple(seq)
                return seq
            return item

        result = _move(obj)
        self._last_move_bytes = bytes_moved
        return result

    @staticmethod
    def _infer_batch_size(batch: Any) -> int:
        """Best-effort batch-size inference for throughput metrics."""

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

    def _restore_checkpoint_state(self, state: Dict[str, Any]) -> None:
        """Restore method/optimizer/scheduler/scaler state from a checkpoint."""

        def _maybe(name: str, payload: Any, loader) -> None:
            if payload in (None, {}):
                return
            try:
                loader(payload)
            except Exception as exc:  # pragma: no cover - defensive
                raise RuntimeError(f"Failed to load {name} from checkpoint") from exc

        method_state = state.get("method", {})
        if isinstance(method_state, dict):
            try:
                self.method.load_state_dict(method_state, strict=False)
            except Exception as exc:  # pragma: no cover - defensive
                raise RuntimeError("Failed to load method state from checkpoint") from exc

        opt_state = state.get("optimizer", {})
        _maybe("optimizer", opt_state, self.optimizer.load_state_dict)

        if self.scheduler is not None:
            sched_state = state.get("scheduler", {})
            _maybe("scheduler", sched_state, self.scheduler.load_state_dict)

        if self.scaler is not None:
            scaler_state = state.get("scaler", {})
            _maybe("scaler", scaler_state, self.scaler.load_state_dict)

        self._global_step = int(state.get("global_step", self._global_step))


__all__ = ["Trainer"]
