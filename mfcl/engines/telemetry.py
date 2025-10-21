"""Telemetry helpers shared across engine components."""

from __future__ import annotations

import time
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Any, Callable, Dict, TYPE_CHECKING

import torch

from mfcl.telemetry.comms_logger import get_comms_logger
from mfcl.mixture.context import _set_active_estimator
from mfcl.moments.third import _set_active_sketch
from mfcl.utils.dist import is_main_process

from .logging import log_exception

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from mfcl.telemetry.memory import MemoryMonitor
    from mfcl.telemetry.power import PowerMonitor
    from mfcl.telemetry.timers import StepTimer
    from mfcl.telemetry.hardness import HardnessMonitor
    from .trainer import Trainer


@dataclass
class TelemetryEpochState:
    """Telemetry primitives tracked for a training epoch."""

    timer: "StepTimer | None"
    comms_logger: Any | None
    memory_monitor: "MemoryMonitor | None"
    energy_monitor: "PowerMonitor | None"
    hardness_monitor: "HardnessMonitor | None"
    mix_estimator: Any | None
    third_sketch: Any | None
    epoch_energy_start_wh: float
    epoch_energy_start_j: float
    step_energy_prev_wh: float
    fidelity_cb: Callable[..., Any] | None


@dataclass
class StepTelemetryResult:
    """Outcome of telemetry hooks executed at the end of a step."""

    step_end: float
    comm_bytes_total: float
    energy_delta_wh: float


class TelemetryManager:
    """Coordinate telemetry hooks used by the trainer."""

    def __init__(self, trainer: "Trainer") -> None:
        self._trainer = trainer

    def build_epoch_state(self) -> TelemetryEpochState:
        trainer = self._trainer
        comms_logger = get_comms_logger()
        memory_monitor = trainer.memory_monitor
        energy_monitor = trainer.energy_monitor
        hardness_monitor = trainer.hardness_monitor
        mix_estimator = (
            trainer.mixture_estimator
            if getattr(trainer.mixture_estimator, "enabled", False)
            else None
        )
        third_sketch = (
            trainer.third_moment if getattr(trainer.third_moment, "enabled", False) else None
        )
        epoch_energy_start_wh = 0.0
        epoch_energy_start_j = 0.0
        if energy_monitor is not None:
            epoch_energy_start_wh, epoch_energy_start_j = energy_monitor.get_totals()
        return TelemetryEpochState(
            timer=trainer.step_timer,
            comms_logger=comms_logger,
            memory_monitor=memory_monitor,
            energy_monitor=energy_monitor,
            hardness_monitor=hardness_monitor,
            mix_estimator=mix_estimator,
            third_sketch=third_sketch,
            epoch_energy_start_wh=epoch_energy_start_wh,
            epoch_energy_start_j=epoch_energy_start_j,
            step_energy_prev_wh=epoch_energy_start_wh,
            fidelity_cb=trainer._fidelity_callback,
        )

    def on_prepare_step(
        self,
        telemetry: TelemetryEpochState,
        *,
        epoch: int,
        step_index: int,
        global_step: int,
        data_elapsed: float,
    ) -> None:
        memory_monitor = telemetry.memory_monitor
        if memory_monitor is not None:
            memory_monitor.update_step_context(
                epoch=epoch,
                step_index=step_index,
                global_step=global_step,
            )
        energy_monitor = telemetry.energy_monitor
        if energy_monitor is not None:
            energy_monitor.update_step_context(
                epoch=epoch,
                step_index=step_index,
                global_step=global_step,
            )
        timer = telemetry.timer
        if timer is not None:
            timer.begin_step(
                epoch=epoch,
                step_index=step_index,
                global_step=global_step,
            )
            timer.record_data(data_elapsed)

        hardness_monitor = telemetry.hardness_monitor
        if hardness_monitor is not None:
            hardness_monitor.begin_step(epoch=epoch, step=global_step)

        mix_estimator = telemetry.mix_estimator
        third_sketch = telemetry.third_sketch
        if mix_estimator is not None:
            _set_active_estimator(mix_estimator)
            if third_sketch is not None:
                try:
                    stats = getattr(mix_estimator, "_last_stats", None)
                    if (
                        isinstance(stats, dict)
                        and "pi" in stats
                        and "mu" in stats
                    ):
                        pi = stats["pi"].detach().to(torch.float32)
                        mu = stats["mu"].detach().to(torch.float32)
                        if pi.ndim == 1 and mu.ndim == 2 and mu.shape[0] == pi.shape[0]:
                            global_mu = (pi.unsqueeze(1) * mu).sum(dim=0)
                            third_sketch.set_mean(global_mu.cpu())
                except Exception as exc:
                    log_exception(
                        "third_moment.set_mean",
                        exc,
                        epoch=epoch,
                        step=global_step,
                    )
        if third_sketch is not None:
            _set_active_sketch(third_sketch)

        if telemetry.comms_logger is not None:
            telemetry.comms_logger.begin_step(
                epoch=epoch,
                step_index=step_index,
                global_step=global_step,
                timer=telemetry.timer,
            )

    def on_forward_complete(
        self,
        telemetry: TelemetryEpochState,
        *,
        epoch: int,
        global_step: int,
    ) -> None:
        hardness_monitor = telemetry.hardness_monitor
        if hardness_monitor is not None:
            hardness_monitor.end_step()

        mix_estimator = telemetry.mix_estimator
        if mix_estimator is not None:
            try:
                mix_estimator.log_step(step=global_step, epoch=epoch)
            except Exception as exc:
                log_exception(
                    "mixture_estimator.log_step",
                    exc,
                    epoch=epoch,
                    step=global_step,
                )

        third_sketch = telemetry.third_sketch
        if third_sketch is not None:
            try:
                third_sketch.log_step(step=global_step, epoch=epoch)
            except Exception as exc:
                log_exception(
                    "third_moment.log_step",
                    exc,
                    epoch=epoch,
                    step=global_step,
                )

        trainer = self._trainer
        if trainer.topr_monitor is not None:
            try:
                trainer.topr_monitor.log_step(step=global_step, epoch=epoch)
            except Exception as exc:
                log_exception(
                    "topr_monitor.log_step",
                    exc,
                    epoch=epoch,
                    step=global_step,
                )

        if trainer.beta_controller is not None:
            try:
                trainer._maybe_update_beta_controller(epoch=epoch, global_step=global_step)
            except Exception as exc:
                log_exception(
                    "beta_controller.update",
                    exc,
                    epoch=epoch,
                    step=global_step,
                )

        _set_active_estimator(None)
        if telemetry.third_sketch is not None:
            _set_active_sketch(None)

    def on_step_finalize(
        self,
        telemetry: TelemetryEpochState,
        *,
        epoch: int,
        state,
        step_ctx,
        hook_metrics: Dict[str, float],
        stats: Dict[str, Any],
        loss: torch.Tensor,
        loss_scalar: float,
    ) -> StepTelemetryResult:
        trainer = self._trainer
        timer = telemetry.timer
        misc_ctx = timer.range_misc() if timer is not None else nullcontext()
        with misc_ctx:
            trainer._global_step += 1
            trainer.hooks.on_batch_end(trainer._global_step, hook_metrics)

            fidelity_cb = telemetry.fidelity_cb
            if fidelity_cb is not None:
                try:
                    fidelity_cb(
                        step=trainer._global_step,
                        epoch=epoch,
                        batch=step_ctx.batch,
                        model=trainer.method,
                    )
                except Exception as exc:
                    log_exception(
                        "fidelity_probe.maybe_log",
                        exc,
                        epoch=epoch,
                        step=trainer._global_step,
                    )

            memory_monitor = telemetry.memory_monitor
            if memory_monitor is not None:
                memory_monitor.record_step_snapshot(
                    epoch=epoch,
                    global_step=trainer._global_step,
                )

            should_tail = state.total > 0 and state.step == state.total - 1
            if is_main_process() and (
                state.step % trainer.log_interval == 0 or should_tail
            ):
                metrics: Dict[str, float] = {
                    "loss": state.loss_meter.global_avg,
                    "lr": float(state.last_lr),
                }
                if state.throughput_meter.count > 0:
                    metrics["ips"] = state.throughput_meter.global_avg
                for key in (
                    "pos_sim",
                    "neg_sim_mean",
                    "cos_sim",
                    "diag_mean",
                    "offdiag_mean",
                    "mse",
                    "std_mean",
                    "cov_offdiag",
                ):
                    value = stats.get(key)
                    if value is None:
                        continue
                    try:
                        metrics[key] = float(value.detach().to(torch.float32).item())
                    except Exception:
                        continue
                trainer.console.live(
                    epoch,
                    state.step + 1,
                    state.total,
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

        comm_bytes_total = 0.0
        if telemetry.comms_logger is not None:
            telemetry.comms_logger.end_step()
            totals = telemetry.comms_logger.pop_last_step_totals()
            if totals is not None:
                comm_bytes_total = float(totals.get("bytes_total", 0.0))

        stability = trainer.stability_sentry
        if stability is not None:
            try:
                loss_value = float(loss.detach().to(torch.float32).item())
            except Exception:
                loss_value = float("nan")
            stability.record_step(
                step=trainer._global_step,
                epoch=epoch,
                loss=loss_value,
                step_time_s=step_end - step_ctx.data_start,
                comm_bytes=comm_bytes_total,
            )

        energy_delta_wh = 0.0
        energy_monitor = telemetry.energy_monitor
        if energy_monitor is not None:
            total_wh, _ = energy_monitor.get_totals()
            energy_delta_wh = max(0.0, total_wh - telemetry.step_energy_prev_wh)
            telemetry.step_energy_prev_wh = total_wh

        return StepTelemetryResult(
            step_end=step_end,
            comm_bytes_total=comm_bytes_total,
            energy_delta_wh=energy_delta_wh,
        )

    def finalize_timer(self, telemetry: TelemetryEpochState, *, step_time: float, ips: float) -> None:
        timer = telemetry.timer
        if timer is not None:
            timer.end_step(step_time_s=step_time, ips=ips)

    def compute_epoch_energy(self, telemetry: TelemetryEpochState) -> Dict[str, float]:
        energy_monitor = telemetry.energy_monitor
        if energy_monitor is None:
            return {
                "epoch_energy_wh": 0.0,
                "epoch_energy_cost": 0.0,
                "epoch_energy_j": 0.0,
            }
        total_wh, total_j = energy_monitor.get_totals()
        energy_epoch_wh = max(0.0, total_wh - telemetry.epoch_energy_start_wh)
        energy_epoch_j = max(0.0, total_j - telemetry.epoch_energy_start_j)
        return {
            "epoch_energy_wh": energy_epoch_wh,
            "epoch_energy_cost": energy_monitor.get_epoch_cost(energy_epoch_wh),
            "epoch_energy_j": energy_epoch_j,
        }


__all__ = [
    "TelemetryManager",
    "TelemetryEpochState",
    "StepTelemetryResult",
]

