"""Lean trainer for SSL methods with AMP, accumulation and clean logs."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Dict, Optional, Iterable, Callable
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LRScheduler as SchedulerBase  # type: ignore
    from mfcl.telemetry.hardness import HardnessMonitor
else:
    from torch.optim.lr_scheduler import _LRScheduler as SchedulerBase  # type: ignore

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

from mfcl.engines.hooks import Hook, HookList
from typing import Protocol
from collections.abc import Iterable as _CIterable
from mfcl.telemetry.timers import StepTimer
from mfcl.telemetry.memory import MemoryMonitor
from mfcl.telemetry.power import PowerMonitor
from mfcl.telemetry.stability import StabilitySentry
from mfcl.runtime.budget import BudgetTracker
from mfcl.runtime.beta_ctrl import BetaController
from mfcl.utils.amp import AmpScaler
from mfcl.utils.consolemonitor import ConsoleMonitor
from mfcl.utils.dist import (
    barrier,
    get_world_size,
    is_main_process,
    unwrap_ddp,
)

from .budget import BudgetEnforcer
from .checkpointing import CheckpointManager
from .context import trainer_context
from .logging import log_exception
from .step_executor import StepExecutor
from .telemetry import TelemetryManager


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
        budget_tracker: Optional[BudgetTracker] = None,
        fidelity_probe: Any | None = None,
        hardness_monitor: "HardnessMonitor | None" = None,
        stability_sentry: StabilitySentry | None = None,
        mixture_estimator: Any | None = None,
        topr_monitor: Any | None = None,
        beta_controller: Optional[BetaController] = None,
        third_moment_sketch: Any | None = None,
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
            budget_tracker: Optional BudgetTracker enforcing runtime limits.
            hardness_monitor: Optional HardnessMonitor for top-K negative tracking.
            stability_sentry: Optional StabilitySentry for crash diagnostics.
            mixture_estimator: Optional MixtureStats instance for diagnostics logging.
            topr_monitor: Optional Top-R diagnostics tracker.
            third_moment_sketch: Optional ThirdMomentSketch for κ₃ diagnostics.
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
        self.budget_tracker = budget_tracker
        self.fidelity_probe = fidelity_probe
        self._fidelity_callback = (
            getattr(fidelity_probe, "maybe_log", None) if fidelity_probe is not None else None
        )
        self.hardness_monitor = hardness_monitor
        self.stability_sentry = stability_sentry
        self.mixture_estimator = mixture_estimator
        self.topr_monitor = topr_monitor
        self.third_moment = third_moment_sketch
        self.beta_controller = beta_controller
        try:
            env_steps = int(os.environ.get("MFCL_OOM_SEARCH_MAX_STEPS", "0"))
        except Exception:
            env_steps = 0
        self._max_steps_override = max(0, env_steps)

        self._telemetry_manager = TelemetryManager(self)
        self._budget_enforcer = BudgetEnforcer(self)
        self._checkpoint_manager = CheckpointManager(self)
        self._step_executor = StepExecutor(
            self,
            telemetry=self._telemetry_manager,
            budget=self._budget_enforcer,
        )

        base_method = unwrap_ddp(self.method)
        base_method.to(self.device)
        if self.channels_last_inputs:
            try:
                base_method.to(memory_format=torch.channels_last)
            except (TypeError, RuntimeError) as exc:
                log_exception(
                    "method.to(memory_format=torch.channels_last)",
                    exc,
                )
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
        del val_loader  # unused, kept for API compatibility

        with trainer_context(self):
            start_epoch = self._checkpoint_manager.resume_from(resume_path)

            self.method.train()
            try:
                fn: Any = getattr(self._method_impl, "on_train_start", None)
                if callable(fn):
                    fn()
            except Exception as exc:
                log_exception(
                    "hook 'method.on_train_start'",
                    exc,
                    epoch=start_epoch,
                )

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

            initial_stop = (
                self.budget_tracker.should_stop() if self.budget_tracker is not None else False
            )
            self._budget_enforcer.reset(initial_stop=initial_stop)

            for epoch in range(start_epoch, epochs + 1):
                if self.budget_tracker is not None and self._budget_enforcer.sync_stop_signal(
                    self.budget_tracker.should_stop()
                ):
                    break
                train_state["epoch"] = epoch
                train_state["global_step"] = self._global_step
                self.hooks.on_epoch_start(epoch, train_state)
                epoch_metrics = self.train_one_epoch(epoch, train_loader)
                if "epoch" not in epoch_metrics:
                    epoch_metrics = dict(epoch_metrics)
                    epoch_metrics["epoch"] = epoch
                train_state["global_step"] = self._global_step

                if is_main_process() and self.save_dir and (epoch % save_every == 0):
                    self._checkpoint_manager.save_epoch(epoch, epoch_metrics)

                if epoch % eval_every == 0:
                    if self.budget_tracker is None or not self.budget_tracker.should_stop():
                        self.hooks.on_eval_end(epoch_metrics)

                if self.scheduler and self.scheduler_step_on == "epoch":
                    self.scheduler.step()

                if self._max_steps_override and self._global_step >= self._max_steps_override:
                    break

            barrier()


    def train_one_epoch(
        self, epoch: int, loader: DataLoader | Iterable[Any]
    ) -> Dict[str, float]:
        """Run one training epoch.

        Returns:
            Dict of averaged metrics for the epoch (includes 'loss' and 'lr').
        """

        state = self._step_executor.start_epoch(epoch, loader)

        while True:
            step_ctx = self._step_executor.prepare_step(epoch, state)
            if step_ctx is None:
                break

            stats, loss = self._step_executor.execute_step(epoch, state, step_ctx)
            loss_scalar = self._step_executor.backward_and_update(
                epoch, state, step_ctx, stats, loss
            )

            if self._step_executor.finalize_step(
                epoch, state, step_ctx, stats, loss, loss_scalar
            ):
                break

        return self._step_executor.finish_epoch(epoch, state)


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


    def _beta_ctrl_raw(self) -> float:
        controller = self.beta_controller
        base = controller.beta_min if controller is not None else 0.0
        method_impl = self._method_impl
        for attr in ("mixture_beta_raw", "mixture_beta", "beta"):
            if hasattr(method_impl, attr):
                value = getattr(method_impl, attr)
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        if controller is not None:
            if controller.last_beta_raw is not None:
                return float(controller.last_beta_raw)
            if controller.last_beta is not None:
                return float(controller.last_beta)
            return float(controller.beta_max)
        return float(base)

    def _apply_beta_to_method(
        self,
        beta_value: float,
        *,
        epoch: int | None = None,
        step: int | None = None,
    ) -> None:
        method_impl = self._method_impl
        if hasattr(method_impl, "set_mixture_beta"):
            try:
                method_impl.set_mixture_beta(float(beta_value))
                return
            except Exception as exc:
                log_exception(
                    "method.set_mixture_beta",
                    exc,
                    epoch=epoch,
                    step=step,
                )
        try:
            setattr(method_impl, "mixture_beta", float(beta_value))
        except Exception as exc:
            log_exception(
                "method.setattr(mixture_beta)",
                exc,
                epoch=epoch,
                step=step,
            )

    def _current_beta_value(
        self,
        *,
        epoch: int | None = None,
        step: int | None = None,
    ) -> float:
        method_impl = self._method_impl
        for attr in ("mixture_beta", "beta", "mixture_beta_raw"):
            if hasattr(method_impl, attr):
                value = getattr(method_impl, attr)
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        controller = self.beta_controller
        if controller is not None and controller.last_beta is not None:
            try:
                return float(controller.last_beta)
            except (TypeError, ValueError) as exc:
                log_exception(
                    "beta_controller.last_beta",
                    exc,
                    epoch=epoch,
                    step=step,
                )
        return 0.0

    def _maybe_update_topr(self, *, epoch: int, global_step: int) -> None:
        monitor = self.topr_monitor
        estimator = self.mixture_estimator
        if monitor is None or estimator is None:
            return
        stats = getattr(estimator, "_last_stats", None)
        if not isinstance(stats, dict) or not stats:
            return
        pi = stats.get("pi")
        mu = stats.get("mu")
        sigma = stats.get("Sigma")
        if pi is None or mu is None or sigma is None:
            return
        responsibilities = stats.get("R")
        Q = stats.get("Q") if isinstance(stats.get("Q"), torch.Tensor) else None
        beta_value = self._current_beta_value(epoch=epoch, step=global_step)
        timer = self.step_timer
        ctx = timer.range_topr() if timer is not None else nullcontext()
        with ctx:
            monitor.update(
                responsibilities=responsibilities,
                pi=pi,
                mu=mu,
                Sigma=sigma,
                beta=beta_value,
                Q=Q,
            )

    def _maybe_update_beta_controller(self, *, epoch: int, global_step: int) -> None:
        controller = self.beta_controller
        estimator = self.mixture_estimator
        if controller is None or estimator is None:
            return
        stats = getattr(estimator, "_last_stats", None)
        if not isinstance(stats, dict) or not stats:
            return
        payload: Dict[str, Any] = {}
        for key in ("pi", "pi_min", "median_xBx"):
            if key in stats:
                payload[key] = stats[key]
        delta_sigma = stats.get("delta_sigma_max")
        beta_raw = self._beta_ctrl_raw()
        timer = self.step_timer
        ctx = timer.range_beta_ctrl() if timer is not None else nullcontext()
        with ctx:
            beta_value, info = controller.step(payload, delta_sigma, beta_raw)
        if dist.is_available() and dist.is_initialized():
            try:
                tensor = torch.tensor(
                    [beta_value], device=self.device, dtype=torch.float32
                )
                dist.broadcast(tensor, src=0)
                beta_value = float(tensor.item())
                info["beta_applied"] = beta_value
                controller.apply_broadcast(beta_value, info)
            except Exception as exc:
                log_exception(
                    "beta_controller.broadcast",
                    exc,
                    epoch=epoch,
                    step=global_step,
                )
        info["beta_raw"] = beta_raw
        controller.log_step(step=global_step, epoch=epoch, info=info)
        self._apply_beta_to_method(
            beta_value,
            epoch=epoch,
            step=global_step,
        )

    def _apply_optimizer_step(
        self,
        micro_batches_in_step: int,
        *,
        epoch: int | None,
        global_step: int | None,
        batch: Any | None,
    ) -> float:
        """Apply an optimizer step handling AMP, clipping and scheduling."""
        self.scaler.unscale_(self.optimizer)
        if micro_batches_in_step <= 0:
            raise ValueError("micro_batches_in_step must be positive")
        if micro_batches_in_step != self.accum_steps:
            scale = self.accum_steps / float(micro_batches_in_step)
            for group in self.optimizer.param_groups:
                for p in group.get("params", []):
                    if p.grad is not None:
                        p.grad.detach().mul_(scale)
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(
                self.method.parameters(), max_norm=float(self.clip_grad)
            )
        if self.stability_sentry is not None:
            self.stability_sentry.check_gradients(
                self.optimizer,
                batch=batch,
                step=global_step if global_step is not None else 0,
                epoch=epoch if epoch is not None else 0,
            )
        elif self.guard_grad_nan:
            total_norm = 0.0
            for group in self.optimizer.param_groups:
                for p in group.get("params", []):
                    if p.grad is not None:
                        v = p.grad.detach()
                        if torch.any(torch.isnan(v)) or torch.any(torch.isinf(v)):
                            raise RuntimeError("Detected NaN/Inf in gradients")
                        total_norm += float(v.norm(2).detach().item() ** 2)
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
            except Exception as exc:
                log_exception(
                    "hook 'method.on_optimizer_step'",
                    exc,
                    epoch=epoch,
                    step=global_step,
                )
        if self.scheduler and self.scheduler_step_on == "batch":
            did_step = True
            if prev_scale is not None and hasattr(self.scaler, "scaler") and self.scaler.scaler is not None:
                try:
                    new_scale = float(self.scaler.scaler.get_scale())
                    if new_scale < float(prev_scale):
                        did_step = False
                except Exception:
                    did_step = True
            if did_step:
                self.scheduler.step()
        return self.optimizer.param_groups[0]["lr"]


    def _beta_ctrl_raw(self) -> float:
        controller = self.beta_controller
        base = controller.beta_min if controller is not None else 0.0
        method_impl = self._method_impl
        for attr in ("mixture_beta_raw", "mixture_beta", "beta"):
            if hasattr(method_impl, attr):
                value = getattr(method_impl, attr)
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        if controller is not None:
            if controller.last_beta_raw is not None:
                return float(controller.last_beta_raw)
            if controller.last_beta is not None:
                return float(controller.last_beta)
            return float(controller.beta_max)
        return float(base)

    def _apply_beta_to_method(
        self,
        beta_value: float,
        *,
        epoch: int | None = None,
        step: int | None = None,
    ) -> None:
        method_impl = self._method_impl
        if hasattr(method_impl, "set_mixture_beta"):
            try:
                method_impl.set_mixture_beta(float(beta_value))
                return
            except Exception as exc:
                log_exception(
                    "method.set_mixture_beta",
                    exc,
                    epoch=epoch,
                    step=step,
                )
        try:
            setattr(method_impl, "mixture_beta", float(beta_value))
        except Exception as exc:
            log_exception(
                "method.setattr(mixture_beta)",
                exc,
                epoch=epoch,
                step=step,
            )

    def _current_beta_value(
        self,
        *,
        epoch: int | None = None,
        step: int | None = None,
    ) -> float:
        method_impl = self._method_impl
        for attr in ("mixture_beta", "beta", "mixture_beta_raw"):
            if hasattr(method_impl, attr):
                value = getattr(method_impl, attr)
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        controller = self.beta_controller
        if controller is not None and controller.last_beta is not None:
            try:
                return float(controller.last_beta)
            except (TypeError, ValueError) as exc:
                log_exception(
                    "beta_controller.last_beta",
                    exc,
                    epoch=epoch,
                    step=step,
                )
        return 0.0

    def _maybe_update_topr(self, *, epoch: int, global_step: int) -> None:
        monitor = self.topr_monitor
        estimator = self.mixture_estimator
        if monitor is None or estimator is None:
            return
        stats = getattr(estimator, "_last_stats", None)
        if not isinstance(stats, dict) or not stats:
            return
        pi = stats.get("pi")
        mu = stats.get("mu")
        sigma = stats.get("Sigma")
        if pi is None or mu is None or sigma is None:
            return
        responsibilities = stats.get("R")
        Q = stats.get("Q") if isinstance(stats.get("Q"), torch.Tensor) else None
        beta_value = self._current_beta_value(epoch=epoch, step=global_step)
        timer = self.step_timer
        ctx = timer.range_topr() if timer is not None else nullcontext()
        with ctx:
            monitor.update(
                responsibilities=responsibilities,
                pi=pi,
                mu=mu,
                Sigma=sigma,
                beta=beta_value,
                Q=Q,
            )

    def _maybe_update_beta_controller(self, *, epoch: int, global_step: int) -> None:
        controller = self.beta_controller
        estimator = self.mixture_estimator
        if controller is None or estimator is None:
            return
        stats = getattr(estimator, "_last_stats", None)
        if not isinstance(stats, dict) or not stats:
            return
        payload: Dict[str, Any] = {}
        for key in ("pi", "pi_min", "median_xBx"):
            if key in stats:
                payload[key] = stats[key]
        delta_sigma = stats.get("delta_sigma_max")
        beta_raw = self._beta_ctrl_raw()
        timer = self.step_timer
        ctx = timer.range_beta_ctrl() if timer is not None else nullcontext()
        with ctx:
            beta_value, info = controller.step(payload, delta_sigma, beta_raw)
        if dist.is_available() and dist.is_initialized():
            try:
                tensor = torch.tensor(
                    [beta_value], device=self.device, dtype=torch.float32
                )
                dist.broadcast(tensor, src=0)
                beta_value = float(tensor.item())
                info["beta_applied"] = beta_value
                controller.apply_broadcast(beta_value, info)
            except Exception as exc:
                log_exception(
                    "beta_controller.broadcast",
                    exc,
                    epoch=epoch,
                    step=global_step,
                )
        info["beta_raw"] = beta_raw
        controller.log_step(step=global_step, epoch=epoch, info=info)
        self._apply_beta_to_method(
            beta_value,
            epoch=epoch,
            step=global_step,
        )

    def _apply_optimizer_step(
        self,
        micro_batches_in_step: int,
        *,
        epoch: int | None,
        global_step: int | None,
        batch: Any | None,
    ) -> float:
        """Apply an optimizer step handling AMP, clipping and scheduling."""
        self.scaler.unscale_(self.optimizer)
        if micro_batches_in_step <= 0:
            raise ValueError("micro_batches_in_step must be positive")
        if micro_batches_in_step != self.accum_steps:
            scale = self.accum_steps / float(micro_batches_in_step)
            for group in self.optimizer.param_groups:
                for p in group.get("params", []):
                    if p.grad is not None:
                        p.grad.detach().mul_(scale)
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(
                self.method.parameters(), max_norm=float(self.clip_grad)
            )
        if self.stability_sentry is not None:
            self.stability_sentry.check_gradients(
                self.optimizer,
                batch=batch,
                step=global_step if global_step is not None else 0,
                epoch=epoch if epoch is not None else 0,
            )
        elif self.guard_grad_nan:
            total_norm = 0.0
            for group in self.optimizer.param_groups:
                for p in group.get("params", []):
                    if p.grad is not None:
                        v = p.grad.detach()
                        if torch.any(torch.isnan(v)) or torch.any(torch.isinf(v)):
                            raise RuntimeError("Detected NaN/Inf in gradients")
                        total_norm += float(v.norm(2).detach().item() ** 2)
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
            except Exception as exc:
                log_exception(
                    "hook 'method.on_optimizer_step'",
                    exc,
                    epoch=epoch,
                    step=global_step,
                )
        if self.scheduler and self.scheduler_step_on == "batch":
            did_step = True
            if prev_scale is not None and hasattr(self.scaler, "scaler") and self.scaler.scaler is not None:
                try:
                    new_scale = float(self.scaler.scaler.get_scale())
                    if new_scale < float(prev_scale):
                        did_step = False
                except Exception:
                    did_step = True
            if did_step:
                self.scheduler.step()
        return self.optimizer.param_groups[0]["lr"]



__all__ = ["Trainer"]
