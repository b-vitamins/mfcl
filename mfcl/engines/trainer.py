"""Lean trainer for SSL methods with AMP, accumulation and clean logs."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Iterable
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LRScheduler as SchedulerBase  # type: ignore
else:
    from torch.optim.lr_scheduler import _LRScheduler as SchedulerBase  # type: ignore

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mfcl.engines.hooks import Hook, HookList
from typing import Protocol
from collections.abc import Iterable as _CIterable
from mfcl.utils.amp import AmpScaler
from mfcl.utils.checkpoint import save_checkpoint, load_checkpoint
from mfcl.utils.consolemonitor import ConsoleMonitor
from mfcl.utils.dist import is_main_process, reduce_dict, barrier, get_world_size
from mfcl.metrics.meter import SmoothedValue


class _TrainableMethod(Protocol):
    def to(self, device: torch.device) -> "_TrainableMethod": ...
    def train(self, mode: bool = ...) -> "_TrainableMethod": ...
    def parameters(self) -> _CIterable[nn.Parameter]: ...
    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = ...) -> Any: ...
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
        log_interval: int = 50,
        accum_steps: int = 1,
        clip_grad: Optional[float] = None,
        scheduler_step_on: str = "batch",
        guard_grad_nan: bool = False,
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
        """
        self.method: _TrainableMethod = method
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

        self.method.to(self.device)
        # Track micro-batches between optimizer steps for accumulation-invariant scheduling
        self._sched_micro_batches = 0
        # Global step counts processed micro-batches across the lifetime of the trainer
        self._global_step = 0

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
        start_epoch = 1
        if resume_path and os.path.exists(resume_path):
            state = load_checkpoint(resume_path, strict=False)
            if state:
                start_epoch = max(1, int(state.get("epoch", 0)) + 1)
                self._restore_checkpoint_state(state)

        self.method.train()
        try:
            fn: Any = getattr(self.method, "on_train_start", None)
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

        barrier()

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
        last_lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer.zero_grad(set_to_none=True)
        self._sched_micro_batches = 0
        # t0 = time.time()

        # Running sums for epoch-level reduction
        sum_loss = 0.0
        count = 0
        samples_seen = 0
        epoch_time = 0.0

        if is_main_process():
            header = ("step", "loss", "lr", "ips", "eta")
            self.console.epoch_start(epoch, total, header=header)

        for step, batch in enumerate(loader):
            batch = self.to_device(batch)
            step_start = time.time()
            with self.scaler.autocast():
                stats = self.method.step(batch)
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
            self.scaler.scale(loss_to_backward).backward()

            # Update running sums for epoch averages
            sum_loss += float(loss.detach().to(torch.float32).item())
            count += 1

            do_step = ((step + 1) % self.accum_steps) == 0
            if self.scheduler and self.scheduler_step_on == "batch":
                self._sched_micro_batches += 1
            if do_step:
                last_lr = self._apply_optimizer_step(self.accum_steps)

            # Update meters and console
            dt = time.time() - step_start
            time_meter.update(dt)
            batch_size = self._infer_batch_size(batch)
            if batch_size > 0 and dt > 0:
                throughput_meter.update(batch_size / dt)
                samples_seen += batch_size
            epoch_time += dt
            loss_meter.update(float(loss.detach().to(torch.float32).item()))

            self._global_step += 1
            hook_metrics = self._stats_to_floats(stats)
            hook_metrics.setdefault("loss", float(loss.detach().to(torch.float32).item()))
            hook_metrics.setdefault("lr", float(last_lr))
            self.hooks.on_batch_end(self._global_step, hook_metrics)

            should_tail = (total > 0 and step == total - 1)
            if is_main_process() and (
                step % self.log_interval == 0 or should_tail
            ):
                # Pull a few optional stats if present
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
                self.console.live(epoch, step + 1, total, metrics)

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

        if is_main_process():
            self.console.newline()
            summary_metrics = {
                "loss": epoch_loss,
                "lr": float(last_lr),
                "time_per_batch": time_meter.global_avg,
            }
            if epoch_time > 0 and samples_seen > 0:
                summary_metrics["imgs_per_sec"] = samples_seen / epoch_time
            self.console.summary(epoch, summary_metrics)

        return {
            "loss": float(epoch_loss),
            "lr": float(last_lr),
            "time_per_batch": float(time_meter.global_avg),
            "imgs_per_sec": float(samples_seen / epoch_time) if epoch_time > 0 else 0.0,
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
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if hasattr(self.method, "on_optimizer_step"):
            try:
                self.method.on_optimizer_step()
            except Exception:
                pass
        if self.scheduler and self.scheduler_step_on == "batch":
            # Step scheduler once per micro-batch since the last optimizer step to
            # emulate "per-batch" schedules under gradient accumulation.
            steps = int(self._sched_micro_batches)
            if steps > 0:
                for _ in range(steps):
                    self.scheduler.step()
            self._sched_micro_batches = 0
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
        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self.to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            seq = [self.to_device(x) for x in obj]
            if isinstance(obj, tuple):
                # Preserve namedtuple types if possible
                try:
                    if hasattr(obj, "_fields"):
                        return type(obj)(*seq)
                    return type(obj)(seq)
                except TypeError:
                    try:
                        return type(obj)(*seq)
                    except TypeError:
                        return tuple(seq)
            return seq
        return obj

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
