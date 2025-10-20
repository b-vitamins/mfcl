"""Training stability sentry for proactive crash diagnostics."""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Optional

import torch


class StabilityError(RuntimeError):
    """Raised when the stability sentry detects a fatal condition."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass
class StepSnapshot:
    """Lightweight record of recent training activity."""

    step: int
    epoch: int
    loss: float
    t_step_ms: float
    comm_bytes: float


def _to_cpu_payload(obj: Any) -> Any:
    """Recursively detach tensors to CPU for safe serialization."""

    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: _to_cpu_payload(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_to_cpu_payload(value) for value in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    return obj


def _summarize_optimizer(optimizer: torch.optim.Optimizer | None) -> dict[str, Any]:
    """Return a light, serializable snapshot of optimizer hyper-parameters."""

    if optimizer is None:
        return {}
    summary: dict[str, Any] = {"param_groups": []}
    for group in optimizer.param_groups:
        filtered = {k: v for k, v in group.items() if k != "params"}
        summary["param_groups"].append(filtered)
    state_summary: dict[str, Any] = {}
    for idx, (param, state) in enumerate(optimizer.state.items()):
        key = str(idx)
        payload: dict[str, Any] = {}
        for name, value in state.items():
            if torch.is_tensor(value):
                payload[name] = {
                    "dtype": str(value.dtype),
                    "shape": tuple(value.shape),
                    "norm": float(value.norm().detach().cpu().item())
                    if value.numel() > 0
                    else 0.0,
                }
            else:
                payload[name] = value
        state_summary[key] = payload
    if state_summary:
        summary["state"] = state_summary
    return summary


class StabilitySentry:
    """Monitor loss/gradient stability and write diagnostics on failure."""

    def __init__(
        self,
        *,
        enabled: bool,
        save_dir: str | Path | None,
        is_main: bool,
        max_history: int = 100,
    ) -> None:
        self.enabled = bool(enabled and is_main)
        self.max_history = max(1, int(max_history))
        self._history: Deque[StepSnapshot] = deque(maxlen=self.max_history)
        self._save_dir = Path(save_dir) if save_dir is not None else None
        if self._save_dir is None:
            self._save_dir = Path.cwd()
        self._crash_dir = self._save_dir / "crash"

    def record_step(
        self,
        *,
        step: int,
        epoch: int,
        loss: float,
        step_time_s: float,
        comm_bytes: float,
    ) -> None:
        if not self.enabled:
            return
        snapshot = StepSnapshot(
            step=int(step),
            epoch=int(epoch),
            loss=float(loss),
            t_step_ms=float(step_time_s) * 1000.0,
            comm_bytes=float(comm_bytes),
        )
        self._history.append(snapshot)

    def check_loss(
        self,
        loss: torch.Tensor | float,
        *,
        batch: Any | None,
        optimizer: torch.optim.Optimizer | None,
        scaler: Any | None,
        step: int,
        epoch: int,
    ) -> None:
        if not self.enabled:
            return
        is_finite = True
        if torch.is_tensor(loss):
            detached = loss.detach()
            try:
                flag = torch.isfinite(detached)
                if flag.numel() == 1:
                    is_finite = bool(flag.item())
                else:
                    is_finite = bool(flag.all().item())
            except Exception:
                is_finite = torch.isfinite(detached).all().item()
        else:
            try:
                is_finite = bool(torch.isfinite(torch.tensor(float(loss))).item())
            except Exception:
                is_finite = True
        if not is_finite:
            raise self._trigger_failure(
                reason="loss_non_finite",
                batch=batch,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                epoch=epoch,
                extra={"loss_repr": repr(loss)},
            )

    def check_gradients(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        batch: Any | None,
        step: int,
        epoch: int,
    ) -> None:
        if not self.enabled:
            return
        offenders: list[dict[str, Any]] = []
        for group_index, group in enumerate(optimizer.param_groups):
            for param_index, param in enumerate(group.get("params", [])):
                if param is None or param.grad is None:
                    continue
                grad = param.grad.detach()
                if torch.any(torch.isnan(grad)) or torch.any(torch.isinf(grad)):
                    offenders.append(
                        {
                            "group": group_index,
                            "param_index": param_index,
                            "shape": tuple(grad.shape),
                        }
                    )
                    break
        if offenders:
            raise self._trigger_failure(
                reason="grad_non_finite",
                batch=batch,
                optimizer=optimizer,
                scaler=None,
                step=step,
                epoch=epoch,
                extra={"offenders": offenders},
            )

    def _trigger_failure(
        self,
        *,
        reason: str,
        batch: Any | None,
        optimizer: torch.optim.Optimizer | None,
        scaler: Any | None,
        step: int,
        epoch: int,
        extra: Optional[dict[str, Any]] = None,
    ) -> StabilityError:
        if not self.enabled:
            return StabilityError(reason, f"{reason} detected at step {step}")
        self._crash_dir.mkdir(parents=True, exist_ok=True)
        batch_path = self._crash_dir / "last_batch.pt"
        if batch is not None:
            try:
                torch.save({"batch": _to_cpu_payload(batch)}, batch_path)
            except Exception:
                # Best effort save; continue with report generation.
                pass

        amp_scale: float | None = None
        if scaler is not None and hasattr(scaler, "scaler"):
            try:
                raw = scaler.scaler
                if raw is not None and hasattr(raw, "get_scale"):
                    amp_scale = float(raw.get_scale())
            except Exception:
                amp_scale = None

        history_payload = [snapshot.__dict__ for snapshot in self._history]
        report: dict[str, Any] = {
            "reason": reason,
            "step": int(step),
            "epoch": int(epoch),
            "timestamp": time.time(),
            "amp_scale": amp_scale,
            "history": history_payload,
            "optimizer": _summarize_optimizer(optimizer),
        }
        if extra:
            report.update(extra)
        report_path = self._crash_dir / "stability_report.json"
        try:
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        except Exception:
            pass
        message = f"Stability failure ({reason}) at step={step} epoch={epoch}"
        return StabilityError(reason, message)


__all__ = ["StabilitySentry", "StabilityError"]
