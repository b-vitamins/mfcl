"""Checkpoint management utilities for trainers."""

from __future__ import annotations

import os
from typing import Any, Dict, TYPE_CHECKING

from mfcl.utils.checkpoint import load_checkpoint, save_checkpoint

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .trainer import Trainer


class CheckpointManager:
    """Handle save/load operations for a :class:`Trainer`."""

    def __init__(self, trainer: "Trainer") -> None:
        self._trainer = trainer

    def resume_from(self, resume_path: str | None) -> int:
        """Restore trainer state from ``resume_path`` if it exists."""

        trainer = self._trainer
        start_epoch = 1
        if resume_path and os.path.exists(resume_path):
            state = load_checkpoint(resume_path, strict=False)
            if state:
                start_epoch = max(1, int(state.get("epoch", 0)) + 1)
                self.restore_state(state)
                budget = trainer.budget_tracker
                if budget is not None:
                    budget.load(state.get("budget"))
        return start_epoch

    def save_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        trainer = self._trainer
        if not trainer.save_dir:
            return
        os.makedirs(trainer.save_dir, exist_ok=True)
        ckpt: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": trainer._global_step,
            "method": trainer.method.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": trainer.scheduler.state_dict() if trainer.scheduler else None,
            "scaler": trainer.scaler.state_dict() if trainer.scaler else None,
            "metrics": metrics,
        }
        if trainer.budget_tracker is not None:
            ckpt["budget"] = trainer.budget_tracker.snapshot()
        path = os.path.join(trainer.save_dir, f"ckpt_ep{epoch:04d}.pt")
        save_checkpoint(path, ckpt, keep_k=trainer.keep_k, make_latest=True)
        trainer.hooks.on_checkpoint(path, ckpt)

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore method/optimizer/scheduler/scaler state from a checkpoint."""

        trainer = self._trainer

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
                trainer.method.load_state_dict(method_state, strict=False)
            except Exception as exc:  # pragma: no cover - defensive
                raise RuntimeError("Failed to load method state from checkpoint") from exc

        opt_state = state.get("optimizer", {})
        _maybe("optimizer", opt_state, trainer.optimizer.load_state_dict)

        if trainer.scheduler is not None:
            sched_state = state.get("scheduler", {})
            _maybe("scheduler", sched_state, trainer.scheduler.load_state_dict)

        if trainer.scaler is not None:
            scaler_state = state.get("scaler", {})
            _maybe("scaler", scaler_state, trainer.scaler.load_state_dict)

        trainer._global_step = int(state.get("global_step", trainer._global_step))


__all__ = ["CheckpointManager"]

