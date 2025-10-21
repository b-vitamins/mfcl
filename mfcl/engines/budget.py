"""Budget enforcement helpers for trainers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .trainer import Trainer


@dataclass
class BudgetResult:
    """Result of a budget check performed before executing a step."""

    should_stop: bool
    tokens_this_step: int


class BudgetEnforcer:
    """Encapsulate distributed budget coordination and token accounting."""

    def __init__(self, trainer: "Trainer") -> None:
        self._trainer = trainer
        self._distributed_stop = False

    def reset(self, *, initial_stop: bool = False) -> None:
        """Clear internal state prior to a new training run."""

        self._distributed_stop = bool(initial_stop)

    def sync_stop_signal(self, local_stop: bool) -> bool:
        """Synchronize ``local_stop`` across all workers."""

        stopped = bool(local_stop) or self._distributed_stop
        trainer = self._trainer
        try:
            if dist.is_available() and dist.is_initialized():
                tensor = torch.tensor(1 if stopped else 0, device=trainer.device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                stopped = bool(tensor.item() > 0)
        except Exception:
            stopped = bool(local_stop) or self._distributed_stop
        self._distributed_stop = stopped
        return stopped

    def count_tokens(self, batch: Any) -> int:
        """Estimate the number of augmented views processed in ``batch``."""

        if torch.is_tensor(batch):
            return int(batch.shape[0]) if batch.ndim > 0 else 0
        if isinstance(batch, dict):
            crops = batch.get("crops")
            if isinstance(crops, list):
                total = 0
                for tensor in crops:
                    if torch.is_tensor(tensor) and tensor.ndim > 0:
                        total += int(tensor.shape[0])
                if total > 0:
                    return total
            view_total = 0
            view_found = False
            for key, value in batch.items():
                if key.startswith("view") and torch.is_tensor(value) and value.ndim > 0:
                    view_total += int(value.shape[0])
                    view_found = True
            if view_found and view_total > 0:
                return view_total
            image = batch.get("image")
            if torch.is_tensor(image) and image.ndim > 0:
                return int(image.shape[0])
            if isinstance(image, list):
                total = 0
                for item in image:
                    if torch.is_tensor(item) and item.ndim > 0:
                        total += int(item.shape[0])
                    elif isinstance(item, (list, tuple)):
                        nested = self.count_tokens(item)
                        if nested > 0:
                            total += nested
                if total > 0:
                    return total
            for key, value in batch.items():
                if key == "index":
                    continue
                nested = self.count_tokens(value)
                if nested > 0:
                    return nested
            return 0
        if isinstance(batch, (list, tuple)):
            total = 0
            for item in batch:
                nested = self.count_tokens(item)
                if nested > 0:
                    total += nested
            return total
        return 0

    def evaluate_step_budget(self, batch: Any) -> BudgetResult:
        """Check whether executing ``batch`` would exceed the active budget."""

        tracker = getattr(self._trainer, "budget_tracker", None)
        if tracker is None:
            return BudgetResult(should_stop=False, tokens_this_step=0)

        tokens_this_step = self.count_tokens(batch)
        exceeds_budget = tracker.would_exceed(step_samples=tokens_this_step)
        should_stop = bool(exceeds_budget)
        if dist.is_available() and dist.is_initialized():
            try:
                should_stop = self.sync_stop_signal(exceeds_budget)
            except Exception:
                self._distributed_stop = self._distributed_stop or bool(exceeds_budget)
                should_stop = self._distributed_stop
        elif exceeds_budget:
            self._distributed_stop = True
        if exceeds_budget or should_stop:
            if not (dist.is_available() and dist.is_initialized()):
                self._distributed_stop = True
        return BudgetResult(should_stop=bool(exceeds_budget or should_stop), tokens_this_step=tokens_this_step)


__all__ = ["BudgetEnforcer", "BudgetResult"]

