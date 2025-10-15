"""Abstract base for self-supervised learning methods."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class BaseMethod(nn.Module):
    """Abstract base for self-supervised methods."""

    def forward_views(
        self, batch: Dict[str, Any]
    ) -> Tuple[Any, ...]:  # pragma: no cover - abstract
        """Encode required views/crops and return projected tensors.

        Args:
            batch: Collated batch dict.

        Returns:
            Tuple of projections in a method-defined order.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def compute_loss(
        self, *proj: Any, **kwargs: Any
    ) -> Dict[str, torch.Tensor]:  # pragma: no cover - abstract
        """Compute loss and aux stats from projections.

        Returns:
            Dict with key 'loss' and any additional scalars for logging.

        Raises:
            NotImplementedError.
        """
        raise NotImplementedError

    def step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """One training step.

        Behavior:
            - Calls forward_views
            - Calls compute_loss
            - Returns stats dict including 'loss'
        """
        # Ensure the module is on the same device as incoming tensors
        try:
            # Infer device from any tensor in the batch
            dev: torch.device | None = None
            if isinstance(batch, dict):
                for v in batch.values():
                    if torch.is_tensor(v):
                        dev = v.device
                        break
                    if isinstance(v, (list, tuple)):
                        for vv in v:
                            if torch.is_tensor(vv):
                                dev = vv.device
                                break
                        if dev is not None:
                            break
            if dev is not None:
                # Only move if different device type or index
                try:
                    p = next(self.parameters(), None)
                except Exception:
                    p = None
                cur_dev = getattr(p, "device", None) if p is not None else None
                if cur_dev is None:
                    try:
                        b = next(self.buffers(), None)
                    except Exception:
                        b = None
                    cur_dev = getattr(b, "device", None) if b is not None else None
                if cur_dev is None or cur_dev != dev:
                    self.to(dev)
        except Exception:
            # Best-effort; fall through if device inference/move fails
            pass

        proj = self.forward_views(batch)
        stats = self.compute_loss(*proj, batch=batch)
        if "loss" not in stats:
            raise KeyError("compute_loss must return a dict containing key 'loss'")
        return stats

    # Optional lifecycle hooks (no-ops by default)
    def on_train_start(self) -> None:  # pragma: no cover - default no-op
        return

    def on_epoch_start(self, epoch: int) -> None:  # pragma: no cover - default no-op
        return

    def on_batch_end(
        self, global_step: int
    ) -> None:  # pragma: no cover - default no-op
        return

    def on_optimizer_step(self) -> None:  # pragma: no cover - default no-op
        return


__all__ = ["BaseMethod"]
