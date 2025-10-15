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
        p = next(self.parameters(), None)

        def _first_tensor(data: Any) -> torch.Tensor | None:
            if isinstance(data, dict):
                for value in data.values():
                    ft = _first_tensor(value)
                    if ft is not None:
                        return ft
            elif torch.is_tensor(data):
                return data
            elif isinstance(data, (list, tuple)):
                for value in data:
                    ft = _first_tensor(value)
                    if ft is not None:
                        return ft
            return None

        if p is not None:
            batch_tensor = _first_tensor(batch)
            if (
                batch_tensor is not None
                and p.device != batch_tensor.device
            ):
                raise RuntimeError(
                    "Model is on {model_device} but batch tensor is on {batch_device}. "
                    "Move modules in the trainer, not inside the method.".format(
                        model_device=p.device, batch_device=batch_tensor.device
                    )
                )

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
