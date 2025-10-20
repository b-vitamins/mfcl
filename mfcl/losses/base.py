"""Base utilities for loss modules with optional fidelity hooks."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn


def _clone_structure(obj: Any, *, require_grad: bool) -> Any:
    """Recursively clone tensors in ``obj`` with an optional grad flag."""

    if torch.is_tensor(obj):
        if require_grad:
            return obj.detach().clone().requires_grad_(True)
        return obj
    if isinstance(obj, dict):
        return {k: _clone_structure(v, require_grad=require_grad) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [_clone_structure(v, require_grad=require_grad) for v in obj]
        if isinstance(obj, tuple):
            try:
                return type(obj)(seq)
            except TypeError:
                return tuple(seq)
        return seq
    return obj


class SelfSupervisedLoss(nn.Module):
    """Common helpers shared by self-supervised loss implementations."""

    def compute_loss(
        self,
        batch: Dict[str, Any],
        model: Any,
        detach_encoder_outputs: bool = False,
        *,
        encoder_outputs: Any | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss using ``model`` projections with optional detachment."""

        if not hasattr(model, "forward_views"):
            raise TypeError("model must define forward_views for fidelity comparisons")

        outputs = encoder_outputs
        if outputs is None:
            context = torch.no_grad() if detach_encoder_outputs else nullcontext()
            with context:
                outputs = model.forward_views(batch)

        prepared = _clone_structure(outputs, require_grad=detach_encoder_outputs)
        args, kwargs = self._prepare_forward_args(prepared, batch, model)
        result = self.forward(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(
                "Loss forward must return (loss, stats) to support fidelity probes"
            )
        loss, stats = result
        extras: Dict[str, torch.Tensor]
        extras = dict(stats)
        extras.setdefault("_encoder_outputs", prepared)
        return loss, extras

    def _prepare_forward_args(
        self, outputs: Any, batch: Dict[str, Any], model: Any
    ) -> Tuple[Iterable[Any], Dict[str, Any]]:
        """Return args/kwargs for :meth:`forward` given model outputs."""

        if isinstance(outputs, tuple):
            args: Iterable[Any] = outputs
        elif isinstance(outputs, list):
            args = tuple(outputs)
        else:
            args = (outputs,)
        return args, {}


__all__ = ["SelfSupervisedLoss"]

