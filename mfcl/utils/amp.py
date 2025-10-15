"""AMP helper wrapping torch.cuda.amp with safe CPU fallback.

This thin wrapper provides a consistent interface for autocast and gradient
scaling across CUDA-enabled and CPU-only environments.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, ContextManager, Dict, Optional

import torch


class AmpScaler:
    """Wrapper over ``torch.cuda.amp.GradScaler`` with CPU fallback.

    The usage pattern in a training step is:

    1. ``with scaler.autocast():`` compute loss
    2. ``scaler.scale(loss).backward()``
    3. ``scaler.unscale_(optimizer)`` then optional grad clipping
    4. ``scaler.step(optimizer); scaler.update()``
    """

    def __init__(
        self, enabled: Optional[bool] = None, init_scale: float = 2.0**16
    ) -> None:
        """Create an AMP scaler wrapper.

        Args:
            enabled: If None, enable only when CUDA is available.
            init_scale: Initial scale for GradScaler.
        """
        if enabled is None:
            enabled = torch.cuda.is_available()
        self._enabled = bool(enabled)
        # Prefer torch.amp API (device-aware); fall back to cuda.amp for older versions
        self._scaler = None
        if self._enabled:
            try:
                self._scaler = torch.amp.GradScaler("cuda", init_scale=init_scale)  # type: ignore[attr-defined]
            except Exception:
                self._scaler = torch.cuda.amp.GradScaler(init_scale=init_scale)
        else:
            self._scaler = None

    def autocast(self) -> ContextManager:
        """Return an autocast context manager (CPU-safe no-op).

        Gradient clipping should run after ``unscale_`` is called so that the
        gradients are in their true scale prior to clipping.
        """
        if self._enabled:
            try:
                return torch.amp.autocast("cuda")  # type: ignore[attr-defined]
            except Exception:
                return torch.cuda.amp.autocast()
        return nullcontext()

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss if enabled, else return as-is.

        Args:
            loss: Scalar loss tensor.

        Returns:
            Possibly scaled loss tensor.
        """
        if self._enabled and self._scaler is not None:
            return self._scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Call optimizer.step with unscale/update logic when enabled.

        Args:
            optimizer: Optimizer to step.
        """
        if self._enabled and self._scaler is not None:
            self._scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self) -> None:
        """Update scaler when enabled; no-op when disabled."""
        if self._enabled and self._scaler is not None:
            self._scaler.update()

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients in-place when enabled; no-op otherwise.

        Args:
            optimizer: Optimizer whose grads will be unscaled.
        """
        if self._enabled and self._scaler is not None:
            self._scaler.unscale_(optimizer)

    def state_dict(self) -> Dict[str, Any]:
        """Return the underlying scaler state for checkpointing.

        Returns an empty mapping when AMP is disabled so that callers can
        unconditionally serialize the result without having to branch on the
        scaler configuration.
        """

        if self._enabled and self._scaler is not None:
            return self._scaler.state_dict()
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore scaler state, tolerating disabled AMP configurations.

        When AMP is disabled ``state`` is ignored (this allows resuming a GPU
        run on CPU without errors).  If AMP is enabled but the incoming state is
        empty, the scaler is left at its default initialization.
        """

        if not state:
            return
        if not self._enabled or self._scaler is None:
            return
        self._scaler.load_state_dict(state)

    @property
    def is_enabled(self) -> bool:
        """Return True if AMP is active."""
        return self._enabled

    @property
    def scaler(self) -> Optional[Any]:
        """Return underlying GradScaler or None when disabled."""

        return self._scaler


__all__ = ["AmpScaler"]
