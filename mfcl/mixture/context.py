"""Context helpers for accessing the active mixture estimator."""

from __future__ import annotations

from typing import Optional

from .estimator import MixtureStats

_ACTIVE: Optional[MixtureStats] = None


def get_active_estimator() -> Optional[MixtureStats]:
    """Return the estimator currently attached to the training step, if any."""

    return _ACTIVE


def _set_active_estimator(estimator: Optional[MixtureStats]) -> None:
    """Set the active estimator (used internally by the trainer)."""

    global _ACTIVE
    _ACTIVE = estimator


__all__ = ["get_active_estimator", "_set_active_estimator"]
