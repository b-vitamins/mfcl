"""Shared logging helpers for engine components."""

from __future__ import annotations

import logging
from typing import Any


LOGGER = logging.getLogger(__name__)


def log_exception(
    source: str,
    exc: Exception,
    *,
    epoch: int | None = None,
    step: int | None = None,
) -> None:
    """Emit a warning with contextual epoch/step information."""

    context = source
    details: list[str] = []
    if epoch is not None:
        details.append(f"epoch={epoch}")
    if step is not None:
        details.append(f"step={step}")
    if details:
        context = f"{context} ({', '.join(details)})"
    LOGGER.warning("Exception raised during %s", context, exc_info=exc)


__all__ = ["log_exception", "LOGGER"]

