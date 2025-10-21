"""Context utilities for engine components."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .trainer import Trainer


_TRAINER_VAR: ContextVar["Trainer | None"] = ContextVar("mfcl_current_trainer", default=None)


def current_trainer() -> "Trainer | None":
    """Return the trainer associated with the current execution context."""

    return _TRAINER_VAR.get()


@contextmanager
def trainer_context(trainer: "Trainer") -> Iterator["Trainer"]:
    """Associate *trainer* with the current context for the duration of the block."""

    token = _TRAINER_VAR.set(trainer)
    try:
        yield trainer
    finally:
        _TRAINER_VAR.reset(token)


__all__ = ["current_trainer", "trainer_context"]

