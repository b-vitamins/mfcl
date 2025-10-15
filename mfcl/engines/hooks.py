"""Lightweight lifecycle hooks to extend trainer behavior without bloat."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


class Hook:
    """Base hook with no-op lifecycle methods."""

    def on_train_start(
        self, state: Dict[str, Any]
    ) -> None:  # pragma: no cover - default no-op
        return

    def on_epoch_start(
        self, epoch: int, state: Dict[str, Any]
    ) -> None:  # pragma: no cover - default no-op
        return

    def on_batch_end(
        self, step: int, metrics: Dict[str, float]
    ) -> None:  # pragma: no cover - default no-op
        return

    def on_eval_end(
        self, metrics: Dict[str, float]
    ) -> None:  # pragma: no cover - default no-op
        return

    def on_checkpoint(
        self, path: str, state: Dict[str, Any]
    ) -> None:  # pragma: no cover - default no-op
        return


class HookList(Hook):
    """Composite that forwards to a list of hooks in order."""

    def __init__(self, hooks: Iterable[Hook] | None = None) -> None:
        self._hooks: List[Hook] = list(hooks) if hooks is not None else []

    def add(self, hook: Hook) -> None:
        """Append a hook to the list in order."""
        self._hooks.append(hook)

    # Exception-safe fan-out to children
    def on_train_start(self, state: Dict[str, Any]) -> None:
        for h in self._hooks:
            try:
                h.on_train_start(state)
            except Exception:
                pass

    def on_epoch_start(self, epoch: int, state: Dict[str, Any]) -> None:
        for h in self._hooks:
            try:
                h.on_epoch_start(epoch, state)
            except Exception:
                pass

    def on_batch_end(self, step: int, metrics: Dict[str, float]) -> None:
        for h in self._hooks:
            try:
                h.on_batch_end(step, metrics)
            except Exception:
                pass

    def on_eval_end(self, metrics: Dict[str, float]) -> None:
        for h in self._hooks:
            try:
                h.on_eval_end(metrics)
            except Exception:
                pass

    def on_checkpoint(self, path: str, state: Dict[str, Any]) -> None:
        for h in self._hooks:
            try:
                h.on_checkpoint(path, state)
            except Exception:
                pass


__all__ = ["Hook", "HookList"]
