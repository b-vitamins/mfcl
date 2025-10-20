"""Budget tracking for iso-time/token/epoch experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


_SUPPORTED_MODES = {"iso_time", "iso_tokens", "iso_epochs", "comm_cap", "energy_cap"}


@dataclass
class _BudgetTotals:
    tokens: float = 0.0
    time_ms: float = 0.0
    comm_bytes: float = 0.0
    energy_Wh: float = 0.0
    steps: int = 0

    def to_dict(self, *, steps_per_epoch: float | None) -> Dict[str, float]:
        epochs = 0.0
        if steps_per_epoch and steps_per_epoch > 0:
            epochs = self.steps / steps_per_epoch
        return {
            "tokens": float(self.tokens),
            "time_ms": float(self.time_ms),
            "time_minutes": float(self.time_ms / 60000.0),
            "comm_bytes": float(self.comm_bytes),
            "energy_Wh": float(self.energy_Wh),
            "steps": int(self.steps),
            "epochs": float(epochs),
        }


class BudgetTracker:
    """Accumulate runtime budgets and check termination conditions."""

    def __init__(self, mode: str, limits: Dict) -> None:
        if mode not in _SUPPORTED_MODES:
            raise ValueError(f"Unsupported budget mode: {mode}")
        self.mode = str(mode)
        self._limits: Dict[str, float] = {}
        for key, value in (limits or {}).items():
            if value is None:
                continue
            if key == "steps_per_epoch":
                try:
                    self._steps_per_epoch = float(value)
                except (TypeError, ValueError):
                    raise ValueError("steps_per_epoch must be a positive number") from None
                continue
            try:
                self._limits[key] = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"Budget limit '{key}' must be numeric") from None
        if not hasattr(self, "_steps_per_epoch"):
            self._steps_per_epoch: Optional[float] = None
        if self.mode == "iso_epochs":
            if self._steps_per_epoch is None or self._steps_per_epoch <= 0:
                raise ValueError(
                    "iso_epochs mode requires a positive steps_per_epoch limit"
                )
            if "max_epochs" not in self._limits:
                raise ValueError("iso_epochs mode requires 'max_epochs' in limits")
        if self.mode == "iso_time" and "max_minutes" not in self._limits:
            raise ValueError("iso_time mode requires 'max_minutes' in limits")
        if self.mode == "iso_tokens" and "max_tokens" not in self._limits:
            raise ValueError("iso_tokens mode requires 'max_tokens' in limits")
        if self.mode == "comm_cap" and "max_comm_bytes" not in self._limits:
            raise ValueError("comm_cap mode requires 'max_comm_bytes' in limits")
        if self.mode == "energy_cap" and "max_energy_Wh" not in self._limits:
            raise ValueError("energy_cap mode requires 'max_energy_Wh' in limits")
        self._totals = _BudgetTotals()

    # ------------------------------------------------------------------
    # Properties and helpers
    # ------------------------------------------------------------------
    @property
    def totals(self) -> Dict[str, float]:
        """Return a copy of cumulative totals."""

        return self._totals.to_dict(steps_per_epoch=self._steps_per_epoch)

    def _limit(self, name: str) -> Optional[float]:
        value = self._limits.get(name)
        if value is None:
            return None
        return float(value)

    def _epochs_from_steps(self, steps: int | float) -> float:
        if self._steps_per_epoch and self._steps_per_epoch > 0:
            return float(steps) / self._steps_per_epoch
        return 0.0

    def _compare(self, *, strict: bool, totals: _BudgetTotals) -> bool:
        cmp = (lambda value, limit: value > limit) if strict else (
            lambda value, limit: value >= limit
        )

        if self.mode == "iso_time":
            limit = self._limit("max_minutes")
            if limit is not None and cmp(totals.time_ms / 60000.0, limit):
                return True
        if self.mode == "iso_tokens":
            limit = self._limit("max_tokens")
            if limit is not None and cmp(totals.tokens, limit):
                return True
        if self.mode == "iso_epochs":
            limit = self._limit("max_epochs")
            if limit is not None and cmp(self._epochs_from_steps(totals.steps), limit):
                return True
        if self.mode == "comm_cap":
            limit = self._limit("max_comm_bytes")
            if limit is not None and cmp(totals.comm_bytes, limit):
                return True
        if self.mode == "energy_cap":
            limit = self._limit("max_energy_Wh")
            if limit is not None and cmp(totals.energy_Wh, limit):
                return True

        # Always enforce optional caps even when not the active mode.
        cap = self._limit("max_comm_bytes")
        if cap is not None and cmp(totals.comm_bytes, cap):
            return True
        cap = self._limit("max_energy_Wh")
        if cap is not None and cmp(totals.energy_Wh, cap):
            return True
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(
        self,
        step_samples: int,
        step_wall_ms: float,
        comm_bytes: int = 0,
        energy_Wh: float = 0.0,
    ) -> None:
        """Accumulate usage for a completed step."""

        if step_samples < 0:
            raise ValueError("step_samples must be non-negative")
        if step_wall_ms < 0:
            raise ValueError("step_wall_ms must be non-negative")
        if comm_bytes < 0:
            raise ValueError("comm_bytes must be non-negative")
        if energy_Wh < 0:
            raise ValueError("energy_Wh must be non-negative")

        self._totals.tokens += float(step_samples)
        self._totals.time_ms += float(step_wall_ms)
        self._totals.comm_bytes += float(comm_bytes)
        self._totals.energy_Wh += float(energy_Wh)
        self._totals.steps += 1

    def should_stop(self) -> bool:
        """Return True when the active budget has been exhausted."""

        return self._compare(strict=False, totals=self._totals)

    def would_exceed(
        self,
        step_samples: int = 0,
        step_wall_ms: float = 0.0,
        comm_bytes: int = 0,
        energy_Wh: float = 0.0,
    ) -> bool:
        """Predict whether adding the provided usage would exceed limits."""

        if step_samples < 0 or step_wall_ms < 0 or comm_bytes < 0 or energy_Wh < 0:
            raise ValueError("Predicted usage must be non-negative")
        prospective = _BudgetTotals(
            tokens=self._totals.tokens + float(step_samples),
            time_ms=self._totals.time_ms + float(step_wall_ms),
            comm_bytes=self._totals.comm_bytes + float(comm_bytes),
            energy_Wh=self._totals.energy_Wh + float(energy_Wh),
            steps=self._totals.steps + 1,
        )
        return self._compare(strict=True, totals=prospective)

    def snapshot(self) -> Dict[str, object]:
        """Return a JSON-serialisable representation of the tracker state."""

        payload: Dict[str, object] = {
            "mode": self.mode,
            "limits": dict(self._limits),
            "totals": self._totals.to_dict(steps_per_epoch=self._steps_per_epoch),
            "steps_per_epoch": self._steps_per_epoch,
        }
        return payload

    def load(self, snapshot: Dict[str, object] | None) -> None:
        """Restore tracker state from :meth:`snapshot`."""

        if not snapshot:
            return
        snap_mode = snapshot.get("mode")
        if snap_mode and snap_mode != self.mode:
            raise ValueError(
                f"Snapshot mode {snap_mode} does not match tracker mode {self.mode}"
            )
        steps_per_epoch = snapshot.get("steps_per_epoch")
        if steps_per_epoch is not None and self._steps_per_epoch is None:
            try:
                self._steps_per_epoch = float(steps_per_epoch)
            except (TypeError, ValueError):
                pass
        totals = snapshot.get("totals")
        if isinstance(totals, dict):
            self._totals.tokens = float(totals.get("tokens", self._totals.tokens))
            self._totals.time_ms = float(totals.get("time_ms", self._totals.time_ms))
            self._totals.comm_bytes = float(
                totals.get("comm_bytes", self._totals.comm_bytes)
            )
            self._totals.energy_Wh = float(
                totals.get("energy_Wh", self._totals.energy_Wh)
            )
            steps_val = totals.get("steps")
            if steps_val is not None:
                try:
                    self._totals.steps = int(steps_val)
                except (TypeError, ValueError):
                    pass
        # Ensure totals dict stays consistent when keys missing
        self._totals = _BudgetTotals(
            tokens=self._totals.tokens,
            time_ms=self._totals.time_ms,
            comm_bytes=self._totals.comm_bytes,
            energy_Wh=self._totals.energy_Wh,
            steps=self._totals.steps,
        )


__all__ = ["BudgetTracker"]
