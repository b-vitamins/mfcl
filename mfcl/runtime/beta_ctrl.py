"""Mixture inflation bound controller for beta parameter."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

import torch


@dataclass(frozen=True)
class BetaControllerResult:
    """Structured result returned by :meth:`BetaController.step`."""

    beta: float
    info: Dict[str, float | str]


def estimate_mixture_inflation(beta_abs: float, scale: float, delta_sigma: float) -> float:
    """Return the quadratic mixture inflation estimate ``ε_mix``.

    The controller upper-bounds the mixture inflation using the approximation

    .. math::

        ε_mix(β) = β \cdot s + \tfrac{1}{2} β^2 δσ,

    where ``s`` captures the linear contribution from mixture overlap and
    ``δσ`` measures the curvature induced by covariance mismatches.  The
    function operates on ``|β|`` because only the magnitude influences the
    inflation.
    """

    beta_abs = max(0.0, float(beta_abs))
    scale = max(0.0, float(scale))
    delta_sigma = max(0.0, float(delta_sigma))
    return beta_abs * scale + 0.5 * beta_abs * beta_abs * delta_sigma


def solve_beta_for_mixture_inflation(
    target_eps: float, *, scale: float, delta_sigma: float
) -> float:
    """Solve ``ε_mix(β) ≤ target_eps`` for the smallest non-negative ``β``.

    For ``δσ = 0`` the inequality reduces to ``β * s ≤ ε`` with solution
    ``β = ε / s``.  When ``δσ > 0`` the inequality is quadratic:

    .. math::

        \tfrac{1}{2} δσ β^2 + s β - ε ≤ 0.

    Solving for the smallest non-negative root yields

    .. math::

        β = \frac{-s + \sqrt{s^2 + 2 δσ ε}}{δσ}.

    The solver gracefully falls back to ``0`` if statistics are degenerate.
    """

    if target_eps <= 0.0:
        return 0.0
    scale = max(0.0, float(scale))
    delta_sigma = max(0.0, float(delta_sigma))
    if delta_sigma <= 0.0:
        if scale <= 0.0:
            return 0.0
        return max(0.0, target_eps / scale)
    discriminant = scale * scale + 2.0 * delta_sigma * float(target_eps)
    if discriminant <= 0.0:
        return 0.0
    root = math.sqrt(discriminant)
    beta = (root - scale) / delta_sigma
    return max(0.0, beta)


def compute_inflation_scale(pi_min: Optional[float], median_xBx: Optional[float]) -> float:
    """Compute the linear inflation scale ``s`` when statistics are available."""

    if pi_min is None or median_xBx is None:
        return 0.0
    if pi_min <= 0.0 or median_xBx < 0.0:
        return 0.0
    if median_xBx == 0.0:
        return 0.0
    return math.sqrt(median_xBx / pi_min)


class BetaControllerLogger(Protocol):
    """Protocol for objects that can log :class:`BetaControllerResult` events."""

    def log(self, *, step: int, epoch: int, result: BetaControllerResult) -> None:
        ...

    def close(self) -> None:
        ...


def _to_float(value: Any) -> Optional[float]:
    """Best-effort conversion to a Python float."""

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().to(torch.float32).item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class BetaController:
    """Clamp beta using mixture inflation bound estimates."""

    def __init__(
        self,
        target_eps: float,
        beta_min: float,
        beta_max: float,
        ema_window: int,
    ) -> None:
        if beta_min > beta_max:
            raise ValueError("beta_min must be <= beta_max")
        self.target_eps = float(target_eps)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        if ema_window <= 0:
            raise ValueError("ema_window must be positive")
        self.ema_window = int(ema_window)
        self._ema_decay = (self.ema_window - 1) / float(self.ema_window)
        self._smoothed: Optional[float] = None
        self._last_beta: Optional[float] = None
        self._last_raw: Optional[float] = None
        self._last_info: Dict[str, float | str] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def last_beta(self) -> Optional[float]:
        return self._last_beta

    @property
    def last_beta_raw(self) -> Optional[float]:
        return self._last_raw

    @property
    def last_info(self) -> Dict[str, float | str]:
        return dict(self._last_info)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply_broadcast(self, beta_value: float, info: Dict[str, float | str]) -> None:
        """Apply a beta value received from a distributed broadcast."""

        beta_float = _to_float(beta_value)
        if beta_float is None or not math.isfinite(beta_float):
            raise ValueError("beta_value must be a finite scalar")
        payload = dict(info)
        payload.setdefault("beta_applied", float(beta_float))
        self._last_beta = float(beta_float)
        self._last_info = payload

    def step(
        self,
        B_hat_stats: Dict[str, Any] | None,
        DeltaSigma_max_hat: Any,
        beta_raw: float,
    ) -> BetaControllerResult:
        """Adjust beta to satisfy the mixture inflation bound."""

        beta_raw_f = _to_float(beta_raw)
        if beta_raw_f is None or not math.isfinite(beta_raw_f):
            raise ValueError("beta_raw must be a finite scalar")
        self._last_raw = beta_raw_f
        beta_sign = 1.0 if beta_raw_f >= 0 else -1.0
        beta_mag_raw = abs(beta_raw_f)

        pi_min = None
        median_xBx = None
        if isinstance(B_hat_stats, dict):
            if "pi_min" in B_hat_stats:
                pi_min = _to_float(B_hat_stats.get("pi_min"))
            if pi_min is None and "pi" in B_hat_stats:
                pi_value = B_hat_stats["pi"]
                if isinstance(pi_value, torch.Tensor):
                    if pi_value.numel() > 0:
                        pi_min = _to_float(pi_value.min())
                elif isinstance(pi_value, (list, tuple)):
                    try:
                        pi_min = min(float(x) for x in pi_value)
                    except Exception:
                        pi_min = None
                else:
                    pi_min = _to_float(pi_value)
            median_xBx = _to_float(B_hat_stats.get("median_xBx"))
        delta_sigma = _to_float(DeltaSigma_max_hat)
        if delta_sigma is None:
            delta_sigma = 0.0
        delta_sigma = max(0.0, delta_sigma)

        scale = compute_inflation_scale(pi_min, median_xBx)
        eps_est = estimate_mixture_inflation(beta_mag_raw, scale, delta_sigma)

        candidate = beta_mag_raw
        if scale == 0.0 and delta_sigma == 0.0:
            reason = "insufficient_stats"
        elif eps_est > self.target_eps and (scale > 0.0 or delta_sigma > 0.0):
            candidate = solve_beta_for_mixture_inflation(
                self.target_eps, scale=scale, delta_sigma=delta_sigma
            )
            reason = "reduced_for_bound"
        else:
            reason = "within_target"
        candidate = max(candidate, 0.0)

        smoothed = self._smooth(candidate, reason)
        clipped = min(max(smoothed, self.beta_min), self.beta_max)
        clip_reason = None
        if clipped != smoothed:
            clip_reason = "min" if clipped == self.beta_min else "max"
        if clip_reason == "min" and reason == "reduced_for_bound":
            reason = "bound_vs_min"

        info: Dict[str, float | str] = {
            "beta_raw": float(beta_raw_f),
            "eps_target": float(self.target_eps),
            "pi_min": float(pi_min) if pi_min is not None else float("nan"),
            "median_xBx": float(median_xBx) if median_xBx is not None else float("nan"),
            "delta_sigma_max": float(delta_sigma),
            "eps_estimated": float(eps_est),
            "beta_candidate": float(candidate),
            "beta_smoothed": float(smoothed),
            "beta_clipped": float(clipped),
            "reason": reason,
        }
        if clip_reason is not None:
            info["clip"] = clip_reason

        beta_applied = beta_sign * clipped
        info["beta_applied"] = float(beta_applied)

        self._last_beta = beta_applied
        self._last_info = info
        return BetaControllerResult(beta=float(beta_applied), info=dict(info))

    def close(self) -> None:
        """Provided for backward compatibility; no resources to release."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _smooth(self, candidate: float, reason: str) -> float:
        if self._smoothed is None:
            smoothed = candidate
        elif candidate < self._smoothed and reason == "reduced_for_bound":
            smoothed = candidate
        elif candidate < self._smoothed and reason == "within_target":
            smoothed = candidate
        else:
            smoothed = self._ema_decay * self._smoothed + (1.0 - self._ema_decay) * candidate
        self._smoothed = smoothed
        return smoothed

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass


class BetaControllerCSVLogger:
    """CSV collaborator for :class:`BetaController` telemetry."""

    _CSV_COLUMNS = [
        "step",
        "epoch",
        "beta_raw",
        "beta_clipped",
        "eps_target",
        "eps_estimated",
        "reason",
    ]

    def __init__(self, log_dir: str | Path, *, is_main: bool = True) -> None:
        self._file = None
        self._writer: csv.DictWriter[str] | None = None
        if not is_main:
            return
        path = Path(log_dir).joinpath("beta_ctrl.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._CSV_COLUMNS)
        if path.stat().st_size == 0:
            self._writer.writeheader()
            self._file.flush()

    def log(self, *, step: int, epoch: int, result: BetaControllerResult) -> None:
        if self._writer is None or self._file is None:
            return
        payload = result.info
        row = {
            "step": int(step),
            "epoch": int(epoch),
            "beta_raw": payload.get("beta_raw"),
            "beta_clipped": payload.get(
                "beta_clipped", payload.get("beta_applied", result.beta)
            ),
            "eps_target": payload.get("eps_target"),
            "eps_estimated": payload.get("eps_estimated"),
            "reason": payload.get("reason"),
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None
                self._writer = None

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "BetaController",
    "BetaControllerResult",
    "BetaControllerCSVLogger",
    "estimate_mixture_inflation",
    "solve_beta_for_mixture_inflation",
    "compute_inflation_scale",
]
