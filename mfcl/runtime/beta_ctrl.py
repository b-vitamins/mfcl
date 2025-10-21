"""Mixture inflation bound controller for beta parameter."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch


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


@dataclass(frozen=True)
class BetaControllerResult:
    """Structured result returned by :meth:`BetaController.step`."""

    beta_raw: float
    beta_candidate: float
    beta_smoothed: float
    beta_clipped: float
    beta_applied: float
    reason: str
    eps_target: float
    eps_estimated: float
    delta_sigma_max: float
    pi_min: Optional[float] = None
    median_xBx: Optional[float] = None
    clip: Optional[str] = None

    def to_dict(self) -> Dict[str, float | str]:
        """Serialize the result to a dictionary compatible with legacy code."""

        payload: Dict[str, float | str] = {
            "beta_raw": float(self.beta_raw),
            "beta_candidate": float(self.beta_candidate),
            "beta_smoothed": float(self.beta_smoothed),
            "beta_clipped": float(self.beta_clipped),
            "beta_applied": float(self.beta_applied),
            "eps_target": float(self.eps_target),
            "eps_estimated": float(self.eps_estimated),
            "delta_sigma_max": float(self.delta_sigma_max),
            "reason": self.reason,
            "pi_min": float(self.pi_min) if self.pi_min is not None else float("nan"),
            "median_xBx": float(self.median_xBx) if self.median_xBx is not None else float("nan"),
        }
        if self.clip is not None:
            payload["clip"] = self.clip
        return payload

    def with_applied(self, beta_applied: float) -> "BetaControllerResult":
        """Return a copy with an updated applied beta value."""

        beta_value = float(beta_applied)
        return replace(self, beta_applied=beta_value, beta_clipped=abs(beta_value))


def compute_inflation_scale(pi_min: Optional[float], median_xBx: Optional[float]) -> float:
    """Compute the linear coefficient used in the inflation bound.

    The mixture inflation constraint is expressed as

    .. math:: \epsilon(\beta) = |\beta| \sqrt{\tfrac{\operatorname{median}(x^\top B x)}{\pi_{\min}}}

    where :math:`\pi_{\min}` is the smallest mixture weight and
    :math:`\operatorname{median}(x^\top B x)` summarises the covariance term.
    When the statistics are missing or numerically unsafe the scale reduces to
    zero, disabling the linear term and forcing the controller to rely solely on
    the quadratic correction.
    """

    if pi_min is None or median_xBx is None:
        return 0.0
    if pi_min <= 0.0 or median_xBx < 0.0:
        return 0.0
    if median_xBx == 0.0:
        return 0.0
    return math.sqrt(median_xBx / pi_min)


def estimate_mixture_inflation(beta_magnitude: float, scale: float, delta_sigma: float) -> float:
    """Estimate the mixture inflation bound for a candidate ``beta``.

    The bound linearises the change in the mixture covariance and is modelled as

    .. math:: \epsilon(\beta) = |\beta| \cdot \text{scale} + \tfrac{1}{2} |\beta|^2 \Delta\sigma,

    combining a first-order term governed by ``scale`` with a second-order term
    representing covariance growth ``delta_sigma``. Both coefficients are
    clamped to be non-negative to ensure the bound remains well-defined.
    """

    beta_mag = abs(float(beta_magnitude))
    if beta_mag == 0.0:
        return 0.0
    scale = max(0.0, float(scale))
    delta_sigma = max(0.0, float(delta_sigma))
    return beta_mag * scale + 0.5 * beta_mag * beta_mag * delta_sigma


def solve_beta_for_inflation_bound(
    target_eps: float, *, scale: float, delta_sigma: float
) -> float:
    """Solve ``scale * |beta| + 0.5 * delta_sigma * |beta|^2 <= target_eps``.

    The inequality is quadratic in ``|beta|``. When ``delta_sigma`` is zero the
    solution reduces to a simple division. Otherwise the positive root of the
    quadratic equation ``0.5 * delta_sigma * x^2 + scale * x - target_eps = 0``
    yields the maximal admissible ``|beta|``. Only non-negative, finite
    solutions are returned; degenerate inputs fall back to zero.
    """

    target_eps = float(target_eps)
    if target_eps <= 0.0:
        return 0.0
    scale = max(0.0, float(scale))
    delta_sigma = max(0.0, float(delta_sigma))
    if delta_sigma == 0.0:
        if scale == 0.0:
            return 0.0
        return max(0.0, target_eps / scale)
    discriminant = scale * scale + 2.0 * delta_sigma * target_eps
    if discriminant <= 0.0:
        return 0.0
    beta = (math.sqrt(discriminant) - scale) / delta_sigma
    return max(0.0, beta)


class BetaControllerLogger:
    """Interface for objects that persist controller results."""

    def log(self, *, step: int, epoch: int, result: BetaControllerResult) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Release any resources held by the logger."""


class BetaControllerCsvLogger(BetaControllerLogger):
    """Persist controller results to ``beta_ctrl.csv`` in the log directory."""

    _COLUMNS = [
        "step",
        "epoch",
        "beta_raw",
        "beta_clipped",
        "eps_target",
        "eps_estimated",
        "reason",
    ]

    def __init__(self, log_dir: str | Path, *, is_main: bool = True) -> None:
        self._writer: Optional[csv.DictWriter[str]] = None
        self._file = None
        if not is_main:
            return
        path = Path(log_dir).joinpath("beta_ctrl.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._COLUMNS)
        if path.stat().st_size == 0:
            self._writer.writeheader()
            self._file.flush()

    def log(self, *, step: int, epoch: int, result: BetaControllerResult) -> None:
        if self._writer is None or self._file is None:
            return
        payload = result.to_dict()
        row = {
            "step": int(step),
            "epoch": int(epoch),
            "beta_raw": payload.get("beta_raw"),
            "beta_clipped": payload.get("beta_clipped", payload.get("beta_applied")),
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


class BetaController:
    """Clamp beta using mixture inflation bound estimates."""

    def __init__(
        self,
        target_eps: float,
        beta_min: float,
        beta_max: float,
        ema_window: int,
        *,
        logger: BetaControllerLogger | None = None,
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
        self._last_raw: Optional[float] = None
        self._last_result: Optional[BetaControllerResult] = None
        self._logger = logger

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def last_beta(self) -> Optional[float]:
        return self._last_result.beta_applied if self._last_result is not None else None

    @property
    def last_beta_raw(self) -> Optional[float]:
        return self._last_raw

    @property
    def last_info(self) -> Dict[str, float | str]:
        if self._last_result is None:
            return {}
        return self._last_result.to_dict()

    @property
    def last_result(self) -> Optional[BetaControllerResult]:
        return self._last_result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply_broadcast(
        self,
        beta_value: float,
        result: BetaControllerResult | Mapping[str, Any] | None = None,
    ) -> BetaControllerResult:
        """Apply a beta value received from a distributed broadcast."""

        beta_float = _to_float(beta_value)
        if beta_float is None or not math.isfinite(beta_float):
            raise ValueError("beta_value must be a finite scalar")

        if isinstance(result, BetaControllerResult):
            payload = result
        elif isinstance(result, Mapping):
            clip_value = result.get("clip")
            clip = str(clip_value) if clip_value is not None else None
            payload = BetaControllerResult(
                beta_raw=_to_float(result.get("beta_raw")) or float("nan"),
                beta_candidate=_to_float(result.get("beta_candidate")) or float("nan"),
                beta_smoothed=_to_float(result.get("beta_smoothed")) or float("nan"),
                beta_clipped=_to_float(result.get("beta_clipped")) or abs(float(beta_float)),
                beta_applied=float(beta_float),
                reason=str(result.get("reason", "broadcast")),
                eps_target=_to_float(result.get("eps_target")) or self.target_eps,
                eps_estimated=_to_float(result.get("eps_estimated")) or float("nan"),
                delta_sigma_max=_to_float(result.get("delta_sigma_max")) or 0.0,
                pi_min=_to_float(result.get("pi_min")),
                median_xBx=_to_float(result.get("median_xBx")),
                clip=clip,
            )
        elif self._last_result is not None:
            payload = self._last_result
        else:
            payload = BetaControllerResult(
                beta_raw=float("nan"),
                beta_candidate=abs(float(beta_float)),
                beta_smoothed=abs(float(beta_float)),
                beta_clipped=abs(float(beta_float)),
                beta_applied=float(beta_float),
                reason="broadcast",
                eps_target=self.target_eps,
                eps_estimated=float("nan"),
                delta_sigma_max=0.0,
            )

        updated = payload.with_applied(float(beta_float))
        self._last_result = updated
        self._last_raw = updated.beta_raw
        self._smoothed = updated.beta_clipped
        return updated

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
        reason = "monitor_only"
        if scale == 0.0 and delta_sigma == 0.0:
            reason = "insufficient_stats"
        elif eps_est > self.target_eps and (scale > 0.0 or delta_sigma > 0.0):
            candidate = solve_beta_for_inflation_bound(
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
            # When the bound requires a smaller beta than beta_min we cannot satisfy it.
            reason = "bound_vs_min"

        beta_applied = beta_sign * clipped
        result = BetaControllerResult(
            beta_raw=beta_raw_f,
            beta_candidate=float(candidate),
            beta_smoothed=float(smoothed),
            beta_clipped=float(clipped),
            beta_applied=float(beta_applied),
            reason=reason,
            eps_target=self.target_eps,
            eps_estimated=float(eps_est),
            delta_sigma_max=float(delta_sigma),
            pi_min=float(pi_min) if pi_min is not None else None,
            median_xBx=float(median_xBx) if median_xBx is not None else None,
            clip=clip_reason,
        )
        self._last_result = result
        return result

    def log_step(
        self,
        *,
        step: int,
        epoch: int,
        result: BetaControllerResult | None = None,
    ) -> None:
        if self._logger is None:
            return
        payload = result or self._last_result
        if payload is None:
            return
        self._logger.log(step=step, epoch=epoch, result=payload)

    def close(self) -> None:
        if self._logger is not None:
            self._logger.close()

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


__all__ = [
    "BetaController",
    "BetaControllerResult",
    "BetaControllerLogger",
    "BetaControllerCsvLogger",
    "compute_inflation_scale",
    "estimate_mixture_inflation",
    "solve_beta_for_inflation_bound",
]
