"""Mixture inflation bound controller for beta parameter."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

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


class BetaController:
    """Clamp beta using mixture inflation bound estimates."""

    _CSV_COLUMNS = ["step", "epoch", "beta_raw", "beta_clipped", "eps_target", "eps_estimated", "reason"]

    def __init__(
        self,
        target_eps: float,
        beta_min: float,
        beta_max: float,
        ema_window: int,
        *,
        log_dir: str | Path | None = None,
        is_main: bool = True,
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

        self._csv_path: Optional[Path] = None
        self._csv_file = None
        self._csv_writer: Optional[csv.DictWriter[str]] = None
        if log_dir is not None and is_main:
            path = Path(log_dir).joinpath("beta_ctrl.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_path = path
            self._csv_file = path.open("a", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._CSV_COLUMNS)
            if path.stat().st_size == 0:
                self._csv_writer.writeheader()
                self._csv_file.flush()

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
    ) -> Tuple[float, Dict[str, float | str]]:
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

        info: Dict[str, float | str] = {
            "beta_raw": beta_raw_f,
            "eps_target": self.target_eps,
            "pi_min": float(pi_min) if pi_min is not None else float("nan"),
            "median_xBx": float(median_xBx) if median_xBx is not None else float("nan"),
            "delta_sigma_max": float(delta_sigma),
        }

        scale = 0.0
        if pi_min is not None and pi_min > 0.0 and median_xBx is not None and median_xBx >= 0.0:
            scale = math.sqrt(median_xBx / pi_min) if median_xBx > 0.0 else 0.0

        eps_est = beta_mag_raw * scale + 0.5 * beta_mag_raw * beta_mag_raw * delta_sigma
        info["eps_estimated"] = float(eps_est)

        candidate = beta_mag_raw
        reason = "monitor_only"
        if scale == 0.0 and delta_sigma == 0.0:
            reason = "insufficient_stats"
        elif eps_est > self.target_eps and (scale > 0.0 or delta_sigma > 0.0):
            candidate = self._solve_for_beta(scale=scale, delta_sigma=delta_sigma)
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
        info["beta_candidate"] = float(candidate)
        info["beta_smoothed"] = float(smoothed)
        info["beta_clipped"] = float(clipped)
        if clip_reason is not None:
            info["clip"] = clip_reason
        info["reason"] = reason

        beta_applied = beta_sign * clipped
        info["beta_applied"] = float(beta_applied)
        self._last_beta = beta_applied
        self._last_info = info
        return beta_applied, dict(info)

    def log_step(self, *, step: int, epoch: int, info: Dict[str, float | str] | None = None) -> None:
        if self._csv_writer is None or self._csv_file is None:
            return
        payload = info or self._last_info
        if not payload:
            return
        row = {
            "step": int(step),
            "epoch": int(epoch),
            "beta_raw": payload.get("beta_raw"),
            "beta_clipped": payload.get("beta_clipped", payload.get("beta_applied", self._last_beta)),
            "eps_target": payload.get("eps_target", self.target_eps),
            "eps_estimated": payload.get("eps_estimated"),
            "reason": payload.get("reason"),
        }
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close(self) -> None:
        if self._csv_file is not None:
            try:
                self._csv_file.close()
            finally:
                self._csv_file = None
                self._csv_writer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _solve_for_beta(self, *, scale: float, delta_sigma: float) -> float:
        if delta_sigma <= 0.0:
            if scale <= 0.0:
                return self.beta_min
            return max(0.0, self.target_eps / scale)
        disc = scale * scale + 2.0 * delta_sigma * self.target_eps
        if disc <= 0.0:
            return self.beta_min
        root = math.sqrt(disc)
        beta = (root - scale) / delta_sigma
        return max(0.0, beta)

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


__all__ = ["BetaController"]
