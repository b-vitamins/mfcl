"""BYOL method with EMA target and predictor head."""

from __future__ import annotations

from typing import Any, Dict

import math

import torch
import torch.nn as nn

from mfcl.methods.base import BaseMethod
from mfcl.models.heads.projector import Projector
from mfcl.models.heads.predictor import Predictor
from mfcl.losses.byolloss import BYOLLoss
from mfcl.utils.ema import MomentumUpdater


class BYOL(BaseMethod):
    """BYOL with EMA target and predictor head."""

    def __init__(
        self,
        encoder_online: nn.Module,
        encoder_target: nn.Module,
        projector_online: Projector,
        projector_target: Projector,
        predictor: Predictor,
        tau_base: float = 0.996,
        tau_final: float | None = None,
        momentum_schedule: str = "const",
        momentum_schedule_steps: int | None = None,
        normalize: bool = True,
        variant: str = "cosine",
    ) -> None:
        """Construct BYOL.

        Args:
            encoder_online: Online encoder.
            encoder_target: Target encoder (EMA).
            projector_online: Projector for online.
            projector_target: Projector for target.
            predictor: Predictor MLP on online branch.
            tau_base: Initial EMA momentum.
            tau_final: Final EMA momentum for scheduled variants (defaults to ``tau_base``).
            momentum_schedule: Momentum schedule name ("const" or "cosine").
            momentum_schedule_steps: Number of optimizer steps for schedules that
                require a horizon (ignored for "const").
            normalize: Normalize in the loss.
            variant: 'cosine' or 'mse'.
        """
        super().__init__()
        if tau_final is None:
            tau_final = tau_base
        if not 0.0 <= tau_base < 1.0:
            raise ValueError("tau_base must be in [0, 1)")
        if not 0.0 <= tau_final < 1.0:
            raise ValueError("tau_final must be in [0, 1)")
        momentum_schedule = momentum_schedule.lower()
        if momentum_schedule not in {"const", "cosine"}:
            raise ValueError("momentum_schedule must be 'const' or 'cosine'")
        if momentum_schedule == "cosine":
            if momentum_schedule_steps is None or momentum_schedule_steps <= 0:
                raise ValueError(
                    "momentum_schedule_steps must be a positive integer when using"
                    " cosine momentum scheduling"
                )
        else:
            momentum_schedule_steps = None

        self.f_q = encoder_online
        self.f_k = encoder_target
        self.g_q = projector_online
        self.g_k = projector_target
        self.q = predictor
        self.updater = MomentumUpdater(self.f_q, self.f_k, tau_base)
        self._tau_base = float(tau_base)
        self._tau_final = float(tau_final)
        self._momentum_schedule = momentum_schedule
        self._schedule_steps = momentum_schedule_steps
        self._optimizer_steps = 0
        self._current_momentum = float(tau_base)
        self.loss_fn = BYOLLoss(normalize=normalize, variant=variant)

    def on_train_start(self) -> None:
        self.updater.copy_params()

    def forward_views(self, batch: Dict[str, Any]):
        z1_q = self.g_q(self.f_q(batch["view1"]))
        z2_q = self.g_q(self.f_q(batch["view2"]))
        with torch.no_grad():
            z1_k = self.g_k(self.f_k(batch["view1"]))
            z2_k = self.g_k(self.f_k(batch["view2"]))
        p1 = self.q(z1_q)
        p2 = self.q(z2_q)
        return p1, z2_k, p2, z1_k

    def compute_loss(self, *proj: Any, batch: Dict[str, Any]):
        p1, z2_k, p2, z1_k = proj
        loss, stats = self.loss_fn(p1, z2_k, p2, z1_k)  # type: ignore[arg-type]
        stats["loss"] = loss
        stats.setdefault(
            "momentum",
            torch.tensor(self._current_momentum, device=loss.device, dtype=loss.dtype),
        )
        return stats

    def on_optimizer_step(self) -> None:
        self._optimizer_steps += 1
        if self._momentum_schedule == "cosine" and self._schedule_steps is not None:
            if self._schedule_steps > 1:
                numerator = min(self._optimizer_steps - 1, self._schedule_steps - 1)
                progress = max(numerator, 0) / float(self._schedule_steps - 1)
            else:
                progress = 1.0
            cos_val = math.cos(math.pi * progress)
            momentum = self._tau_final - (self._tau_final - self._tau_base) * (cos_val + 1.0) * 0.5
        else:
            momentum = self._tau_base
        self._current_momentum = float(momentum)
        self.updater.set_momentum(self._current_momentum)
        self.updater.update()


__all__ = ["BYOL"]
