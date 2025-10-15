"""BYOL method with EMA target and predictor head."""

from __future__ import annotations

from typing import Any, Dict

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
            tau_base: Initial EMA momentum (will be scheduled by caller if desired).
            normalize: Normalize in the loss.
            variant: 'cosine' or 'mse'.
        """
        super().__init__()
        self.f_q = encoder_online
        self.f_k = encoder_target
        self.g_q = projector_online
        self.g_k = projector_target
        self.q = predictor
        self.updater = MomentumUpdater(self.f_q, self.f_k, tau_base)
        self.loss_fn = BYOLLoss(normalize=normalize, variant=variant)

    def on_train_start(self) -> None:
        self.updater.copy_params()

    def forward_views(self, batch: Dict[str, Any]):
        z1_q = self.g_q(self.f_q(batch["view1"]))
        z2_q = self.g_q(self.f_q(batch["view2"]))
        with torch.no_grad():
            self.updater.update()
            z1_k = self.g_k(self.f_k(batch["view1"]))
            z2_k = self.g_k(self.f_k(batch["view2"]))
        p1 = self.q(z1_q)
        p2 = self.q(z2_q)
        return p1, z2_k, p2, z1_k

    def compute_loss(self, *proj: Any, batch: Dict[str, Any]):
        p1, z2_k, p2, z1_k = proj
        loss, stats = self.loss_fn(p1, z2_k, p2, z1_k)  # type: ignore[arg-type]
        stats["loss"] = loss
        return stats

    def on_optimizer_step(self) -> None:
        # If using post-step EMA, caller can update here instead of in forward.
        return


__all__ = ["BYOL"]
