"""VICReg method: invariance + variance + covariance objective."""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from mfcl.methods.base import BaseMethod
from mfcl.models.heads.projector import Projector
from mfcl.losses.vicregloss import VICRegLoss


class VICReg(BaseMethod):
    """VICReg with 3-term objective."""

    def __init__(
        self,
        encoder: nn.Module,
        projector: Projector,
        lambda_inv: float = 25.0,
        mu_var: float = 25.0,
        nu_cov: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.loss_fn = VICRegLoss(
            lambda_invar=lambda_inv, mu_var=mu_var, nu_cov=nu_cov, gamma=gamma, eps=eps
        )

    def forward_views(self, batch: Dict[str, Any]):
        z1 = self.projector(self.encoder(batch["view1"]))
        z2 = self.projector(self.encoder(batch["view2"]))
        return z1, z2

    def compute_loss(self, *proj: Any, batch: Dict[str, Any]):
        z1, z2 = proj
        loss, stats = self.loss_fn(z1, z2)  # type: ignore[arg-type]
        stats["loss"] = loss
        return stats


__all__ = ["VICReg"]
