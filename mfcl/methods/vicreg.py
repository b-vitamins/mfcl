"""VICReg method: invariance + variance + covariance objective."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
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
        loss_fp32: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.loss_fn = VICRegLoss(
            lambda_inv=lambda_inv,
            mu_var=mu_var,
            nu_cov=nu_cov,
            gamma=gamma,
            eps=eps,
            force_fp32=loss_fp32,
        )

    def forward_views(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        z1 = self.projector(self.encoder(batch["view1"]))
        z2 = self.projector(self.encoder(batch["view2"]))
        return z1, z2

    def compute_loss(self, *proj: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        z1, z2 = proj
        loss, stats = self.loss_fn(z1, z2)
        stats["loss"] = loss
        return stats


__all__ = ["VICReg"]
