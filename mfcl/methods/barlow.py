"""Barlow Twins method with projector and correlation identity objective."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from mfcl.methods.base import BaseMethod
from mfcl.models.heads.projector import Projector
from mfcl.losses.barlowtwins import BarlowTwinsLoss
from mfcl.data.schema import extract_labels


class BarlowTwins(BaseMethod):
    """Barlow Twins with projector and correlation identity objective."""

    def __init__(
        self,
        encoder: nn.Module,
        projector: Projector,
        lambda_offdiag: float = 5e-3,
        eps: float = 1e-4,
        loss_fp32: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.loss_fn = BarlowTwinsLoss(
            lambda_offdiag=lambda_offdiag,
            eps=eps,
            force_fp32=loss_fp32,
        )

    def forward_views(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        z1 = self.projector(self.encoder(batch["view1"]))
        z2 = self.projector(self.encoder(batch["view2"]))
        return z1, z2

    def compute_loss(self, *proj: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        z1, z2 = proj
        extract_labels(batch)
        loss, stats = self.loss_fn(z1, z2)
        stats["loss"] = loss
        return stats


__all__ = ["BarlowTwins"]
