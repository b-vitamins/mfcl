"""SimCLR method with 2-view NT-Xent loss."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from mfcl.methods.base import BaseMethod
from mfcl.models.heads.projector import Projector
from mfcl.losses.ntxent import NTXentLoss


class SimCLR(BaseMethod):
    """SimCLR method with 2-view NT-Xent loss."""

    def __init__(
        self,
        encoder: nn.Module,
        projector: Projector,
        temperature: float = 0.1,
        normalize: bool = True,
    ) -> None:
        """Construct SimCLR.

        Args:
            encoder: Backbone returning [B, D] features.
            projector: Projector MLP mapping D -> d.
            temperature: NT-Xent temperature.
            normalize: If True, L2-normalize projections in the loss.
        """
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.loss_fn = NTXentLoss(temperature=temperature, normalize=normalize)

    def forward_views(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        z1 = self.projector(self.encoder(batch["view1"]))
        z2 = self.projector(self.encoder(batch["view2"]))
        return z1, z2

    def compute_loss(
        self, *proj: Any, batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        z1, z2 = proj
        loss, stats = self.loss_fn(z1, z2)
        stats["loss"] = loss
        return stats


__all__ = ["SimCLR"]
