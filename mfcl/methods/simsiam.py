"""SimSiam method: single encoder, projector, and predictor with stop-grad in loss."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from mfcl.methods.base import BaseMethod
from mfcl.models.heads.projector import Projector
from mfcl.models.heads.predictor import Predictor
from mfcl.losses.simsiamloss import SimSiamLoss
from mfcl.data.schema import extract_labels


class SimSiam(BaseMethod):
    """SimSiam with symmetric predictor and stopgrad in loss."""

    def __init__(
        self,
        encoder: nn.Module,
        projector: Projector,
        predictor: Predictor,
        normalize: bool = True,
        loss_fp32: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.predictor = predictor
        self.loss_fn = SimSiamLoss(normalize=normalize, force_fp32=loss_fp32)

    def forward_views(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z1 = self.projector(self.encoder(batch["view1"]))
        z2 = self.projector(self.encoder(batch["view2"]))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, z2, p2, z1

    def compute_loss(
        self, *proj: Any, batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        p1, z2, p2, z1 = proj
        extract_labels(batch)
        loss, stats = self.loss_fn(p1, z2, p2, z1)
        stats["loss"] = loss
        return stats


__all__ = ["SimSiam"]
