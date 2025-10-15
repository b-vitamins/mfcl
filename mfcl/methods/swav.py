"""SwAV method with multi-crop, projector, prototypes, and swapped assignment loss."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from mfcl.methods.base import BaseMethod
from mfcl.models.heads.projector import Projector
from mfcl.models.prototypes.swavproto import SwAVPrototypes
from mfcl.losses.swavloss import SwAVLoss


class SwAV(BaseMethod):
    """SwAV with multi-crop and prototype assignments."""

    def __init__(
        self,
        encoder: nn.Module,
        projector: Projector,
        prototypes: SwAVPrototypes,
        temperature: float = 0.1,
        epsilon: float = 0.05,
        sinkhorn_iters: int = 3,
        normalize_input: bool = True,
    ) -> None:
        """Construct SwAV.

        Args:
            encoder: Backbone.
            projector: Projects to prototype space dim D_p.
            prototypes: SwAVPrototypes with K clusters.
            temperature: Softmax temperature for logits before loss.
            epsilon: Sinkhorn entropic regularization.
            sinkhorn_iters: Number of Sinkhorn iterations.
            normalize_input: Whether to L2-normalize projector outputs before prototypes.
        """
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.prototypes = prototypes
        self.loss_fn = SwAVLoss(
            epsilon=epsilon, sinkhorn_iters=sinkhorn_iters, temperature=temperature
        )
        self.normalize_input = normalize_input

    def forward_views(self, batch: Dict[str, Any]):
        crops: List[torch.Tensor] = batch["crops"]
        # Encode and project each crop independently
        zs = [self.projector(self.encoder(x)) for x in crops]
        if self.normalize_input:
            zs = [torch.nn.functional.normalize(z, dim=1) for z in zs]
        logits = [self.prototypes(z) for z in zs]  # list of [B, K]
        return logits, batch["code_crops"]

    def compute_loss(self, *proj: Any, batch: Dict[str, Any]):
        logits, code_idx = proj
        loss, stats = self.loss_fn(logits, code_idx)  # type: ignore[arg-type]
        stats["loss"] = loss
        return stats

    def on_optimizer_step(self) -> None:
        # Normalize prototype weights after each optimizer step
        self.prototypes.normalize_weights()


__all__ = ["SwAV"]
