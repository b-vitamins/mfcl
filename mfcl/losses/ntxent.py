"""NT-Xent (SimCLR) loss.

Symmetric temperature-scaled cross-entropy between two batches of projections
using in-batch negatives. Efficient 2-view variant.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross-entropy for two views.

    Returns:
        loss: Scalar tensor, mean over the batch. The loss uses mean reduction.

    Notes:
        Similarities are computed in float32 for stability.
    """

    def __init__(self, temperature: float = 0.1, normalize: bool = True) -> None:
        """Initialize NT-Xent loss.

        Args:
            temperature: Softmax temperature τ > 0.
            normalize: If True, L2-normalize embeddings before similarity.

        Raises:
            ValueError: If temperature <= 0.
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.t = float(temperature)
        self.normalize = bool(normalize)

    def forward(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute symmetric NT-Xent.

        Args:
            z1: [B, D] projections for view 1.
            z2: [B, D] projections for view 2.

        Returns:
            loss: Scalar tensor, mean over the batch. The loss uses mean reduction.
            stats: {'pos_sim': mean cosine of positives,
                    'neg_sim_mean': mean cosine over negatives}

        Raises:
            ValueError: If batch size < 2 or shapes mismatch.

        Notes:
            Similarities are computed in float32 for stability.
        """
        if z1.ndim != 2 or z2.ndim != 2 or z1.shape != z2.shape:
            raise ValueError("z1 and z2 must be 2D tensors with identical shapes")
        B = z1.shape[0]
        if B < 2:
            raise ValueError("batch size must be >= 2 for NT-Xent")

        z1f = z1.to(torch.float32)
        z2f = z2.to(torch.float32)
        if self.normalize:
            z1f = F.normalize(z1f, dim=1)
            z2f = F.normalize(z2f, dim=1)

        sim = z1f @ z2f.t()
        # Standard temperature scaling divides by tau.  Smaller temperatures
        # sharpen the distribution and should reduce the loss when positives are
        # strong.  Multiplying by tau would invert that behaviour, so use
        # division and keep logits in float32 for numerical stability.
        logits_i = sim / self.t
        logits_j = logits_i.t()
        labels = torch.arange(B, device=sim.device)

        loss_i = F.cross_entropy(logits_i, labels)
        loss_j = F.cross_entropy(logits_j, labels)
        loss = 0.5 * (loss_i + loss_j)

        pos_sim = torch.diag(sim).mean().detach()
        neg_mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
        neg_sim_mean = sim.masked_select(neg_mask).mean().detach()
        stats = {"pos_sim": pos_sim, "neg_sim_mean": neg_sim_mean}
        return loss, stats


__all__ = ["NTXentLoss"]
