"""BYOL symmetric loss with internal stopgrad for targets."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BYOLLoss(nn.Module):
    """Symmetric cosine/MSE BYOL loss with internal stopgrad on target."""

    def __init__(self, normalize: bool = True, variant: str = "cosine") -> None:
        """Initialize BYOL loss.

        Args:
            normalize: If True, L2-normalize inputs before similarity.
            variant: 'cosine' or 'mse'.

        Raises:
            ValueError: If variant not in {'cosine','mse'}.
        """
        super().__init__()
        if variant not in {"cosine", "mse"}:
            raise ValueError("variant must be 'cosine' or 'mse'")
        self.normalize = bool(normalize)
        self.variant = variant

    def forward(
        self,
        p1: torch.Tensor,
        z2: torch.Tensor,
        p2: torch.Tensor,
        z1: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute symmetric BYOL loss.

        Args:
            p1: [B, D] predictor output for view 1 (grad flows).
            z2: [B, D] target projection for view 2.
            p2: [B, D] predictor output for view 2.
            z1: [B, D] target projection for view 1.

        Returns:
            loss: Scalar mean over batch.
            stats: {'cos_sim': mean cosine(p, sg(z))}
        """
        # Stop gradient on targets
        z1 = z1.detach()
        z2 = z2.detach()

        p1f = p1.to(torch.float32)
        p2f = p2.to(torch.float32)
        z1f = z1.to(torch.float32)
        z2f = z2.to(torch.float32)

        if self.normalize:
            p1f = F.normalize(p1f, dim=1)
            p2f = F.normalize(p2f, dim=1)
            z1f = F.normalize(z1f, dim=1)
            z2f = F.normalize(z2f, dim=1)

        if self.variant == "cosine":
            cos1 = torch.sum(p1f * z2f, dim=1)
            cos2 = torch.sum(p2f * z1f, dim=1)
            # 1 - cos per pair, averaged symmetrically
            loss = 0.5 * ((1.0 - cos1).mean() + (1.0 - cos2).mean())
            cos_sim = 0.5 * (cos1.mean() + cos2.mean())
        else:  # mse variant
            loss = 0.5 * (
                (p1f - z2f).pow(2).sum(dim=1).mean()
                + (p2f - z1f).pow(2).sum(dim=1).mean()
            )
            # Provide cosine similarity stat for monitoring
            cos1 = F.cosine_similarity(p1f, z2f, dim=1)
            cos2 = F.cosine_similarity(p2f, z1f, dim=1)
            cos_sim = 0.5 * (cos1.mean() + cos2.mean())

        stats = {"cos_sim": cos_sim.detach()}
        return loss, stats


__all__ = ["BYOLLoss"]
