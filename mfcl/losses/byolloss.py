"""BYOL symmetric loss with internal stopgrad for targets."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BYOLLoss(nn.Module):
    """Symmetric cosine/MSE BYOL loss with internal stopgrad on target."""

    def __init__(
        self,
        normalize: bool = True,
        variant: str = "cosine",
        force_fp32: bool = True,
    ) -> None:
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
        self.force_fp32 = bool(force_fp32)

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

        Raises:
            ValueError: If any tensor is not 2D [B, D].
            ValueError: If tensor shapes differ.
            ValueError: If batch size < 1.
        """
        if not (p1.ndim == p2.ndim == z1.ndim == z2.ndim == 2):
            raise ValueError("all BYOL tensors must be 2D [B, D]")
        if p1.shape != p2.shape or p1.shape != z1.shape or p1.shape != z2.shape:
            raise ValueError("all BYOL tensors must share the same shape [B, D]")
        if p1.size(0) < 1:
            raise ValueError("batch size must be >= 1 for BYOL")

        # Stop gradient on targets
        z1 = z1.detach()
        z2 = z2.detach()

        target_dtype = torch.float32 if self.force_fp32 else p1.dtype
        p1f = p1.to(target_dtype)
        p2f = p2.to(target_dtype)
        z1f = z1.to(target_dtype)
        z2f = z2.to(target_dtype)

        if self.normalize:
            p1f = F.normalize(p1f, dim=1)
            p2f = F.normalize(p2f, dim=1)
            z1f = F.normalize(z1f, dim=1)
            z2f = F.normalize(z2f, dim=1)

        cos1 = F.cosine_similarity(p1f, z2f, dim=1)
        cos2 = F.cosine_similarity(p2f, z1f, dim=1)
        cos_sim = 0.5 * (cos1.mean() + cos2.mean())

        if self.variant == "cosine":
            # 1 - cos per pair, averaged symmetrically
            loss = 0.5 * ((1.0 - cos1).mean() + (1.0 - cos2).mean())
        else:  # mse variant
            loss = 0.5 * (
                (p1f - z2f).pow(2).sum(dim=1).mean()
                + (p2f - z1f).pow(2).sum(dim=1).mean()
            )
        if loss.dtype != torch.float32:
            loss = loss.to(torch.float32)

        stats = {"cos_sim": cos_sim.detach()}
        return loss, stats


__all__ = ["BYOLLoss"]
