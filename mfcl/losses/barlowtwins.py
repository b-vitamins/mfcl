"""Barlow Twins cross-correlation loss implementation."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def _offdiag(M: torch.Tensor) -> torch.Tensor:
    n, m = M.shape
    if n != m:
        raise ValueError("correlation matrix must be square to extract off-diagonal entries")
    return M.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinsLoss(nn.Module):
    """Cross-correlation identity loss."""

    def __init__(self, lambda_offdiag: float = 5e-3, eps: float = 1e-4) -> None:
        """Initialize Barlow Twins loss.

        Args:
            lambda_offdiag: Weight for off-diagonal terms.
            eps: Small constant for std normalization.

        Raises:
            ValueError: If lambda_offdiag <= 0 or eps <= 0.
        """
        super().__init__()
        if lambda_offdiag <= 0 or eps <= 0:
            raise ValueError("lambda_offdiag and eps must be > 0")
        self.lmb = float(lambda_offdiag)
        self.eps = float(eps)

    def forward(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute Barlow Twins loss.

        Args:
            z1: [B, D] projections for view 1.
            z2: [B, D] projections for view 2.

        Returns:
            loss: Scalar.
            stats: {'diag_mean': mean diag(C), 'offdiag_mean': mean abs offdiag(C)}

        Raises:
            ValueError: If batch size < 2 or shapes mismatch.
        """
        if z1.ndim != 2 or z2.ndim != 2 or z1.shape != z2.shape:
            raise ValueError("z1 and z2 must be 2D tensors with identical shapes")
        B = z1.shape[0]
        if B < 2:
            raise ValueError("batch size must be >= 2 for Barlow Twins")

        x = z1.to(torch.float32)
        y = z2.to(torch.float32)
        # Center and normalize per-dimension
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        x = x / (x.std(dim=0) + self.eps)
        y = y / (y.std(dim=0) + self.eps)

        C = (x.t() @ y) / B  # [D,D]
        diag = torch.diag(C)
        off = _offdiag(C)
        loss = ((1.0 - diag) ** 2).sum() + self.lmb * (off**2).sum()

        stats = {
            "diag_mean": diag.mean().detach(),
            "offdiag_mean": off.abs().mean().detach(),
        }
        return loss, stats


__all__ = ["BarlowTwinsLoss"]
