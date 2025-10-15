"""MoCo-style InfoNCE loss with a negative queue."""

from __future__ import annotations

from typing import Dict, Protocol, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueueLike(Protocol):
    def get(self) -> torch.Tensor: ...


class MoCoContrastLoss(nn.Module):
    """InfoNCE with momentum keys and queue negatives."""

    def __init__(self, temperature: float = 0.2, normalize: bool = True) -> None:
        """Initialize the MoCo contrastive loss.

        Args:
            temperature: Softmax temperature τ > 0.
            normalize: If True, L2-normalize q and k.
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.t = float(temperature)
        self.normalize = bool(normalize)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, queue: QueueLike
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute InfoNCE over queue negatives.

        Args:
            q: [B, D] query projections (grad flows).
            k: [B, D] key projections (typically stopgrad from momentum encoder).
            queue: Provides negatives via queue.get(): [K, D].

        Returns:
            loss: Scalar cross-entropy against positives.
            stats: {'pos_sim': ..., 'neg_sim_mean': ...}

        Raises:
            ValueError: If shapes mismatch, queue dimension mismatches, or K == 0.
        """
        if q.shape != k.shape or q.ndim != 2:
            raise ValueError("q and k must be 2D tensors with identical shapes")

        qf = q.to(torch.float32)
        kf = k.detach().to(torch.float32)
        if self.normalize:
            qf = F.normalize(qf, dim=1)
            kf = F.normalize(kf, dim=1)

        negs = queue.get()
        has_negs = negs is not None and negs.numel() > 0
        if has_negs:
            if negs.ndim != 2 or negs.shape[1] != qf.shape[1]:
                raise ValueError("queue negatives shape must be [K, D] matching q's D")
            # Move negatives to query device for matmul compatibility and cast to fp32
            negs = negs.detach()
            if negs.device != q.device or negs.dtype != torch.float32:
                negs = negs.to(device=q.device, dtype=torch.float32, non_blocking=True)
            else:
                # Break alias with queue storage so subsequent updates don't mutate logits
                negs = negs.clone()
        else:
            raise ValueError("queue returned empty negatives")

        pos = torch.sum(qf * kf, dim=1, keepdim=True)  # [B,1]
        neg = qf @ negs.t()  # [B,K]
        logits = torch.cat([pos, neg], dim=1) / self.t
        labels = torch.zeros(qf.size(0), dtype=torch.long, device=q.device)
        loss = F.cross_entropy(logits, labels)

        stats = {"pos_sim": pos.mean().detach(), "neg_sim_mean": neg.mean().detach()}
        return loss, stats


__all__ = ["MoCoContrastLoss", "QueueLike"]
