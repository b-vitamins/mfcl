"""MoCo-style InfoNCE loss with a negative queue."""

from __future__ import annotations

from typing import Any, Dict, Protocol, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mfcl.losses.base import SelfSupervisedLoss


class QueueLike(Protocol):
    def get(self) -> torch.Tensor: ...


class MoCoContrastLoss(SelfSupervisedLoss):
    """InfoNCE with momentum keys and queue negatives."""

    def __init__(
        self,
        temperature: float = 0.2,
        normalize: bool = True,
        force_fp32: bool = True,
    ) -> None:
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
        self.force_fp32 = bool(force_fp32)

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

        target_dtype = torch.float32 if self.force_fp32 else q.dtype
        qf = q.to(target_dtype)
        kf = k.detach().to(target_dtype)
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
            if negs.device != q.device or negs.dtype != target_dtype:
                negs = negs.to(device=q.device, dtype=target_dtype, non_blocking=True)
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
        if loss.dtype != torch.float32:
            loss = loss.to(torch.float32)

        stats = {"pos_sim": pos.mean().detach(), "neg_sim_mean": neg.mean().detach()}
        return loss, stats

    def _prepare_forward_args(
        self, outputs: Any, batch: Dict[str, Any], model: Any
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, QueueLike], Dict[str, Any]]:
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 2:
            raise TypeError("MoCoContrastLoss expects (q, k) outputs from the model")
        q, k = outputs

        queue = getattr(model, "queue", None)
        if queue is None or not callable(getattr(queue, "get", None)):
            raise TypeError(
                "MoCoContrastLoss requires model.queue providing a get() method for fidelity"
            )

        return (q, k, queue), {}


__all__ = ["MoCoContrastLoss", "QueueLike"]
