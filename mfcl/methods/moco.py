"""MoCo v2 method with momentum key encoder and negative queue."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from mfcl.methods.base import BaseMethod
from mfcl.models.heads.projector import Projector
from mfcl.losses.mococontrast import MoCoContrastLoss
from mfcl.utils.ema import MomentumUpdater
from mfcl.utils.queue import RingQueue


class MoCo(BaseMethod):
    """MoCo v2 with momentum encoder and negative queue."""

    def __init__(
        self,
        encoder_q: nn.Module,
        encoder_k: nn.Module,
        projector_q: Projector,
        projector_k: Projector,
        temperature: float = 0.2,
        momentum: float = 0.999,
        queue_size: int = 65536,
        normalize: bool = True,
    ) -> None:
        """Construct MoCo.

        Args:
            encoder_q: Online encoder (grad flows).
            encoder_k: Key encoder (EMA updated).
            projector_q: Projector for queries.
            projector_k: Projector for keys.
            temperature: InfoNCE temperature.
            momentum: EMA momentum for key encoder params.
            queue_size: Number of stored negatives.
            normalize: Normalize in the loss.
        """
        super().__init__()
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        self.projector_q = projector_q
        self.projector_k = projector_k
        self.updater = MomentumUpdater(self.encoder_q, self.encoder_k, momentum)
        # Store queue on CPU by default; loss will move negatives to device as needed.
        self.queue = RingQueue(
            dim=projector_q.output_dim, size=queue_size, device="cpu"
        )
        self.loss_fn = MoCoContrastLoss(temperature=temperature, normalize=normalize)

    def on_train_start(self) -> None:
        self.updater.copy_params()
        # Seed the negative queue to avoid empty-queue edge cases on first step
        try:
            with torch.no_grad():
                seed = torch.randn(
                    self.queue.size, self.projector_q.output_dim, device=self.queue.buf.device
                )
                self.queue.enqueue(seed)
        except Exception:
            pass

    def forward_views(self, batch: Dict[str, Any]):
        q = self.projector_q(self.encoder_q(batch["view1"]))
        with torch.no_grad():
            self.updater.update()
            k = self.projector_k(self.encoder_k(batch["view2"]))
        return q, k

    def compute_loss(self, *proj: Any, batch: Dict[str, Any]):
        q, k = proj  # type: ignore[assignment]
        loss, stats = self.loss_fn(q, k, self.queue)  # type: ignore[arg-type]
        # Enqueue detached keys (again, acts as FIFO ring buffer)
        self.queue.enqueue(k.detach())
        stats["loss"] = loss
        stats["queue_size"] = torch.tensor(
            self.queue.size, device=q.device, dtype=torch.float32
        )
        return stats

    def on_optimizer_step(self) -> None:
        # No action; EMA occurs pre-forward; queue updated during compute_loss.
        return


__all__ = ["MoCo"]
