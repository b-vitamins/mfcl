import torch
import pytest

from mfcl.losses.mococontrast import MoCoContrastLoss
from mfcl.utils.queue import RingQueue


def test_moco_loss_shapes_and_queue():
    loss_fn = MoCoContrastLoss(temperature=0.2, normalize=True)
    q = torch.randn(8, 16)
    k = q.clone()
    queue = RingQueue(dim=16, size=32)
    queue.enqueue(torch.randn(16, 16))
    loss, stats = loss_fn(q, k, queue)
    assert torch.isfinite(loss)
    assert stats["pos_sim"].ndim == 0
    # queue.get() should return a tensor without grad
    assert queue.get().requires_grad is False


def test_moco_empty_queue_raises():
    loss_fn = MoCoContrastLoss(temperature=0.2)
    with pytest.raises(ValueError):
        loss_fn(torch.randn(2, 4), torch.randn(2, 4), RingQueue(dim=4, size=4))
