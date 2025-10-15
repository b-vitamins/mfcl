import torch

from mfcl.losses.swavloss import SwAVLoss


def test_swav_sinkhorn_properties():
    loss_fn = SwAVLoss(epsilon=0.05, sinkhorn_iters=3, temperature=0.1)
    logits = [torch.randn(16, 64), torch.randn(16, 64)]
    loss, stats = loss_fn(logits, (0, 1))
    assert torch.isfinite(loss)
    assert "entropy" in stats and "q_max_mean" in stats
