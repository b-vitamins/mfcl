import torch

from mfcl.losses.swavloss import SwAVLoss


def test_sinkhorn_doubly_stochastic_and_entropy():
    B, K = 64, 128
    scores = torch.randn(B, K)
    loss = SwAVLoss(epsilon=0.1, sinkhorn_iters=5, temperature=0.1)
    Q = loss._sinkhorn(scores)  # type: ignore[attr-defined]
    row_sums = Q.sum(dim=1)
    col_sums = Q.sum(dim=0)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3, rtol=1e-3)
    # Columns should sum to ~B/K
    target = (B / K) * torch.ones_like(col_sums)
    assert torch.allclose(col_sums, target, atol=2e-1, rtol=1e-3)
    # Entropy monotonicity with epsilon
    Q_hi = SwAVLoss(epsilon=0.2, sinkhorn_iters=3, temperature=0.1)._sinkhorn(scores)  # type: ignore[attr-defined]

    def entropy(Q):
        return -(Q * (Q.add(1e-12).log())).sum(dim=1).mean()

    assert entropy(Q) <= entropy(Q_hi) + 1e-6
