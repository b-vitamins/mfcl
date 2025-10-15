import torch
import pytest

from mfcl.losses.barlowtwins import BarlowTwinsLoss


def test_barlow_diag_and_offdiag():
    loss_fn = BarlowTwinsLoss(lambda_offdiag=5e-3)
    x = torch.randn(8, 16)
    # identical inputs -> small loss
    loss, stats = loss_fn(x, x)
    assert torch.isfinite(loss)
    assert stats["diag_mean"] <= 1.0 + 1e-3
    with pytest.raises(ValueError):
        loss_fn(torch.randn(1, 4), torch.randn(1, 4))
