import torch
import pytest

from mfcl.losses.ntxent import NTXentLoss


def test_ntxent_basic_and_temp():
    loss_fn = NTXentLoss(temperature=0.2, normalize=True)
    z = torch.randn(8, 16)
    loss1, stats1 = loss_fn(z, z)
    assert torch.isfinite(loss1)
    loss_fn2 = NTXentLoss(temperature=0.1, normalize=True)
    loss2, _ = loss_fn2(z, z)
    assert loss2 >= loss1 - 1e-6


def test_ntxent_batch_too_small():
    loss_fn = NTXentLoss(temperature=0.1)
    with pytest.raises(ValueError):
        loss_fn(torch.randn(1, 4), torch.randn(1, 4))
