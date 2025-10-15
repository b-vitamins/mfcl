import torch
import pytest

from mfcl.losses.vicregloss import VICRegLoss


def test_vicreg_terms_behavior():
    loss_fn = VICRegLoss(lambda_invar=25.0, mu_var=25.0, nu_cov=1.0)
    x = torch.randn(8, 16)
    loss, stats = loss_fn(x, x)
    assert torch.isfinite(loss)
    assert stats["mse"] <= 1e-5
    with pytest.raises(ValueError):
        loss_fn(torch.randn(1, 4), torch.randn(1, 4))
