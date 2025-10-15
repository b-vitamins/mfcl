import torch

from mfcl.utils.seed import set_seed


def test_seed_reproducibility():
    set_seed(123, deterministic=True)
    a = torch.randn(3)
    set_seed(123, deterministic=True)
    b = torch.randn(3)
    assert torch.allclose(a, b)
