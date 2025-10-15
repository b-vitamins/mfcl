import torch

from tests.helpers.nets import TinyEncoder


def test_flip_invariance_with_average_pooling():
    enc = TinyEncoder(16).eval()
    x = torch.randn(2, 3, 32, 32)
    y = enc(x)
    x_flip = torch.flip(x, dims=[-1])
    y_flip = enc(x_flip)
    # Invariance because TinyEncoder ends with global average pooling
    assert torch.allclose(y, y_flip, atol=1e-2, rtol=1e-4)
