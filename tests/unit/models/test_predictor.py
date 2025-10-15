import torch

from mfcl.models.heads.predictor import Predictor


def test_predictor_shape_and_bn_toggle():
    x = torch.randn(4, 16)
    p = Predictor(16, 32, 8, use_bn=True)
    y = p(x)
    assert y.shape == (4, 8)
    p2 = Predictor(16, 32, 8, use_bn=False)
    has_bn = any(isinstance(m, torch.nn.BatchNorm1d) for m in p2.modules())
    assert not has_bn
