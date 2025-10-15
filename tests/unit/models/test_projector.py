import torch

from mfcl.models.heads.projector import Projector
from tests.helpers.asserts import assert_unit_norm


def test_projector_shapes_and_norm():
    x = torch.randn(4, 16)
    p = Projector(16, 32, 8, num_layers=2, use_bn=True, norm_out=True)
    y = p(x)
    assert y.shape == (4, 8)
    assert_unit_norm(y, tol=1e-4)


def test_projector_bn_last():
    p = Projector(16, 32, 8, num_layers=3, use_bn=True, bn_last=True)
    assert any(isinstance(m, torch.nn.BatchNorm1d) for m in p.modules())
