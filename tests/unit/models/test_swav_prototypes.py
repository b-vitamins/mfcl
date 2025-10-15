import torch

from mfcl.models.prototypes.swavproto import SwAVPrototypes
from tests.helpers.asserts import assert_unit_norm


def test_swav_prototypes_normalize_and_temp():
    proto = SwAVPrototypes(
        num_prototypes=10, feat_dim=8, normalize=True, temperature=2.0
    )
    with torch.no_grad():
        proto.normalize_weights()
    assert_unit_norm(proto.weight, tol=1e-4)
    z = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    logits1 = proto(z)
    proto2 = SwAVPrototypes(
        num_prototypes=10, feat_dim=8, normalize=False, temperature=1.0
    )
    proto2._weight.data.copy_(proto.weight)
    logits2 = proto2(z)
    assert torch.allclose(logits2, 2.0 * logits1, atol=1e-4, rtol=1e-4)
