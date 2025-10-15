import torch

from mfcl.models.prototypes.swavproto import SwAVPrototypes


def test_swav_prototypes_normalization_and_logits_shape():
    prototypes = SwAVPrototypes(num_prototypes=8, feat_dim=16, normalize=True, temperature=0.2)
    with torch.no_grad():
        prototypes.normalize_weights()
        norms = prototypes.weight.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    z = torch.randn(4, 16)
    z = z / z.norm(dim=1, keepdim=True)
    logits = prototypes(z)
    assert logits.shape == (4, 8)
