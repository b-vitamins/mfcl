import torch

from mfcl.methods.swav import SwAV
from mfcl.models.heads.projector import Projector
from mfcl.models.prototypes.swavproto import SwAVPrototypes
from tests.helpers.nets import TinyEncoder


def test_swav_step_and_prototype_norm(toy_ssl_batch_multicrop):
    enc = TinyEncoder(32)
    proj = Projector(32, 64, 16)
    proto = SwAVPrototypes(num_prototypes=32, feat_dim=16, normalize=True)
    m = SwAV(enc, proj, proto, temperature=0.1, epsilon=0.05, sinkhorn_iters=2)
    batch = toy_ssl_batch_multicrop(B=2, C=3, Hg=16, Hl=8, locals_n=2)
    out = m.step(batch)
    assert "loss" in out
    out["loss"].backward()
    m.on_optimizer_step()
    with torch.no_grad():
        norms = torch.norm(proto.weight, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)
