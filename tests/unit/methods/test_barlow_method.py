from mfcl.methods.barlow import BarlowTwins
from mfcl.models.heads.projector import Projector
from tests.helpers.nets import TinyEncoder


def test_barlow_method_step_and_grads(toy_ssl_batch_pair):
    enc = TinyEncoder(32)
    proj = Projector(32, 64, 16)
    m = BarlowTwins(encoder=enc, projector=proj)
    out = m.step(toy_ssl_batch_pair(B=4, C=3, H=16))
    assert "loss" in out
    out["loss"].backward()
    assert any(p.grad is not None for p in enc.parameters())
