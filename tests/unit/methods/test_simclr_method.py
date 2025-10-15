from mfcl.methods.simclr import SimCLR
from mfcl.models.heads.projector import Projector
from tests.helpers.nets import TinyEncoder


def test_simclr_step_and_grads(toy_ssl_batch_pair):
    enc = TinyEncoder(32)
    proj = Projector(32, 64, 16)
    m = SimCLR(enc, proj, temperature=0.1, normalize=True)
    batch = toy_ssl_batch_pair(B=4, C=3, H=16)
    out = m.step(batch)
    assert "loss" in out
    out["loss"].backward()
    # check grads exist
    assert any(p.grad is not None for p in enc.parameters())
