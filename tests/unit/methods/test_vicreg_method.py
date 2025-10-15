from mfcl.methods.vicreg import VICReg
from mfcl.models.heads.projector import Projector
from tests.helpers.nets import TinyEncoder


def test_vicreg_step(toy_ssl_batch_pair):
    enc = TinyEncoder(32)
    proj = Projector(32, 64, 16)
    m = VICReg(enc, proj)
    out = m.step(toy_ssl_batch_pair(B=4, C=3, H=16))
    assert "loss" in out
    out["loss"].backward()
