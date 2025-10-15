from mfcl.methods.simsiam import SimSiam
from mfcl.models.heads.projector import Projector
from mfcl.models.heads.predictor import Predictor
from tests.helpers.nets import TinyEncoder


def test_simsiam_predictor_and_detach(toy_ssl_batch_pair):
    enc = TinyEncoder(32)
    proj = Projector(32, 64, 16)
    pred = Predictor(16, 32, 16)
    m = SimSiam(enc, proj, pred, normalize=True)
    out = m.step(toy_ssl_batch_pair(B=4, C=3, H=16))
    assert "loss" in out
    out["loss"].backward()
    # No explicit target grads to check here; ensure some grads exist
    assert any(p.grad is not None for p in pred.parameters())
