from mfcl.methods.byol import BYOL
from mfcl.models.heads.projector import Projector
from mfcl.models.heads.predictor import Predictor
from tests.helpers.nets import TinyEncoder


def test_byol_ema_and_detach(toy_ssl_batch_pair):
    f_q = TinyEncoder(32)
    f_k = TinyEncoder(32)
    g_q = Projector(32, 64, 16)
    g_k = Projector(32, 64, 16)
    q = Predictor(16, 32, 16)
    m = BYOL(f_q, f_k, g_q, g_k, q, tau_base=0.99)
    m.on_train_start()
    batch = toy_ssl_batch_pair(B=4, C=3, H=16)
    out = m.step(batch)
    assert "loss" in out
    out["loss"].backward()
    assert all(p.grad is None for p in f_k.parameters())
