from mfcl.methods.moco import MoCo
from mfcl.models.heads.projector import Projector
from tests.helpers.nets import TinyEncoder


def test_moco_queue_and_no_grad_key(toy_ssl_batch_pair):
    enc_q = TinyEncoder(32)
    enc_k = TinyEncoder(32)
    proj_q = Projector(32, 64, 16)
    proj_k = Projector(32, 64, 16)
    m = MoCo(
        enc_q, enc_k, proj_q, proj_k, temperature=0.2, momentum=0.999, queue_size=128
    )
    m.on_train_start()
    batch = toy_ssl_batch_pair(B=4, C=3, H=16)
    out = m.step(batch)
    assert "loss" in out
    out["loss"].backward()
    # Key encoder params should have no grad
    assert all(p.grad is None for p in enc_k.parameters())
