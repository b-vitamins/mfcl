import torch

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


def test_moco_updates_post_optimizer_step():
    enc_q = TinyEncoder(16)
    enc_k = TinyEncoder(16)
    proj_q = Projector(16, 32, 8)
    proj_k = Projector(16, 32, 8)
    method = MoCo(
        enc_q, enc_k, proj_q, proj_k, temperature=0.2, momentum=0.5, queue_size=32
    )
    method.on_train_start()
    with torch.no_grad():
        torch.manual_seed(0)
        for p in enc_q.parameters():
            p.add_(torch.randn_like(p))
    before = [p.clone() for p in enc_k.parameters()]
    method.on_optimizer_step()
    assert any(
        not torch.allclose(b, a)
        for b, a in zip(before, enc_k.parameters())
    )


def test_moco_queue_keys_gathers_across_ranks(monkeypatch):
    enc_q = TinyEncoder(4)
    enc_k = TinyEncoder(4)
    proj_q = Projector(4, 8, 4)
    proj_k = Projector(4, 8, 4)
    method = MoCo(
        enc_q,
        enc_k,
        proj_q,
        proj_k,
        temperature=0.2,
        momentum=0.5,
        queue_size=16,
        cross_rank_queue=True,
    )
    method.on_train_start()
    monkeypatch.setattr("mfcl.methods.moco.dist_utils.get_world_size", lambda: 2)
    monkeypatch.setattr("mfcl.methods.moco.dist_utils.get_rank", lambda: 1)

    def _fake_all_gather(t: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.zeros_like(t), t + 1.0], dim=0)

    monkeypatch.setattr(
        "mfcl.methods.moco.dist_utils.all_gather_tensor", _fake_all_gather
    )
    k = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    keys = method._queue_keys(k)
    assert keys.shape[0] == 2
    assert torch.allclose(keys[0], torch.zeros(4))
    assert torch.allclose(keys[1], torch.tensor([2.0, 3.0, 4.0, 5.0]))
