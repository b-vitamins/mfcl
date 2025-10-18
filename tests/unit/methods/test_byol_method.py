import math

import torch

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


def test_byol_updates_post_optimizer_step():
    f_q = TinyEncoder(8)
    f_k = TinyEncoder(8)
    g_q = Projector(8, 16, 4)
    g_k = Projector(8, 16, 4)
    q = Predictor(4, 8, 4)
    method = BYOL(f_q, f_k, g_q, g_k, q, tau_base=0.5)
    method.on_train_start()
    # Simulate optimizer changing online weights before EMA update
    with torch.no_grad():
        torch.manual_seed(0)
        for p in f_q.parameters():
            p.add_(torch.randn_like(p))
    target_before = [p.clone() for p in f_k.parameters()]
    method.on_optimizer_step()
    assert any(
        not torch.allclose(before, after)
        for before, after in zip(target_before, f_k.parameters())
    )


def test_byol_cosine_momentum_schedule_progression():
    f_q = TinyEncoder(4)
    f_k = TinyEncoder(4)
    g_q = Projector(4, 8, 2)
    g_k = Projector(4, 8, 2)
    q = Predictor(2, 4, 2)
    steps = 10
    tau_base = 0.9
    tau_final = 0.99
    method = BYOL(
        f_q,
        f_k,
        g_q,
        g_k,
        q,
        tau_base=tau_base,
        tau_final=tau_final,
        momentum_schedule="cosine",
        momentum_schedule_steps=steps,
    )
    method.on_train_start()
    seen = []
    for _ in range(steps):
        method.on_optimizer_step()
        seen.append(method._current_momentum)
    assert math.isclose(seen[0], tau_base, rel_tol=1e-5, abs_tol=1e-5)
    assert seen[-1] > seen[0]
    assert math.isclose(seen[-1], tau_final, rel_tol=1e-5, abs_tol=1e-5)
