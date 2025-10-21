import torch

from mfcl.methods.swav import SwAV
from mfcl.models.heads.projector import Projector
from mfcl.models.prototypes.swavproto import SwAVPrototypes
from tests.helpers.nets import TinyEncoder


def test_swav_step_and_prototype_norm(toy_ssl_batch_multicrop):
    enc = TinyEncoder(32)
    proj = Projector(32, 64, 16)
    proto = SwAVPrototypes(num_prototypes=32, feat_dim=16, normalize=True)
    m = SwAV(
        enc,
        proj,
        proto,
        temperature=0.1,
        epsilon=0.05,
        sinkhorn_iters=2,
        codes_queue_size=0,
    )
    batch = toy_ssl_batch_multicrop(B=2, C=3, Hg=16, Hl=8, locals_n=2)
    out = m.step(batch)
    assert "loss" in out
    out["loss"].backward()
    m.on_optimizer_step()
    with torch.no_grad():
        norms = torch.norm(proto.weight, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_swav_codes_queue_accumulates(toy_ssl_batch_multicrop):
    enc = TinyEncoder(16)
    proj = Projector(16, 32, 8)
    proto = SwAVPrototypes(num_prototypes=16, feat_dim=8, normalize=True)
    m = SwAV(
        enc,
        proj,
        proto,
        temperature=0.1,
        epsilon=0.05,
        sinkhorn_iters=2,
        codes_queue_size=4,
    )
    batch = toy_ssl_batch_multicrop(B=2, C=3, Hg=16, Hl=8, locals_n=1)
    m.step(batch)
    assert m._codes_queue_rows.get(0, 0) == 2
    assert m._codes_queue_rows.get(1, 0) == 2
    m.step(batch)
    assert m._codes_queue_rows.get(0, 0) == 4
    assert m._codes_queue_rows.get(1, 0) == 4
    m.step(batch)
    assert m._codes_queue_rows.get(0, 0) == 4
    assert m._codes_queue_rows.get(1, 0) == 4


def test_swav_codes_queue_respects_maxlen_fifo():
    enc = TinyEncoder(8)
    proj = Projector(8, 16, 4)
    proto = SwAVPrototypes(num_prototypes=8, feat_dim=4, normalize=True)
    m = SwAV(
        enc,
        proj,
        proto,
        temperature=0.1,
        epsilon=0.05,
        sinkhorn_iters=2,
        codes_queue_size=3,
    )
    code_idx = (0, 1)
    for step in range(5):
        logits = [
            torch.full((1, 4), float(step), dtype=torch.float32),
            torch.full((1, 4), float(step + 100), dtype=torch.float32),
        ]
        m._enqueue_codes(logits, code_idx)

    assert m._codes_queue_rows[0] == 3
    assert m._codes_queue_rows[1] == 3

    queue0 = torch.cat(list(m._codes_queue[0]), dim=0)
    queue1 = torch.cat(list(m._codes_queue[1]), dim=0)

    expected0 = torch.tensor(
        [[2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0]]
    )
    expected1 = torch.tensor(
        [[102.0, 102.0, 102.0, 102.0], [103.0, 103.0, 103.0, 103.0], [104.0, 104.0, 104.0, 104.0]]
    )

    assert torch.allclose(queue0, expected0)
    assert torch.allclose(queue1, expected1)
