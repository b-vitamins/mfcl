import pytest
import torch
import torch.nn.functional as F

from mfcl.losses.ntxent import NTXentLoss
from mfcl.mixture import MixtureStats
from mfcl.mixture.context import _set_active_estimator
from mfcl.utils import dist as dist_utils


def test_ntxent_basic_and_temp():
    loss_fn = NTXentLoss(temperature=0.2, normalize=True)
    z = torch.randn(8, 16)
    loss1, stats1 = loss_fn(z, z)
    assert torch.isfinite(loss1)
    loss_fn2 = NTXentLoss(temperature=0.1, normalize=True)
    loss2, _ = loss_fn2(z, z)
    assert loss2 <= loss1 + 1e-6


def test_ntxent_batch_too_small():
    loss_fn = NTXentLoss(temperature=0.1)
    with pytest.raises(ValueError):
        loss_fn(torch.randn(1, 4), torch.randn(1, 4))


def test_ntxent_two_n_mode_matches_manual_computation():
    z1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    z2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    temperature = 0.25
    loss_fn = NTXentLoss(temperature=temperature, normalize=False, mode="2N")
    loss, stats = loss_fn(z1, z2)
    z_all = torch.cat([z1, z2], dim=0)
    sim = z_all @ z_all.t()
    logits = sim / temperature
    neg_inf = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(torch.eye(4, dtype=torch.bool), neg_inf)
    labels = torch.tensor([2, 3, 0, 1], dtype=torch.long)
    expected = F.cross_entropy(logits, labels)
    assert torch.allclose(loss, expected, atol=1e-6)
    pos_vals = torch.cat([torch.diag(sim, diagonal=2), torch.diag(sim, diagonal=-2)])
    assert torch.allclose(stats["pos_sim"], pos_vals.mean())


def test_ntxent_invalid_mode_raises():
    with pytest.raises(ValueError):
        NTXentLoss(mode="bogus")


def test_ntxent_cross_rank_paired_matches_manual(monkeypatch):
    z1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    z2 = z1.clone()
    temperature = 0.5

    monkeypatch.setattr(dist_utils, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist_utils, "get_rank", lambda: 1)

    def _fake_all_gather(t: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(t)
        return torch.cat([zeros, t.detach()], dim=0)

    monkeypatch.setattr(dist_utils, "all_gather_tensor", _fake_all_gather)

    loss_fn = NTXentLoss(
        temperature=temperature,
        normalize=False,
        mode="paired",
        cross_rank_negatives=True,
    )
    loss, _ = loss_fn(z1, z2)

    z1_all = _fake_all_gather(z1)
    z2_all = _fake_all_gather(z2)
    offset = z1.shape[0]
    targets = torch.arange(z1.shape[0], dtype=torch.long) + offset
    logits_i = (z1 @ z2_all.t()) / temperature
    logits_j = (z2 @ z1_all.t()) / temperature
    expected = 0.5 * (
        F.cross_entropy(logits_i, targets) + F.cross_entropy(logits_j, targets)
    )
    assert torch.allclose(loss, expected, atol=1e-6)


def test_ntxent_cross_rank_two_n_matches_manual(monkeypatch):
    z1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    z2 = z1.clone()
    temperature = 0.25

    monkeypatch.setattr(dist_utils, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist_utils, "get_rank", lambda: 0)

    def _fake_all_gather(t: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(t)
        return torch.cat([t.detach(), zeros], dim=0)

    monkeypatch.setattr(dist_utils, "all_gather_tensor", _fake_all_gather)

    loss_fn = NTXentLoss(
        temperature=temperature,
        normalize=False,
        mode="2N",
        cross_rank_negatives=True,
    )
    loss, _ = loss_fn(z1, z2)

    z1_all = _fake_all_gather(z1)
    z2_all = _fake_all_gather(z2)
    anchors = torch.cat([z1, z2], dim=0)
    all_feats = torch.cat([z1_all, z2_all], dim=0)
    logits = anchors @ all_feats.t() / temperature
    neg_inf = torch.finfo(logits.dtype).min
    mask = torch.zeros_like(logits, dtype=torch.bool)
    idx = torch.arange(z1.shape[0])
    mask[idx, idx] = True
    mask[idx + z1.shape[0], z1_all.shape[0] + idx] = True
    logits = logits.masked_fill(mask, neg_inf)
    targets = torch.cat([idx + z1_all.shape[0], idx], dim=0)
    expected = F.cross_entropy(logits, targets)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_ntxent_updates_label_supervised_mixture():
    estimator = MixtureStats(
        K=3,
        assigner="label_supervised",
        mode="label_supervised",
        enabled=True,
    )
    try:
        _set_active_estimator(estimator)
        loss_fn = NTXentLoss()
        z1 = torch.randn(4, 8)
        z2 = torch.randn(4, 8)
        labels = torch.tensor([0, 1, 2, 1], dtype=torch.long)
        loss, _ = loss_fn(z1, z2, labels=labels)
        assert torch.isfinite(loss)
        stored = estimator._last_stats  # type: ignore[attr-defined]
        assert stored is not None
        assert "pi" in stored and stored["pi"].numel() == estimator.K
    finally:
        _set_active_estimator(None)
        estimator.close()
