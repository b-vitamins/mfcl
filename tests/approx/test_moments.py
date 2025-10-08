import math

import pytest
import torch

from mfcl.approx.moments import (
    NegStats,
    compute_neg_stats,
    ema_update_diag,
    negstats_from_diag,
    proj_mu,
)


@torch.no_grad()
def _rand_unit(n: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


@pytest.mark.parametrize(
    "variant, seed",
    [("iso", 0), ("diag", 1), ("full", 2)],
)
@torch.no_grad()
def test_compute_neg_stats_basic_properties(variant: str, seed: int) -> None:
    y = _rand_unit(128, 32, seed=seed)
    stats = compute_neg_stats(y, variant=variant, shrinkage=0.1, ridge=1e-4)

    assert isinstance(stats, NegStats)
    assert stats.variant == variant
    assert stats.mu.shape == (y.shape[1],)

    if variant == "iso":
        assert stats.alpha is not None and stats.alpha > 0
        assert stats.diag is None and stats.Sigma is None
    elif variant == "diag":
        assert stats.diag is not None and stats.diag.shape == (y.shape[1],)
        assert torch.all(stats.diag > 0)
    else:
        assert stats.Sigma is not None and stats.Sigma.shape == (y.shape[1], y.shape[1])
        eigvals = torch.linalg.eigvalsh(stats.Sigma)
        assert float(eigvals.min().item()) >= -1e-6


@torch.no_grad()
def test_compute_neg_stats_lowrank_quad_matches_manual() -> None:
    y = _rand_unit(192, 48, seed=1)
    stats = compute_neg_stats(
        y,
        variant="full",
        shrinkage=0.05,
        ridge=1e-4,
        lowrank_r=12,
    )
    assert stats.U is not None and stats.lam is not None and stats.alpha_tail is not None

    x = _rand_unit(64, 48, seed=2)
    quad = stats.quad(x)

    proj = x @ stats.U
    head = (proj * proj) @ stats.lam
    tail = stats.alpha_tail * (x * x).sum(dim=1)
    expected = head + tail

    assert torch.allclose(quad, expected, atol=1e-6)


@torch.no_grad()
def test_compute_neg_stats_rejects_unknown_variant() -> None:
    y = _rand_unit(8, 3, seed=3)
    with pytest.raises(ValueError):
        compute_neg_stats(y, variant="bad")  # type: ignore[arg-type]


@torch.no_grad()
def test_ema_update_diag_matches_manual_average() -> None:
    mu_prev = torch.tensor([1.0, -1.0])
    var_prev = torch.tensor([0.5, 2.0])
    y = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, -5.0]], dtype=torch.float32)
    decay = 0.8

    mu_batch = y.mean(dim=0)
    var_batch = ((y - mu_batch) ** 2).mean(dim=0)
    expected_mu = decay * mu_prev + (1 - decay) * mu_batch
    expected_var = decay * var_prev + (1 - decay) * var_batch

    mu_new, var_new = ema_update_diag(mu_prev, var_prev, y, decay=decay)
    assert torch.allclose(mu_new, expected_mu)
    assert torch.allclose(var_new, expected_var)


@torch.no_grad()
def test_negstats_from_diag_switches_between_iso_and_diag() -> None:
    mu = torch.tensor([0.0, 1.0, -1.0])
    var = torch.tensor([0.25, 0.5, 1.0])

    diag_stats = negstats_from_diag(mu, var, use_iso=False, ridge=0.1)
    assert diag_stats.variant == "diag"
    assert diag_stats.diag is not None
    assert torch.allclose(diag_stats.diag, var + 0.1)

    iso_stats = negstats_from_diag(mu, var, use_iso=True, ridge=0.1)
    assert iso_stats.variant == "iso"
    assert iso_stats.alpha is not None
    expected_alpha = float(var.mean().item()) + 0.1
    assert math.isclose(iso_stats.alpha, expected_alpha, rel_tol=1e-6)


@torch.no_grad()
def test_proj_mu_returns_inner_products() -> None:
    x = torch.tensor([[1.0, 2.0, 0.0], [0.5, -1.0, 1.0]])
    mu = torch.tensor([2.0, -1.0, 3.0])
    out = proj_mu(x, mu)
    expected = torch.tensor([0.0, 5.0])
    assert torch.allclose(out, expected)
