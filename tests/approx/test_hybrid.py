import math

import pytest
import torch

from mfcl.approx.hybrid import (
    _tail_logmass,
    hybrid_infonce_logmass,
    hybrid_loob_centered,
    topk_neg_sim_streaming,
)
from mfcl.approx.moments import compute_neg_stats
from mfcl.exact.logZ import exact_infonce_logmass, exact_loob_logmass_centered


@torch.no_grad()
def _rand_unit(n: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def test_topk_neg_sim_streaming_matches_naive() -> None:
    x = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    x = torch.nn.functional.normalize(x, dim=1)
    y = x.clone()

    topk, max_neg = topk_neg_sim_streaming(x, y, k=2, chunk_size=2)
    assert topk.shape == (4, 2)
    assert torch.all(max_neg <= 1.0)

    full_scores = x @ y.t()
    for i in range(4):
        # remove positive
        scores = torch.cat([full_scores[i, :i], full_scores[i, i + 1 :]])
        expected_topk, _ = torch.topk(scores, k=2)
        assert torch.allclose(topk[i], expected_topk)
        assert math.isclose(float(max_neg[i].item()), float(scores.max().item()), rel_tol=1e-6)


@torch.no_grad()
def test_topk_neg_sim_streaming_handles_zero_k() -> None:
    x = _rand_unit(5, 4, seed=10)
    y = _rand_unit(5, 4, seed=11)
    topk, max_neg = topk_neg_sim_streaming(x, y, k=0, chunk_size=2)
    assert topk.shape == (5, 0)
    assert max_neg.shape == (5,)
    naive = x @ y.t()
    naive.fill_diagonal_(float("-inf"))
    assert torch.allclose(max_neg, naive.max(dim=1).values)


@torch.no_grad()
def test_tail_logmass_matches_manual_expression() -> None:
    x = _rand_unit(6, 3, seed=12)
    y = _rand_unit(6, 3, seed=13)
    stats = compute_neg_stats(y, variant="diag")
    beta = 2.0
    k = 2

    tail = _tail_logmass(beta, y.shape[0], k, x, stats)

    m = max(y.shape[0] - 1 - k, 0)
    mu_proj = (x @ stats.mu)
    quad = stats.quad(x)
    expected = math.log(m) + beta * mu_proj + 0.5 * (beta**2) * quad
    assert torch.allclose(tail, expected)


@torch.no_grad()
def test_hybrid_equals_exact_when_k_full() -> None:
    n, d = 24, 16
    tau = 0.3
    x = _rand_unit(n, d, seed=21)
    y = _rand_unit(n, d, seed=22)
    stats = compute_neg_stats(y, variant="diag")

    h_loob, gap_loob = hybrid_loob_centered(x, y, tau=tau, k=n - 1, stats=stats)
    e_loob = exact_loob_logmass_centered(x, y, tau=tau, chunk_size=16)
    assert torch.allclose(h_loob, e_loob, atol=1e-6)
    assert torch.all(gap_loob > 0)

    h_info, gap_info = hybrid_infonce_logmass(x, y, tau=tau, k=n - 1, stats=stats)
    e_info = exact_infonce_logmass(x, y, tau=tau, chunk_size=16)
    assert torch.allclose(h_info, e_info, atol=1e-6)
    assert torch.all(gap_info > 0)


@torch.no_grad()
def test_gap_trigger_falls_back_to_mf2() -> None:
    # Highly symmetric batch makes the tail dominate and triggers the fallback.
    x = torch.eye(4, dtype=torch.float32)
    y = x.clone()
    tau = 0.5
    stats = compute_neg_stats(y, variant="iso")

    logz_hybrid, gap = hybrid_infonce_logmass(
        x,
        y,
        tau=tau,
        k=0,
        stats=stats,
        covariance="iso",
        gap_trigger=0.0,
    )
    assert torch.all(gap <= 0)

    beta = 1.0 / tau
    mu_proj = x @ stats.mu
    quad = stats.quad(x)
    m_neg = math.log(max(x.shape[0] - 1, 1)) + beta * mu_proj + 0.5 * (beta**2) * quad
    a_pos = beta * (x * y).sum(dim=1)
    expected_mf2 = m_neg + torch.nn.functional.softplus(a_pos - m_neg)
    assert torch.allclose(logz_hybrid, expected_mf2)


@torch.no_grad()
def test_hybrid_loob_gap_trigger_matches_mf2_centered() -> None:
    x = torch.eye(3, dtype=torch.float32)
    y = x.clone()
    tau = 0.7
    stats = compute_neg_stats(y, variant="iso")

    centered, gap = hybrid_loob_centered(
        x,
        y,
        tau=tau,
        k=0,
        stats=stats,
        covariance="iso",
        gap_trigger=0.0,
    )
    assert torch.all(gap <= 0)

    beta = 1.0 / tau
    mu_proj = x @ stats.mu
    quad = stats.quad(x)
    expected = beta * mu_proj + 0.5 * (beta**2) * quad
    assert torch.allclose(centered, expected)
