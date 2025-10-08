import torch

from mfcl.exact.logZ import (
    _streaming_logsumexp_xy,
    exact_infonce_logmass,
    exact_loob_logmass_centered,
)


@torch.no_grad()
def _rand_unit(n: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def test_streaming_logsumexp_matches_dense() -> None:
    n, d = 20, 6
    tau = 0.4
    beta = 1.0 / tau
    x = _rand_unit(n, d, seed=10)
    y = _rand_unit(n, d, seed=11)

    logz_stream, diag = _streaming_logsumexp_xy(x, y, beta=beta, chunk_size=5, exclude_pos=False)
    scores = beta * (x @ y.t())
    logz_dense = torch.logsumexp(scores, dim=1)
    diag_dense = beta * torch.sum(x * y, dim=1)

    assert torch.allclose(logz_stream, logz_dense, atol=1e-6)
    assert torch.allclose(diag, diag_dense, atol=1e-6)


@torch.no_grad()
def test_streaming_logsumexp_excludes_positives() -> None:
    n, d = 12, 5
    tau = 0.7
    beta = 1.0 / tau
    x = _rand_unit(n, d, seed=20)
    y = _rand_unit(n, d, seed=21)

    logz_excl, diag = _streaming_logsumexp_xy(x, y, beta=beta, chunk_size=4, exclude_pos=True)
    scores = beta * (x @ y.t())
    scores.fill_diagonal_(float("-inf"))
    dense = torch.logsumexp(scores, dim=1)

    assert torch.allclose(logz_excl, dense, atol=1e-6)
    assert torch.allclose(diag, beta * torch.sum(x * y, dim=1), atol=1e-6)


@torch.no_grad()
def test_exact_infonce_matches_naive() -> None:
    n, d = 64, 16
    tau = 0.25
    x = _rand_unit(n, d, seed=30)
    y = _rand_unit(n, d, seed=31)

    logz = exact_infonce_logmass(x, y, tau=tau, chunk_size=32)
    scores = (1.0 / tau) * (x @ y.t())
    naive = torch.logsumexp(scores, dim=1)
    diff = (logz - naive).abs()
    assert float(diff.max().item()) < 1e-5


@torch.no_grad()
def test_exact_loob_centered_matches_naive_centered() -> None:
    n, d = 48, 12
    tau = 0.5
    x = _rand_unit(n, d, seed=40)
    y = _rand_unit(n, d, seed=41)

    centered = exact_loob_logmass_centered(x, y, tau=tau, chunk_size=24)
    scores = (1.0 / tau) * (x @ y.t())
    scores.fill_diagonal_(float("-inf"))
    naive = torch.logsumexp(scores, dim=1) - torch.log(torch.tensor(float(n - 1)))
    diff = (centered - naive).abs()
    assert float(diff.max().item()) < 1e-5
