import torch

from mfcl.exact.logZ import exact_infonce_logmass, exact_loob_logmass_centered


@torch.no_grad()
def _rand_unit(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return X


@torch.no_grad()
def test_exact_infonce_matches_naive():
    N, d = 96, 32
    tau = 0.25
    beta = 1.0 / tau
    X = _rand_unit(N, d, seed=10)
    Y = _rand_unit(N, d, seed=11)

    logZ = exact_infonce_logmass(X, Y, tau=tau, chunk_size=48)

    S = beta * (X @ Y.t())
    naive = torch.logsumexp(S, dim=1)

    diff = (logZ - naive).abs()
    assert float(diff.max().item()) < 1e-5


@torch.no_grad()
def test_exact_loob_centered_matches_naive_centered():
    N, d = 80, 20
    tau = 0.5
    beta = 1.0 / tau
    X = _rand_unit(N, d, seed=12)
    Y = _rand_unit(N, d, seed=13)

    cent = exact_loob_logmass_centered(X, Y, tau=tau, chunk_size=40)

    S = beta * (X @ Y.t())
    S[torch.arange(N), torch.arange(N)] = float("-inf")
    naive = torch.logsumexp(S, dim=1) - torch.log(torch.tensor(float(N - 1)))

    diff = (cent - naive).abs()
    assert float(diff.max().item()) < 1e-5
