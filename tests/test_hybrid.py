import torch

from mfcl.approx.hybrid import hybrid_loob_centered, hybrid_infonce_logmass
from mfcl.approx.moments import compute_neg_stats
from mfcl.exact.logZ import exact_loob_logmass_centered, exact_infonce_logmass


@torch.no_grad()
def _rand_unit(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return X


@torch.no_grad()
def test_hybrid_equals_exact_when_k_full():
    N, d = 64, 32
    tau = 0.25
    X = _rand_unit(N, d, seed=21)
    Y = _rand_unit(N, d, seed=22)
    stats = compute_neg_stats(Y, variant="diag")

    h_loob, _ = hybrid_loob_centered(
        X, Y, tau=tau, k=N - 1, stats=stats, covariance="diag"
    )
    e_loob = exact_loob_logmass_centered(X, Y, tau=tau, chunk_size=64)
    assert float((h_loob - e_loob).abs().max().item()) < 1e-6

    h_info, _ = hybrid_infonce_logmass(
        X, Y, tau=tau, k=N - 1, stats=stats, covariance="diag"
    )
    e_info = exact_infonce_logmass(X, Y, tau=tau, chunk_size=64)
    assert float((h_info - e_info).abs().max().item()) < 1e-6
