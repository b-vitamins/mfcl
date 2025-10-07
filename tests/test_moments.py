import torch

from mfcl.approx.moments import compute_neg_stats


@torch.no_grad()
def _rand_unit(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return X


@torch.no_grad()
def test_compute_neg_stats_shapes_and_psd():
    N, d = 256, 64
    Y = _rand_unit(N, d, seed=1)

    iso = compute_neg_stats(Y, variant="iso")
    assert (
        iso.variant == "iso"
        and iso.alpha is not None
        and iso.diag is None
        and iso.Sigma is None
    )

    diag = compute_neg_stats(Y, variant="diag")
    assert diag.variant == "diag" and diag.diag is not None and diag.diag.shape == (d,)
    assert (diag.diag >= 0).all()

    full = compute_neg_stats(Y, variant="full", shrinkage=0.05)
    assert (
        full.variant == "full" and full.Sigma is not None and full.Sigma.shape == (d, d)
    )
    eig = torch.linalg.eigvalsh(full.Sigma)
    assert float(eig.min().item()) >= -1e-6


@torch.no_grad()
def test_lowrank_quad_matches_dense():
    N, d = 384, 96
    Y = _rand_unit(N, d, seed=2)
    full = compute_neg_stats(Y, variant="full", shrinkage=0.05, ridge=1e-5)
    lr = compute_neg_stats(Y, variant="full", shrinkage=0.05, ridge=1e-5, lowrank_r=16)

    X = _rand_unit(200, d, seed=3)
    q_dense = full.quad(X)
    q_lr = lr.quad(X)

    rel = (q_lr - q_dense).abs() / q_dense.clamp_min(1e-8)
    assert float(rel.mean().item()) < 0.06
    assert float(rel.max().item()) < 0.20
