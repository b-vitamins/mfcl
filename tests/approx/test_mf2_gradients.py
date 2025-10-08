import torch

from mfcl.approx.gradients import _apply_cov, mf2_gradients
from mfcl.approx.moments import compute_neg_stats
from mfcl.exact.gradients import exact_gradients


@torch.no_grad()
def _rand_unit(n: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def test_apply_cov_uses_lowrank_decomposition() -> None:
    y = _rand_unit(32, 8, seed=0)
    stats = compute_neg_stats(y, variant="full", lowrank_r=2, shrinkage=0.1, ridge=1e-4)
    x = _rand_unit(16, 8, seed=1)

    cov_lowrank = _apply_cov(stats, x)

    # Reconstruct the dense covariance manually from the low-rank parameters
    assert stats.U is not None and stats.lam is not None and stats.alpha_tail is not None
    sigma_dense = stats.U @ torch.diag(stats.lam) @ stats.U.t() + stats.alpha_tail * torch.eye(
        x.shape[1]
    )
    cov_dense = x @ sigma_dense
    assert torch.allclose(cov_lowrank, cov_dense, atol=1e-5)


def test_mf2_gradients_direction_reasonable() -> None:
    torch.manual_seed(2)
    n, d = 32, 12
    tau = 0.25
    x = _rand_unit(n, d, seed=10)
    y = _rand_unit(n, d, seed=11)

    g_exact = exact_gradients(x, y, tau=tau, objective="infonce", chunk_size=16)
    g_mf = mf2_gradients(x, y, tau=tau, objective="infonce", covariance="diag")

    def cos(u: torch.Tensor, v: torch.Tensor) -> float:
        return float((u.flatten() @ v.flatten()).item() / (u.norm() * v.norm()).clamp_min(1e-12))

    assert cos(g_exact[0], g_mf[0]) > 0.6
    assert cos(g_exact[1], g_mf[1]) > 0.6


def test_mf2_gradients_respect_objective_switch() -> None:
    n, d = 16, 6
    tau = 0.4
    x = _rand_unit(n, d, seed=20)
    y = _rand_unit(n, d, seed=21)

    g_infonce = mf2_gradients(x, y, tau=tau, objective="infonce", covariance="iso")
    g_loob = mf2_gradients(x, y, tau=tau, objective="loob", covariance="iso")
    # Direction should differ between objectives.
    assert not torch.allclose(g_infonce[0], g_loob[0])
    assert not torch.allclose(g_infonce[1], g_loob[1])
