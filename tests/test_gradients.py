import torch

from mfcl.exact.gradients import exact_gradients
from mfcl.approx.gradients import mf2_gradients


@torch.no_grad()
def _rand_unit(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return X


def _loss_infonce_sym(X, Y, tau):
    beta = 1.0 / tau
    N = X.shape[0]
    Sxy = beta * (X @ Y.t())
    Syx = beta * (Y @ X.t())
    align = -(beta / N) * torch.sum(torch.sum(X * Y, dim=1))
    norm = (torch.logsumexp(Sxy, dim=1).sum() + torch.logsumexp(Syx, dim=1).sum()) / (
        2 * N
    )
    return align + norm


def _loss_loob_sym(X, Y, tau):
    beta = 1.0 / tau
    N = X.shape[0]
    Sxy = beta * (X @ Y.t())
    Syx = beta * (Y @ X.t())
    diag = torch.arange(N)
    Sxy[diag, diag] = float("-inf")
    Syx[diag, diag] = float("-inf")
    align = -(beta / N) * torch.sum(torch.sum(X * Y, dim=1))
    norm = (torch.logsumexp(Sxy, dim=1).sum() + torch.logsumexp(Syx, dim=1).sum()) / (
        2 * N
    )
    return align + norm


def test_exact_gradients_match_autograd_infonce():
    torch.manual_seed(0)
    N, d = 24, 12
    tau = 0.25
    X = _rand_unit(N, d, seed=30).clone().requires_grad_(True)
    Y = _rand_unit(N, d, seed=31).clone().requires_grad_(True)

    L = _loss_infonce_sym(X, Y, tau)
    gX_auto, gY_auto = torch.autograd.grad(L, (X, Y))

    gX_ex, gY_ex = exact_gradients(
        X.detach(), Y.detach(), tau=tau, objective="infonce", chunk_size=16
    )

    cos_x = (gX_ex.flatten() @ gX_auto.flatten()).item() / (
        gX_ex.norm() * gX_auto.norm()
    ).clamp_min(1e-12)
    cos_y = (gY_ex.flatten() @ gY_auto.flatten()).item() / (
        gY_ex.norm() * gY_auto.norm()
    ).clamp_min(1e-12)
    assert cos_x > 0.5 and cos_y > 0.5


def test_exact_gradients_match_autograd_loob():
    torch.manual_seed(1)
    N, d = 20, 10
    tau = 0.5
    X = _rand_unit(N, d, seed=40).clone().requires_grad_(True)
    Y = _rand_unit(N, d, seed=41).clone().requires_grad_(True)

    L = _loss_loob_sym(X, Y, tau)
    gX_auto, gY_auto = torch.autograd.grad(L, (X, Y))

    gX_ex, gY_ex = exact_gradients(
        X.detach(), Y.detach(), tau=tau, objective="loob", chunk_size=16
    )

    cos_x = (gX_ex.flatten() @ gX_auto.flatten()).item() / (
        gX_ex.norm() * gX_auto.norm()
    ).clamp_min(1e-12)
    cos_y = (gY_ex.flatten() @ gY_auto.flatten()).item() / (
        gY_ex.norm() * gY_auto.norm()
    ).clamp_min(1e-12)
    assert cos_x > 0.5 and cos_y > 0.5


def test_mf2_gradients_direction_reasonable():
    torch.manual_seed(2)
    N, d = 48, 16
    tau = 0.25
    X = _rand_unit(N, d, seed=50)
    Y = _rand_unit(N, d, seed=51)
    gX_ex, gY_ex = exact_gradients(X, Y, tau=tau, objective="infonce", chunk_size=24)
    gX_mf, gY_mf = mf2_gradients(X, Y, tau=tau, objective="infonce", covariance="diag")

    def cos(u, v):
        return float(
            (u.flatten() @ v.flatten()).item() / (u.norm() * v.norm()).clamp_min(1e-12)
        )

    assert cos(gX_ex, gX_mf) > 0.6
    assert cos(gY_ex, gY_mf) > 0.6
