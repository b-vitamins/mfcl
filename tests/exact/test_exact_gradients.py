import torch

from mfcl.exact.gradients import exact_gradients


@torch.no_grad()
def _rand_unit(n: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _loss_infonce_sym(x: torch.Tensor, y: torch.Tensor, tau: float) -> torch.Tensor:
    beta = 1.0 / tau
    n = x.shape[0]
    s_xy = beta * (x @ y.t())
    s_yx = beta * (y @ x.t())
    align = -(beta / n) * torch.sum(torch.sum(x * y, dim=1))
    norm = (torch.logsumexp(s_xy, dim=1).sum() + torch.logsumexp(s_yx, dim=1).sum()) / (2 * n)
    return align + norm


def _loss_loob_sym(x: torch.Tensor, y: torch.Tensor, tau: float) -> torch.Tensor:
    beta = 1.0 / tau
    n = x.shape[0]
    s_xy = beta * (x @ y.t())
    s_yx = beta * (y @ x.t())
    diag = torch.arange(n)
    s_xy[diag, diag] = float("-inf")
    s_yx[diag, diag] = float("-inf")
    align = -(beta / n) * torch.sum(torch.sum(x * y, dim=1))
    norm = (torch.logsumexp(s_xy, dim=1).sum() + torch.logsumexp(s_yx, dim=1).sum()) / (2 * n)
    return align + norm


def test_exact_gradients_match_autograd_infonce() -> None:
    torch.manual_seed(0)
    n, d = 16, 8
    tau = 0.25
    x = _rand_unit(n, d, seed=30).clone().requires_grad_(True)
    y = _rand_unit(n, d, seed=31).clone().requires_grad_(True)

    loss = _loss_infonce_sym(x, y, tau)
    g_auto = torch.autograd.grad(loss, (x, y))
    g_exact = exact_gradients(x.detach(), y.detach(), tau=tau, objective="infonce", chunk_size=12)

    def cosine(u: torch.Tensor, v: torch.Tensor) -> float:
        return float((u.flatten() @ v.flatten()).item() / (u.norm() * v.norm()).clamp_min(1e-12))

    assert cosine(g_exact[0], g_auto[0]) > 0.6
    assert cosine(g_exact[1], g_auto[1]) > 0.6


def test_exact_gradients_match_autograd_loob() -> None:
    torch.manual_seed(1)
    n, d = 14, 6
    tau = 0.5
    x = _rand_unit(n, d, seed=40).clone().requires_grad_(True)
    y = _rand_unit(n, d, seed=41).clone().requires_grad_(True)

    loss = _loss_loob_sym(x, y, tau)
    g_auto = torch.autograd.grad(loss, (x, y))
    g_exact = exact_gradients(x.detach(), y.detach(), tau=tau, objective="loob", chunk_size=10)

    def cosine(u: torch.Tensor, v: torch.Tensor) -> float:
        return float((u.flatten() @ v.flatten()).item() / (u.norm() * v.norm()).clamp_min(1e-12))

    assert cosine(g_exact[0], g_auto[0]) > 0.6
    assert cosine(g_exact[1], g_auto[1]) > 0.6
