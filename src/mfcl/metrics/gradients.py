from __future__ import annotations

from typing import Dict

import torch


@torch.no_grad()
def cosine(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> float:
    num = (u * v).sum()
    den = u.norm().clamp_min(eps) * v.norm().clamp_min(eps)
    return float((num / den).item())


@torch.no_grad()
def per_row_cosine(
    U: torch.Tensor, V: torch.Tensor, eps: float = 1e-12
) -> Dict[str, float]:
    U = U.detach().to("cpu")
    V = V.detach().to("cpu")
    num = (U * V).sum(dim=1)
    den = U.norm(dim=1).clamp_min(eps) * V.norm(dim=1).clamp_min(eps)
    c = (num / den).clamp(min=-1.0, max=1.0)
    return {
        "row_cos_mean": float(c.mean().item()),
        "row_cos_median": float(c.median().item()),
        "row_cos_q25": float(c.quantile(0.25).item()),
        "row_cos_q75": float(c.quantile(0.75).item()),
    }


@torch.no_grad()
def relative_norm_error(U: torch.Tensor, V: torch.Tensor, eps: float = 1e-12) -> float:
    U = U.detach().to("cpu")
    V = V.detach().to("cpu")
    num = (U - V).norm()
    den = V.norm().clamp_min(eps)
    return float((num / den).item())


@torch.no_grad()
def sign_agreement_topk_per_row(
    U: torch.Tensor,
    V: torch.Tensor,
    k: int = 10,
) -> float:
    U = U.detach().to("cpu")
    V = V.detach().to("cpu")
    N, d = V.shape
    if k <= 0:
        return float("nan")
    k = min(k, d)
    idx = torch.topk(V.abs(), k=k, dim=1, largest=True, sorted=False).indices
    ar = torch.arange(N).unsqueeze(1)
    sign_u = torch.sign(U[ar, idx])
    sign_v = torch.sign(V[ar, idx])
    agree = (sign_u == sign_v).float().mean(dim=1)
    return float(agree.mean().item())


@torch.no_grad()
def gradient_metrics(
    gX_mf: torch.Tensor,
    gY_mf: torch.Tensor,
    gX_ex: torch.Tensor,
    gY_ex: torch.Tensor,
    *,
    topk: int = 10,
) -> Dict[str, float]:
    flat_mf = torch.cat([gX_mf.reshape(-1), gY_mf.reshape(-1)], dim=0)
    flat_ex = torch.cat([gX_ex.reshape(-1), gY_ex.reshape(-1)], dim=0)

    out = {
        "cos_all": cosine(flat_mf, flat_ex),
        "rel_norm_err_all": relative_norm_error(flat_mf, flat_ex),
        "cos_X": cosine(gX_mf.reshape(-1), gX_ex.reshape(-1)),
        "cos_Y": cosine(gY_mf.reshape(-1), gY_ex.reshape(-1)),
        "rel_norm_err_X": relative_norm_error(gX_mf, gX_ex),
        "rel_norm_err_Y": relative_norm_error(gY_mf, gY_ex),
        "sign_agree_topk_X": sign_agreement_topk_per_row(gX_mf, gX_ex, k=topk),
        "sign_agree_topk_Y": sign_agreement_topk_per_row(gY_mf, gY_ex, k=topk),
    }
    out.update({f"X_{k}": v for k, v in per_row_cosine(gX_mf, gX_ex).items()})
    out.update({f"Y_{k}": v for k, v in per_row_cosine(gY_mf, gY_ex).items()})
    return out
