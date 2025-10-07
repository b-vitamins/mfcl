from __future__ import annotations

from typing import Dict, List, Optional

import torch


@torch.no_grad()
def spectral_diagnostics(
    Y: torch.Tensor,
    r_list: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute condition number, eigen-mass ratios, and kurtosis stats for Y.
    """
    if r_list is None:
        r_list = [8, 16, 32]

    mu = Y.mean(dim=0)
    Yc = Y - mu
    N, d = Y.shape

    Sigma = (Yc.T @ Yc) / float(N)
    evals = torch.linalg.eigvalsh(Sigma).clamp_min(0)

    lam_min = float(evals[0].item())
    lam_max = float(evals[-1].item())
    kappa = float(lam_max / max(lam_min, 1e-12))
    lam_sum = float(evals.sum().item())

    evals_desc = evals.flip(0)
    cums = torch.cumsum(evals_desc, dim=0)
    out: Dict[str, float] = {
        "kappa": kappa,
        "trace": lam_sum,
    }
    for r in r_list:
        r_eff = min(int(r), d)
        rho = float(cums[r_eff - 1].item() / max(lam_sum, 1e-12))
        out[f"rho_top{r_eff}"] = rho

    xs = Yc
    m2 = (xs * xs).mean(dim=0)
    m4 = (xs**4).mean(dim=0)
    safe = torch.clamp(m2, min=1e-12)
    kurt = m4 / (safe * safe) - 3.0
    out["kurtosis_mean"] = float(kurt.mean().item())
    out["kurtosis_max"] = float(kurt.max().item())

    return out
