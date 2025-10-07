from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from mfcl.approx.moments import NegStats, CovarianceKind, compute_neg_stats, proj_mu


@torch.no_grad()
def topk_neg_sim_streaming(
    X: torch.Tensor,
    Y: torch.Tensor,
    k: int,
    *,
    chunk_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Streaming top-k over negatives for each anchor i.

    Returns:
      - topk_vals: [N, k_eff] top-k dot products x_i^T y_j, descending
      - max_neg:  [N] the maximum negative dot per anchor
    Notes:
      - Excludes j == i from candidates.
      - If k <= 0, returns an empty [N, 0] tensor and max_neg for convenience.
    Complexity: O(N^2 d) compute like exact, but never materializes N×N.
    """
    device = X.device
    N, d = X.shape
    assert Y.shape == (N, d)
    k_eff = max(0, min(k, N - 1))

    # Initialize running top-k values; start as -inf so any score wins.
    if k_eff > 0:
        topk_vals = torch.full(
            (N, k_eff), float("-inf"), device=device, dtype=torch.float32
        )
    else:
        topk_vals = torch.empty((N, 0), device=device, dtype=torch.float32)

    max_neg = torch.full((N,), float("-inf"), device=device, dtype=torch.float32)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        Yc = Y[start:end]  # [C, d]
        scores = X @ Yc.t()  # [N, C]

        # mask out positives where row index equals global column index
        rows = torch.arange(start, end, device=device)
        scores[rows, rows - start] = float("-inf")

        # update max
        m_chunk, _ = scores.max(dim=1)
        max_neg = torch.maximum(max_neg, m_chunk)

        if k_eff == 0:
            continue

        # merge into running top-k: concatenate and take top-k again
        merged = torch.cat([topk_vals, scores], dim=1)  # [N, k_eff + C]
        topk_vals, _ = torch.topk(merged, k=k_eff, dim=1, largest=True, sorted=True)

    return topk_vals, max_neg


@torch.no_grad()
def _tail_logmass(
    beta: float,
    N: int,
    k: int,
    X: torch.Tensor,
    stats: NegStats,
) -> torch.Tensor:
    """
    MF2 tail mass:
      log T_i ≈ log M + β x_i^T μ + 0.5 β^2 x_i^T Σ x_i,  where M = N - 1 - k
    """
    M = max(N - 1 - max(0, k), 0)
    if M == 0:
        # No tail left; return -inf so it doesn't contribute to logsumexp
        return torch.full(
            (X.shape[0],), float("-inf"), device=X.device, dtype=torch.float32
        )
    mu_proj = proj_mu(X, stats.mu)
    quad = stats.quad(X)
    return math.log(M) + beta * mu_proj + 0.5 * (beta**2) * quad


@torch.no_grad()
def hybrid_loob_centered(
    X: torch.Tensor,
    Y: torch.Tensor,
    tau: float,
    *,
    k: int,
    stats: Optional[NegStats] = None,
    covariance: CovarianceKind = "diag",
    shrinkage: float = 0.0,
    ridge: float = 1e-5,
    chunk_size: int = 4096,
    gap_trigger: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Head–tail splice for LOOB centered log-mass:
      log sum_{j != i} e^{β x_i^T y_j} ≈ logsumexp( head_logsumexp, tail_logmass )
      then subtract log(N-1) to center.

    Returns:
      (approx_vector [N], gap [N]) where gap = max_neg_logit - tail_logmass
    If gap_trigger is provided, uses pure MF2 for anchors with gap <= gap_trigger.
    """
    beta = 1.0 / float(tau)
    N = X.shape[0]
    if stats is None:
        stats = compute_neg_stats(
            Y, variant=covariance, shrinkage=shrinkage, ridge=ridge
        )

    # top-k similarities (beta-independent), plus per-anchor max negative
    k_eff = max(0, min(k, N - 1))
    # compute at least max_neg even if k=0
    need_k = max(k_eff, 1)
    topk_vals, max_neg_raw = topk_neg_sim_streaming(X, Y, need_k, chunk_size=chunk_size)
    if k_eff == 0:
        head_logsumexp = torch.full(
            (N,), float("-inf"), device=X.device, dtype=torch.float32
        )
    else:
        head_vals = topk_vals[:, :k_eff]  # [N, k]
        head_logsumexp = torch.logsumexp(beta * head_vals, dim=1)  # [N]

    tail_logmass = _tail_logmass(beta, N, k_eff, X, stats)  # [N]
    # negative max logit in β scale:
    max_neg = beta * max_neg_raw  # [N]
    gap = max_neg - tail_logmass  # [N]

    # Combine head and tail
    m = torch.maximum(head_logsumexp, tail_logmass)
    logZ_neg = m + torch.log(
        torch.exp(head_logsumexp - m) + torch.exp(tail_logmass - m)
    )
    centered = logZ_neg - math.log(max(N - 1, 1))

    if gap_trigger is not None:
        # Pure MF2 centered LOOB
        mu_proj = proj_mu(X, stats.mu)
        quad = stats.quad(X)
        mf2_centered = beta * mu_proj + 0.5 * (beta**2) * quad
        use_hybrid = gap > float(gap_trigger)
        centered = torch.where(use_hybrid, centered, mf2_centered)

    return centered, gap


@torch.no_grad()
def hybrid_infonce_logmass(
    X: torch.Tensor,
    Y: torch.Tensor,
    tau: float,
    *,
    k: int,
    stats: Optional[NegStats] = None,
    covariance: CovarianceKind = "diag",
    shrinkage: float = 0.0,
    ridge: float = 1e-5,
    chunk_size: int = 4096,
    gap_trigger: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Head–tail splice for InfoNCE log normalization:
      log Z_i ≈ logsumexp( a_pos, head_logsumexp, tail_logmass )
    Returns:
      (approx_vector [N], gap [N]) with gap defined on the negative tail as in LOOB.
    If gap_trigger is provided, anchors with gap <= gap_trigger fall back to MF2 InfoNCE.
    """
    beta = 1.0 / float(tau)
    N = X.shape[0]
    if stats is None:
        stats = compute_neg_stats(
            Y, variant=covariance, shrinkage=shrinkage, ridge=ridge
        )

    # top-k similarities
    k_eff = max(0, min(k, N - 1))
    need_k = max(k_eff, 1)
    topk_vals, max_neg_raw = topk_neg_sim_streaming(X, Y, need_k, chunk_size=chunk_size)
    if k_eff == 0:
        head_logsumexp = torch.full(
            (N,), float("-inf"), device=X.device, dtype=torch.float32
        )
    else:
        head_vals = topk_vals[:, :k_eff]
        head_logsumexp = torch.logsumexp(beta * head_vals, dim=1)

    tail_logmass = _tail_logmass(beta, N, k_eff, X, stats)
    max_neg = beta * max_neg_raw
    gap = max_neg - tail_logmass

    a_pos = beta * (X * Y).sum(dim=1)  # [N]
    # Stack and logsumexp over three terms; handle -inf correctly
    logZ = torch.logsumexp(
        torch.stack([a_pos, head_logsumexp, tail_logmass], dim=1), dim=1
    )

    if gap_trigger is not None:
        # MF2 InfoNCE: m_neg + softplus(a - m_neg)
        mu_proj = proj_mu(X, stats.mu)
        quad = stats.quad(X)
        m_neg = math.log(max(N - 1, 1)) + beta * mu_proj + 0.5 * (beta**2) * quad
        mf2 = m_neg + F.softplus(a_pos - m_neg)
        use_hybrid = gap > float(gap_trigger)
        logZ = torch.where(use_hybrid, logZ, mf2)

    return logZ, gap
