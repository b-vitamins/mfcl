from __future__ import annotations

import math
from typing import Tuple

import torch


@torch.no_grad()
def _streaming_logsumexp_xy(
    X: torch.Tensor,
    Y: torch.Tensor,
    beta: float,
    chunk_size: int = 4096,
    exclude_pos: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-row logsumexp(beta * x_i^T y_j) across all j, in a streaming fashion.
    Returns (logZ, diag_scores) where:
      - logZ: [N] log-sum-exp over all j
      - diag_scores: [N] with beta * x_i^T y_i (positive pairs)
    If exclude_pos=True, caller can subtract e^{diag - max} from streaming sums before final log.
    """
    device = X.device
    N, d = X.shape
    assert Y.shape == (N, d)

    # Running max and scaled sums for each i
    M = torch.full((N,), float("-inf"), device=device, dtype=torch.float32)
    S = torch.zeros((N,), device=device, dtype=torch.float32)

    # precompute diag scores in a single pass later; for now store to compute once
    diag_scores = torch.empty((N,), device=device, dtype=torch.float32)

    # Process Y in chunks (over columns j)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        Yc = Y[start:end]  # [C, d]
        # Scores: [N, C]
        scores = beta * (X @ Yc.t())
        # per-row max over this chunk
        m_chunk, _ = scores.max(dim=1)  # [N]
        # sum exp(score - m_chunk)
        s_chunk = torch.exp(scores - m_chunk[:, None]).sum(dim=1)  # [N]

        # Combine with running (M, S)
        # new M* = max(M, m_chunk)
        M_new = torch.maximum(M, m_chunk)
        # update S: if M >= m_chunk: S + exp(m_chunk - M) * s_chunk
        # else: S * exp(M - m_chunk) + s_chunk
        mask = M >= m_chunk
        S = torch.where(
            mask,
            S + torch.exp(m_chunk - M) * s_chunk,
            S * torch.exp(M - m_chunk) + s_chunk,
        )
        M = M_new

        # Fill diag scores for rows where j range includes i
        # For rows i in [start, end), the diagonal hits column (i - start)
        rows = torch.arange(start, end, device=device)
        # dot(x_i, y_i)
        diag_slice = beta * (X[rows] * Yc[rows - start]).sum(dim=1)
        diag_scores[rows] = diag_slice

    if exclude_pos:
        # subtract exp(diag - M) from S safely
        S = S - torch.exp(diag_scores - M)
        # numerical guard
        eps = 1e-30
        S = torch.clamp(S, min=eps)

    logZ = torch.log(S) + M
    return logZ, diag_scores


@torch.no_grad()
def exact_loob_logmass_centered(
    X: torch.Tensor,
    Y: torch.Tensor,
    tau: float,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """
    Compute centered LOOB normalization per anchor:
      \tilde{ell}_i = log sum_{j != i} exp(beta x_i^T y_j) - log(N - 1)
    Returns tensor [N].
    """
    beta = 1.0 / float(tau)
    N = X.shape[0]
    logZ, _ = _streaming_logsumexp_xy(
        X, Y, beta=beta, chunk_size=chunk_size, exclude_pos=True
    )
    return logZ - math.log(max(N - 1, 1))


@torch.no_grad()
def exact_infonce_logmass(
    X: torch.Tensor,
    Y: torch.Tensor,
    tau: float,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """
    Compute per-anchor InfoNCE normalization:
      ell_i = log sum_{j} exp(beta x_i^T y_j)
    Returns tensor [N].
    """
    beta = 1.0 / float(tau)
    logZ, _ = _streaming_logsumexp_xy(
        X, Y, beta=beta, chunk_size=chunk_size, exclude_pos=False
    )
    return logZ
