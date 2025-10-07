from __future__ import annotations

from typing import Literal, Tuple

import torch

ObjectiveKind = Literal["loob", "infonce"]


@torch.no_grad()
def _row_softmax_expectation(
    X: torch.Tensor,
    Y: torch.Tensor,
    beta: float,
    *,
    chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Row-wise softmax over Y for each x_i:
      π^x_{i,j} = exp(β x_i^T y_j) / Z_i,  with Z_i = Σ_j exp(β x_i^T y_j)

    Returns:
      E_y[i]     = Σ_j π_{i,j} y_j           shape [N, d]
      logZ[i]    = log Z_i                   shape [N]
      a_pos[i]   = β x_i^T y_i               shape [N]
    Streaming and memory-safe. No N×N materialization.
    """
    device = X.device
    N, d = X.shape
    assert Y.shape == (N, d)

    # Running logsumexp stats per row
    M = torch.full((N,), float("-inf"), device=device, dtype=torch.float32)
    S = torch.zeros((N,), device=device, dtype=torch.float32)
    V = torch.zeros(
        (N, d), device=device, dtype=torch.float32
    )  # Σ exp(score - M) * y_j

    a_pos = torch.empty((N,), device=device, dtype=torch.float32)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        Yc = Y[start:end]  # [C, d]
        scores = beta * (X @ Yc.t())  # [N, C]

        # per-row chunk max and sums
        m_chunk, _ = scores.max(dim=1)  # [N]
        s_chunk = torch.exp(scores - m_chunk[:, None]).sum(dim=1)  # [N]
        v_chunk = torch.exp(scores - m_chunk[:, None]) @ Yc  # [N, d]

        # combine with running stats
        M_new = torch.maximum(M, m_chunk)
        mask = M >= m_chunk
        scale_old = torch.exp(m_chunk - M)
        scale_new = torch.exp(M - m_chunk)

        S = torch.where(mask, S + scale_old * s_chunk, S * scale_new + s_chunk)
        V = torch.where(
            mask[:, None],
            V + scale_old[:, None] * v_chunk,
            V * scale_new[:, None] + v_chunk,
        )
        M = M_new

        # fill diagonal scores for rows in this slice
        rows = torch.arange(start, end, device=device)
        a_pos[rows] = beta * (X[rows] * Yc[rows - start]).sum(dim=1)

    logZ = torch.log(S) + M  # [N]
    Ey = V / S[:, None]  # [N, d]
    return Ey, logZ, a_pos


@torch.no_grad()
def _column_weighted_sum_from_rows(
    A_cols: torch.Tensor,  # columns we accumulate onto, shape [N, d]
    B_rows: torch.Tensor,  # anchors whose rows define distributions, shape [N, d]
    beta: float,
    *,
    chunk_size: int = 4096,
    objective: ObjectiveKind = "infonce",
) -> torch.Tensor:
    """
    Compute C[j] = Σ_k w_{k,j} * B_rows[k], where
      - if objective == 'infonce': w_{k,j} = π^B_{k,j} (row-softmax over columns j of A)
      - if objective == 'loob':   w_{k,j} = π^B_{k,j} / (1 - π^B_{k,k}) for j != k, and 0 if j == k

    This is the cross-view contribution you need to build exact gradients for the symmetric loss.
    Implementation: two passes over chunks of A:
      1) get final (logZ_k, π^B_{k,k}) for each B-row using _row_softmax_expectation(B, A, ...)
      2) stream columns of A, form W_block = exp(scores - M_k) / S_k, adjust for LOOB, and do W_block^T @ B_rows

    Returns:
      C: shape [N, d]
    """
    device = A_cols.device
    N, d = A_cols.shape
    assert B_rows.shape == (N, d)

    # Pass 1: row normalizers and diag probs for B vs A
    Ey_dummy, logZ_rows, a_diag = _row_softmax_expectation(
        B_rows, A_cols, beta, chunk_size=chunk_size
    )
    pi_diag = torch.exp(a_diag - logZ_rows)  # π_{k,k}

    # Precompute per-row factors
    denom_neg = 1.0 - pi_diag
    if objective == "loob":
        denom_neg = torch.clamp(denom_neg, min=1e-12)

    C = torch.zeros((N, d), device=device, dtype=torch.float32)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        Ac = A_cols[start:end]  # [C, d]
        scores = beta * (B_rows @ Ac.t())  # [N, C]
        W = torch.exp(scores - logZ_rows[:, None])  # [N, C]

        if objective == "loob":
            rows = torch.arange(start, end, device=device)
            W[rows, rows - start] = 0.0
            W = W / denom_neg[:, None]

        C_block = W.t() @ B_rows  # [C, d]
        C[start:end] = C_block

    return C


@torch.no_grad()
def exact_gradients(
    X: torch.Tensor,
    Y: torch.Tensor,
    tau: float,
    *,
    objective: ObjectiveKind = "infonce",
    chunk_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact gradients of the symmetric per-batch loss:

      L_InfoNCE = -(β/N) Σ_i x_i^T y_i
                  + (1/2N) [ Σ_i log Σ_j e^{β x_i^T y_j} + Σ_i log Σ_j e^{β y_i^T x_j} ]

      L_LOOB    = same, but each log-sum excludes its own positive index.

    Returns:
      (gX, gY), each shape [N, d]
    """
    beta = 1.0 / float(tau)
    N, d = X.shape
    assert Y.shape == (N, d)

    # alignment gradients
    gX = -(beta / N) * Y
    gY = -(beta / N) * X

    # X-view row softmax over Y
    Ey_x, logZ_x, a_pos_x = _row_softmax_expectation(X, Y, beta, chunk_size=chunk_size)
    if objective == "loob":
        pi_diag_x = torch.exp(a_pos_x - logZ_x)
        denom_x = torch.clamp(1.0 - pi_diag_x, min=1e-12)
        Ey_x = (Ey_x - pi_diag_x[:, None] * Y) / denom_x[:, None]

    # Y-view row softmax over X, then accumulate per-column for cross term
    Cx_from_y = _column_weighted_sum_from_rows(
        A_cols=X, B_rows=Y, beta=beta, chunk_size=chunk_size, objective=objective
    )

    # Combine contributions to gX
    gX = gX + (beta / (2 * N)) * (beta * Ey_x + beta * Cx_from_y)

    # Symmetric for gY:
    Ex_y, logZ_y, a_pos_y = _row_softmax_expectation(Y, X, beta, chunk_size=chunk_size)
    if objective == "loob":
        pi_diag_y = torch.exp(a_pos_y - logZ_y)
        denom_y = torch.clamp(1.0 - pi_diag_y, min=1e-12)
        Ex_y = (Ex_y - pi_diag_y[:, None] * X) / denom_y[:, None]

    Cy_from_x = _column_weighted_sum_from_rows(
        A_cols=Y, B_rows=X, beta=beta, chunk_size=chunk_size, objective=objective
    )

    gY = gY + (beta / (2 * N)) * (beta * Ex_y + beta * Cy_from_x)
    return gX, gY
