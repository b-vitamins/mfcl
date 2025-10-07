from __future__ import annotations

import torch


@torch.no_grad()
def cosine_topk_streaming(
    X: torch.Tensor,
    Y: torch.Tensor,
    k: int,
    *,
    chunk_size: int = 4096,
    exclude_self: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Streaming cosine/dot-product top-k between X and Y without materializing N×N.

    Inputs:
      X: [N, d], Y: [N, d] on same device
      k: number of neighbors to return (will clamp to N - 1 if exclude_self)
      exclude_self: if True, mask the diagonal j == i

    Returns:
      (vals, idx): both [N, k_eff], with largest first. If k<=0, returns empty tensors.

    Notes:
      - If you just need values and max, prefer `mfcl.approx.hybrid.topk_neg_sim_streaming`.
    """
    device = X.device
    N, d = X.shape
    assert Y.shape == (N, d)
    k_eff = max(0, min(k, N - 1 if exclude_self else N))

    if k_eff == 0:
        return torch.empty((N, 0), device=device, dtype=torch.float32), torch.empty(
            (N, 0), device=device, dtype=torch.long
        )

    # Running top-k buffers
    top_vals = torch.full((N, k_eff), float("-inf"), device=device, dtype=torch.float32)
    top_idx = torch.full((N, k_eff), -1, device=device, dtype=torch.long)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        Yc = Y[start:end]  # [C, d]
        scores = X @ Yc.t()  # [N, C]
        if exclude_self:
            rows = torch.arange(start, end, device=device)
            scores[rows, rows - start] = float("-inf")

        # Merge existing top-k with new block
        # Build candidate values and their global indices
        cand_vals = torch.cat([top_vals, scores], dim=1)  # [N, k_eff + C]
        # Build candidate indices matrix to track provenance
        # For the first k_eff columns, use existing top_idx; for new block, use global j indices
        existing_idx = top_idx
        new_idx = torch.arange(start, end, device=device).view(1, -1).expand(N, -1)
        cand_idx = torch.cat([existing_idx, new_idx], dim=1)  # [N, k_eff + C]

        # Take top-k again
        vals, idx = torch.topk(cand_vals, k=k_eff, dim=1, largest=True, sorted=True)
        # Fancy index to pick corresponding indices
        ar = torch.arange(N, device=device).view(-1, 1)
        top_vals = vals
        top_idx = cand_idx[ar, idx]

    return top_vals, top_idx
