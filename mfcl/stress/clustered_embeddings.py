"""Synthetic clustered embedding generator for diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def generate_clustered_embeddings(
    num_embeddings: int,
    dim: int,
    *,
    k: int,
    scale: float,
    seed: int = 0,
    output_dir: str | Path | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate unit-norm embeddings drawn from a spherical Gaussian mixture."""

    if num_embeddings <= 0:
        raise ValueError("num_embeddings must be > 0")
    if dim <= 0:
        raise ValueError("dim must be > 0")
    if k <= 0:
        raise ValueError("k must be > 0")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    centroids = torch.randn(k, dim, generator=generator, dtype=torch.float32)
    centroids = F.normalize(centroids, dim=1)

    assignments = torch.randint(0, k, (num_embeddings,), generator=generator)
    if k <= num_embeddings:
        base = torch.arange(k, dtype=torch.long)
        assignments[:k] = base
        perm = torch.randperm(num_embeddings, generator=generator)
        assignments = assignments[perm]
    embeddings = torch.empty(num_embeddings, dim, dtype=torch.float32)

    for cluster in range(k):
        mask = assignments == cluster
        count = int(mask.sum().item())
        if count == 0:
            continue
        noise = torch.randn(count, dim, generator=generator, dtype=torch.float32) * scale
        samples = centroids[cluster].unsqueeze(0) + noise
        samples = F.normalize(samples, dim=1)
        embeddings[mask] = samples

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, out_path / "embeddings.pt")
        params: Dict[str, object] = {
            "k": int(k),
            "dim": int(dim),
            "scale": float(scale),
            "seed": int(seed),
            "num_embeddings": int(num_embeddings),
            "centroids": centroids.tolist(),
            "assignments": assignments.tolist(),
        }
        (out_path / "params.json").write_text(json.dumps(params, indent=2, sort_keys=True))

    return embeddings, assignments, centroids


__all__ = ["generate_clustered_embeddings"]
