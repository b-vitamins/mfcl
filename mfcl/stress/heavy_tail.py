"""Heavy-tail injection utilities for stress testing similarity metrics."""

from __future__ import annotations

import math
from typing import Optional

import torch


def inject_heavy_tails(
    values: torch.Tensor,
    p_tail: float,
    *,
    tail_scale: float = 5.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Replace a fraction of entries with samples from a heavy-tailed mixture."""

    if not torch.is_tensor(values):
        raise TypeError("values must be a torch.Tensor")
    if not 0.0 <= p_tail <= 1.0:
        raise ValueError("p_tail must be in [0, 1]")
    if values.numel() == 0 or p_tail == 0:
        return values.clone()
    if tail_scale <= 0:
        raise ValueError("tail_scale must be > 0")

    flat = values.reshape(-1)
    result = flat.clone()
    num_tail = int(math.floor(flat.numel() * p_tail))
    if num_tail == 0:
        return result.view_as(values)

    device = result.device
    gen = generator
    if gen is not None and hasattr(gen, "device") and gen.device != device:
        raise ValueError("generator device must match tensor device")

    if gen is None:
        perm = torch.randperm(result.numel())
    else:
        perm = torch.randperm(result.numel(), generator=gen)
    perm = perm.to(device=device)
    tail_indices = perm[:num_tail]
    heavy_samples = _sample_heavy_mix(num_tail, dtype=result.dtype, device=device, tail_scale=tail_scale, generator=gen)
    result[tail_indices] = heavy_samples
    return result.view_as(values)


def _sample_heavy_mix(
    count: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    tail_scale: float,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    base = torch.randn(count, dtype=dtype, device=device, generator=generator)
    tail = torch.randn(count, dtype=dtype, device=device, generator=generator) * tail_scale
    mask = torch.rand(count, device=device, generator=generator) < 0.5
    samples = base.clone()
    samples[mask] = tail[mask]
    return samples


__all__ = ["inject_heavy_tails"]
