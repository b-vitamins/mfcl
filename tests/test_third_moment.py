"""Unit tests for the third-moment sketch diagnostics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mfcl.moments.third import ThirdMomentSketch


def test_third_moment_symmetric_distribution_is_near_zero() -> None:
    torch.manual_seed(0)
    dim = 6
    batch = 4096
    data = torch.randn(batch, dim)
    mu = torch.zeros(dim)

    sketch = ThirdMomentSketch(rank=32, seed=7, ema_decay=0.0, enabled=True)
    sketch.set_mean(mu)
    clipped = sketch.update(data)

    assert clipped.shape[0] == batch
    anchors = F.normalize(torch.randn(8, dim), dim=1)
    estimates = sketch.estimate(anchors)
    assert torch.max(estimates.abs()).item() < 0.25


def test_third_moment_detects_skew_and_scale() -> None:
    torch.manual_seed(123)
    dim = 3
    samples = 8192
    exp = torch.distributions.Exponential(rate=torch.tensor(1.0))

    base = exp.sample((samples,)) - 1.0
    skewed = torch.zeros(samples, dim)
    skewed[:, 0] = base

    amplified = torch.zeros_like(skewed)
    amplified[:, 0] = 2.0 * base

    mu = torch.zeros(dim)

    sketch_a = ThirdMomentSketch(rank=32, seed=21, ema_decay=0.0, enabled=True)
    sketch_a.set_mean(mu)
    sketch_a.update(skewed)

    sketch_b = ThirdMomentSketch(rank=32, seed=21, ema_decay=0.0, enabled=True)
    sketch_b.set_mean(mu)
    sketch_b.update(amplified)

    direction = torch.tensor([[1.0, 0.0, 0.0]])
    kappa_a = sketch_a.estimate(direction)[0].item()
    kappa_b = sketch_b.estimate(direction)[0].item()

    assert kappa_a > 0
    assert kappa_b > kappa_a
