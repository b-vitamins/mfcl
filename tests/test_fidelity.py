from __future__ import annotations

import pytest
import torch

from mfcl.losses.ntxent import NTXentLoss
from mfcl.methods.base import BaseMethod
from mfcl.telemetry.fidelity import compare_losses


class _ToyMethod(BaseMethod):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4, bias=False)
        self.projector = torch.nn.Linear(4, 4, bias=False)

        torch.nn.init.eye_(self.encoder.weight)
        torch.nn.init.eye_(self.projector.weight)

    def forward_views(self, batch):
        view1 = batch["view1"]
        view2 = batch["view2"]
        z1 = self.projector(self.encoder(view1))
        z2 = self.projector(self.encoder(view2))
        return z1, z2

    def compute_loss(self, *proj, batch):  # pragma: no cover - unused in tests
        raise NotImplementedError


def _dummy_batch(batch_size: int = 8) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    view1 = torch.randn(batch_size, 4)
    view2 = torch.randn(batch_size, 4)
    return {"view1": view1, "view2": view2}


def test_compare_losses_identical() -> None:
    method = _ToyMethod()
    batch = _dummy_batch()

    loss_a = NTXentLoss(temperature=0.2)
    loss_b = NTXentLoss(temperature=0.2)

    metrics = compare_losses(loss_a, loss_b, method, batch, beta=1e-8)

    assert metrics["loss_diff"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["grad_cos"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["grad_rel_norm"] == pytest.approx(0.0, abs=1e-6)


def test_compare_losses_temperature_variation_monotonic() -> None:
    method = _ToyMethod()
    batch = _dummy_batch()

    base = NTXentLoss(temperature=0.2)
    temps = [0.2, 0.4, 0.8]
    results = [compare_losses(base, NTXentLoss(temperature=t), method, batch, beta=1e-8) for t in temps]

    loss_diffs = [r["loss_diff"] for r in results]
    grad_rel = [r["grad_rel_norm"] for r in results]
    grad_cos = [r["grad_cos"] for r in results]

    assert all(a <= b + 1e-6 for a, b in zip(loss_diffs, loss_diffs[1:]))
    assert all(a <= b + 1e-6 for a, b in zip(grad_rel, grad_rel[1:]))
    assert all(a >= b - 1e-6 for a, b in zip(grad_cos, grad_cos[1:]))
