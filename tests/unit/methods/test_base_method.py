import pytest
import torch
from mfcl.methods.base import BaseMethod


class DummyMethod(BaseMethod):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward_views(self, batch):
        return (batch["x"],)

    def compute_loss(self, x, batch):
        loss = x.mean()
        return {"loss": loss}


def test_base_method_step():
    m = DummyMethod()
    # ensure matching devices for normal execution
    m.weight = torch.nn.Parameter(torch.ones_like(m.weight))
    out = m.step({"x": torch.ones(2, 3)})
    assert "loss" in out and torch.isfinite(out["loss"])


def test_base_method_step_device_mismatch():
    m = DummyMethod()
    batch = {"x": torch.ones(2, 3, device="meta")}
    with pytest.raises(RuntimeError) as exc:
        m.step(batch)
    assert "cpu" in str(exc.value)
    assert "meta" in str(exc.value)
