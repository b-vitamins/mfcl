import torch
from mfcl.methods.base import BaseMethod


class DummyMethod(BaseMethod):
    def forward_views(self, batch):
        return (batch["x"],)

    def compute_loss(self, x, batch):
        loss = x.mean()
        return {"loss": loss}


def test_base_method_step():
    m = DummyMethod()
    out = m.step({"x": torch.ones(2, 3)})
    assert "loss" in out and torch.isfinite(out["loss"])
