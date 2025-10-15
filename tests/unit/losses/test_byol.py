import torch

from mfcl.losses.byolloss import BYOLLoss


def test_byol_cosine_and_detach():
    loss_fn = BYOLLoss(normalize=True, variant="cosine")
    p1 = torch.randn(4, 8, requires_grad=True)
    z2 = p1.clone().detach()
    p2 = torch.randn(4, 8, requires_grad=True)
    z1 = p2.clone().detach()
    loss, stats = loss_fn(p1, z2, p2, z1)
    loss.backward()
    assert torch.isfinite(loss)
    assert stats["cos_sim"].ndim == 0
    assert z1.grad is None and z2.grad is None
