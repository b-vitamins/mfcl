import torch

from mfcl.models.encoders.timmwrap import TimmEncoder


def test_timm_encoder_dims_and_freeze():
    enc = TimmEncoder(
        model_name="resnet18",
        pretrained=False,
        train_bn=False,
        norm_feat=True,
        global_pool="avg",
        freeze_backbone=True,
    )
    x = torch.randn(2, 3, 160, 160)
    y = enc(x)
    assert y.ndim == 2 and y.shape[0] == 2 and y.shape[1] == enc.feature_dim
    for p in enc.parameters():
        assert p.requires_grad is False
