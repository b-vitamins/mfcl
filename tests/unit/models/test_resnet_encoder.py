import torch

from mfcl.models.encoders.resnet import ResNetEncoder


def test_resnet18_avg_pool_shape_and_flags():
    enc = ResNetEncoder(
        variant="resnet18",
        pretrained=False,
        train_bn=False,
        norm_feat=True,
        pool="avg",
        freeze_backbone=True,
    )
    x = torch.randn(2, 3, 160, 160)
    y = enc(x)
    assert y.shape == (2, enc.feature_dim)
    # BN frozen
    for m in enc.backbone.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            assert not m.training
            for p in m.parameters():
                assert p.requires_grad is False
    # All params frozen
    for p in enc.parameters():
        assert p.requires_grad is False


def test_resnet34_identity_pool_spatial_dims():
    enc = ResNetEncoder(variant="resnet34", pretrained=False, pool="identity")
    x = torch.randn(2, 3, 160, 160)
    y = enc(x)
    assert (
        y.ndim == 4
        and y.shape[1] == enc.feature_dim
        and y.shape[-1] > 1
        and y.shape[-2] > 1
    )
