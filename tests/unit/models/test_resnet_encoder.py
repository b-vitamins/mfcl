import torch
import torch.nn as nn

from mfcl.models.encoders import resnet as resnet_module
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


def test_resnet_custom_in_channels_keeps_dim():
    enc = ResNetEncoder(variant="resnet18", pretrained=False, in_channels=1)
    x = torch.randn(2, 1, 160, 160)
    y = enc(x)
    assert y.shape == (2, enc.feature_dim)


def test_resnet_multi_channel_input():
    enc = ResNetEncoder(variant="resnet18", pretrained=False, in_channels=5)
    x = torch.randn(2, 5, 160, 160)
    y = enc(x)
    assert y.shape == (2, enc.feature_dim)


def test_resnet_set_train_bn_runtime_toggle():
    enc = ResNetEncoder(variant="resnet18", pretrained=False, train_bn=True)
    x = torch.randn(2, 3, 160, 160)
    _ = enc(x)
    enc.set_train_bn(False)
    for m in enc.backbone.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            assert not m.training
            for p in m.parameters(recurse=False):
                assert p.requires_grad is False
    enc.set_train_bn(True)
    for m in enc.backbone.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            assert m.training
            for p in m.parameters(recurse=False):
                assert p.requires_grad is True


def test_resnet_two_channel_pretrained_adapt(monkeypatch):
    class DummyResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 2, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(2)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.Identity()
            self.layer1 = nn.Identity()
            self.layer2 = nn.Identity()
            self.layer3 = nn.Identity()
            self.layer4 = nn.Identity()
            self.fc = nn.Linear(1, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
            raise NotImplementedError

    dummy = DummyResNet()
    original = dummy.conv1.weight.detach().clone()

    def _fake_factory(variant: str, pretrained: bool) -> nn.Module:
        assert variant == "resnet18"
        assert pretrained is True
        return dummy

    monkeypatch.setattr(resnet_module, "_create_tv_resnet", _fake_factory)

    encoder = ResNetEncoder(variant="resnet18", pretrained=True, in_channels=2)
    adapted = encoder.backbone.conv1.weight.detach()
    expected = original.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)
    assert adapted.shape[1] == 2
    assert torch.allclose(adapted, expected)
