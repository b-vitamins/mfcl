import pytest
import torch.nn as nn

from mfcl.utils.ema import MomentumUpdater


class TinyNet(nn.Module):
    def __init__(self, out_channels: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)


class DifferentTinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)


def test_momentum_updater_validates_momentum_range():
    net = TinyNet()
    with pytest.raises(ValueError):
        MomentumUpdater(net, net, momentum=-0.1)
    with pytest.raises(ValueError):
        MomentumUpdater(net, net, momentum=1.0)


def test_momentum_updater_detects_structure_mismatch():
    online = TinyNet()
    target = DifferentTinyNet()
    with pytest.raises(ValueError):
        MomentumUpdater(online, target, momentum=0.5)
