import torch
from torch.utils.data import Dataset, DataLoader

from mfcl.engines.evaluator import LinearProbe


class FeatureDataset(Dataset):
    def __init__(self, feats: torch.Tensor, labels: torch.Tensor):
        self.X = feats
        self.y = labels

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], int(self.y[idx].item())


class IdentityEncoder(torch.nn.Module):
    def forward(self, x):
        return x


def test_linearprobe_on_separable_features():
    # Linearly separable
    n, d = 64, 16
    half = n // 2
    a = torch.randn(half, d) + 2.0
    b = torch.randn(n - half, d) - 2.0
    feats = torch.cat([a, b], dim=0)
    labels = torch.cat(
        [torch.zeros(half, dtype=torch.long), torch.ones(n - half, dtype=torch.long)],
        dim=0,
    )
    ds = FeatureDataset(feats, labels)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    val = DataLoader(ds, batch_size=32, shuffle=False)
    enc = IdentityEncoder()
    probe = LinearProbe(
        enc,
        feature_dim=d,
        num_classes=2,
        device=torch.device("cpu"),
        epochs=10,
        lr=1.0,
        milestones=(7, 9),
    )
    metrics = probe.fit(loader, val)
    assert metrics["top1"] >= 0.9
