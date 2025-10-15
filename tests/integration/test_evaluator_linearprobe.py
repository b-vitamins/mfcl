import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

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


def test_linearprobe_batch_size_override_changes_batch():
    feats = torch.randn(8, 4)
    labels = torch.randint(0, 2, (8,), dtype=torch.long)
    ds = FeatureDataset(feats, labels)

    train_batches: list[int] = []
    val_batches: list[int] = []

    def train_collate(batch):
        train_batches.append(len(batch))
        return default_collate(batch)

    def val_collate(batch):
        val_batches.append(len(batch))
        return default_collate(batch)

    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=train_collate)
    val = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=val_collate)
    enc = IdentityEncoder()
    probe = LinearProbe(
        enc,
        feature_dim=4,
        num_classes=2,
        device=torch.device("cpu"),
        epochs=1,
        lr=0.5,
        milestones=(1, 2),
        batch_size_override=2,
    )
    probe.fit(loader, val)

    assert train_batches and all(size == 2 for size in train_batches)
    assert val_batches and all(size == 2 for size in val_batches)
