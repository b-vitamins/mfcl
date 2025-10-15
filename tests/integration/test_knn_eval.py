import torch
from torch.utils.data import Dataset, DataLoader

from mfcl.engines.evaluator import KNNEval


class VecDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], int(self.y[idx].item())


class IdentityEncoder(torch.nn.Module):
    def forward(self, x):
        return x


def test_knn_eval_identical_bank_query():
    X = torch.randn(50, 8)
    y = (torch.arange(50) % 5).long()
    ds = VecDataset(X, y)
    loader = DataLoader(ds, batch_size=10)
    enc = IdentityEncoder()
    knn = KNNEval(k=1, temperature=0.07, normalize=False)
    bank_feats, bank_labels = knn.build_bank(enc, loader, device=torch.device("cpu"))
    metrics = knn.evaluate(
        enc, loader, bank_feats, bank_labels, device=torch.device("cpu")
    )
    assert metrics["top1"] == 1.0
