import torch
from torch.utils.data import DataLoader, Dataset

from train import KNNHook


class TensorDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.features.size(0)

    def __getitem__(self, idx: int):
        return self.features[idx], int(self.labels[idx].item())


class IdentityEncoder(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        return x


def test_knn_hook_excludes_self_from_bank():
    features = torch.eye(3)
    labels = torch.arange(3)
    loader = DataLoader(TensorDataset(features, labels), batch_size=1, shuffle=False)

    encoder = IdentityEncoder()
    hook = KNNHook(
        lambda: encoder,
        loader,
        k=1,
        temperature=0.07,
        every_n_epochs=1,
        bank_device="cpu",
    )
    hook._device = torch.device("cpu")

    metrics = {"epoch": 1}
    hook.on_eval_end(metrics)

    assert metrics["knn_top1"] == 0.0
