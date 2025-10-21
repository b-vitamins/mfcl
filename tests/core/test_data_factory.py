import pytest
import torch
from torch.utils.data import Dataset

from mfcl.core.config import (
    AugConfig,
    ClassPackedConfig,
    Config,
    DataConfig,
    MethodConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
)
from mfcl.core.factory import build_dataset, build_loader, build_sampler
from mfcl.data.sampler import DistSamplerWrapper
from mfcl.data.samplers import ClassPackedSampler


class TargetsDataset(Dataset):
    def __init__(self, targets: list[int]):
        super().__init__()
        self.targets = list(targets)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.targets)

    def __getitem__(self, index: int):  # type: ignore[override]
        return torch.zeros(1), self.targets[index]


def _make_cfg(data_cfg: DataConfig) -> Config:
    return Config(
        data=data_cfg,
        aug=AugConfig(),
        model=ModelConfig(),
        method=MethodConfig(),
        optim=OptimConfig(),
        train=TrainConfig(epochs=1, warmup_epochs=0, amp=False, save_dir=""),
    )


def test_build_sampler_class_packed_returns_sampler():
    dataset = TargetsDataset([0, 0, 1, 1, 2, 2, 3, 3])
    data_cfg = DataConfig(
        root="/tmp",
        name="imagenet",
        batch_size=4,
        shuffle=True,
        drop_last=True,
        class_packed=ClassPackedConfig(
            enabled=True,
            num_classes_per_batch=2,
            instances_per_class=2,
            seed=7,
        ),
    )
    cfg = _make_cfg(data_cfg)

    sampler = build_sampler(cfg, dataset, world_size=1, rank=0)

    assert isinstance(sampler, ClassPackedSampler)
    assert sampler.batch_size == 4


def test_build_sampler_distributed_returns_dist_wrapper():
    dataset = TargetsDataset([0] * 16)
    data_cfg = DataConfig(
        root="/tmp",
        name="imagenet",
        batch_size=8,
        shuffle=False,
        drop_last=False,
    )
    cfg = _make_cfg(data_cfg)

    sampler = build_sampler(cfg, dataset, world_size=2, rank=1)

    assert isinstance(sampler, DistSamplerWrapper)
    assert sampler.drop_last is False


def test_build_dataset_and_loader_synthetic(tmp_path):
    pytest.importorskip("torchvision.datasets")

    data_cfg = DataConfig(
        root=str(tmp_path),
        name="synthetic",
        batch_size=8,
        num_workers=0,
        shuffle=True,
        drop_last=True,
    )
    cfg = _make_cfg(data_cfg)

    train_transform = lambda img: {"view0": torch.zeros(1)}
    eval_transform = lambda img: torch.ones(1)

    train_ds, val_ds = build_dataset(cfg, train_transform, eval_transform)
    assert len(train_ds) == cfg.data.synthetic_train_size
    assert val_ds is not None

    sampler = build_sampler(cfg, train_ds, world_size=1, rank=0)
    assert sampler is None

    loader = build_loader(
        cfg,
        train_ds,
        sampler=sampler,
        collate_fn=lambda batch: batch,
    )
    batch = next(iter(loader))
    assert len(batch) == cfg.data.batch_size
