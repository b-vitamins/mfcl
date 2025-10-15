from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image


def make_synthetic_imagefolder(
    root: Path, num_per_class: int = 3, size: int = 32
) -> Tuple[Path, Path]:
    """Create a tiny ImageFolder structure with two classes and PNGs.

    Returns (train_dir, val_dir).
    """
    train = root / "train"
    val = root / "val"
    for base in (train, val):
        for cname in ("class0", "class1"):
            (base / cname).mkdir(parents=True, exist_ok=True)
            for i in range(num_per_class):
                img = Image.new(
                    "RGB", (size, size), color=(i * 10 % 255, 128, 255 - i * 10 % 255)
                )
                img.save(base / cname / f"{cname}_{i}.png")
    return train, val


def make_linearly_separable_features(n: int = 64, dim: int = 16):
    """Return (features, labels) that are linearly separable across two classes."""
    import torch

    half = n // 2
    a = torch.randn(half, dim) + 1.0
    b = torch.randn(n - half, dim) - 1.0
    feats = torch.cat([a, b], dim=0)
    labels = torch.cat(
        [torch.zeros(half, dtype=torch.long), torch.ones(n - half, dtype=torch.long)],
        dim=0,
    )
    return feats, labels
