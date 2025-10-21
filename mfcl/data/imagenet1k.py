"""ImageNet-1K dataset helpers for SSL pretraining and optional validation."""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets


class ImageListDataset(Dataset):
    """Dataset from explicit file list; labels optional/implied by subdirs.

    For SSL pretraining, transforms typically return a dict of multiple views.
    This dataset returns ``(transform(img), idx)`` for bookkeeping. No label
    inference is attempted for SSL; evaluation should provide label-aware
    datasets or transforms.
    """

    def __init__(self, root: str, file_list: str, transform: Optional[Callable[[Image.Image], object]] = None):
        """Construct dataset from a file list.

        Args:
            root: Root directory containing images (paths in list are relative or absolute).
            file_list: Path to a text file with one image path per line (relative to root or absolute).
            transform: Callable that maps a PIL image -> dict of tensors (SSL) or tensor (eval).

        Raises:
            FileNotFoundError: If ``file_list`` missing.
        """
        super().__init__()
        self.root = root
        self.transform = transform
        list_path = file_list if os.path.isabs(file_list) else os.path.join(root, file_list)
        list_path = os.path.abspath(list_path)
        if not os.path.exists(list_path):
            raise FileNotFoundError(list_path)
        with open(list_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        self.paths: List[str] = []
        for p in lines:
            if os.path.isabs(p):
                full = p
            else:
                full = os.path.join(root, p)
            self.paths.append(full)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, object], int]:  # type: ignore[override]
        """Return a mapping and the index.

        The returned mapping is typically a dict of multiple views for SSL.
        If ``transform`` returns a tensor or other type, it is wrapped under
        the key ``"input"`` to maintain a consistent mapping return type.
        """
        path = self.paths[idx]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found at index {idx}: {path}")
        with Image.open(path) as img:
            converted_img = img.convert("RGB").copy()
        if self.transform is None:
            return {"img": converted_img}, idx
        out = self.transform(converted_img)
        if isinstance(out, dict):
            return out, idx
        return {"input": out}, idx


def build_imagenet_datasets(
    root: str,
    train_list: Optional[str],
    val_list: Optional[str],
    train_transform: Callable[[Image.Image], object],
    val_transform: Optional[Callable[[Image.Image], object]] = None,
) -> Tuple[Dataset, Optional[Dataset]]:
    """Build train and optional val datasets for ImageNet-1K.

    Args:
        root: Image root.
        train_list: If provided, use ImageListDataset for train split.
        val_list: If provided, use ImageListDataset for val split; else use ImageFolder if exists.
        train_transform: Transform for the training split (SSL or eval).
        val_transform: Transform for val split; if None and val dataset is built, reuse train_transform.

    Returns:
        (train_ds, val_ds_or_None).
    """
    # Train dataset
    if train_list is not None:
        train_ds: Dataset = ImageListDataset(
            root, train_list, transform=train_transform
        )
    else:
        train_root = (
            os.path.join(root, "train")
            if os.path.isdir(os.path.join(root, "train"))
            else root
        )
        train_ds = datasets.ImageFolder(root=train_root, transform=train_transform)

    # Val dataset
    val_ds: Optional[Dataset] = None
    vt = val_transform if val_transform is not None else train_transform
    if val_list is not None:
        val_ds = ImageListDataset(root, val_list, transform=vt)
    else:
        val_root = os.path.join(root, "val")
        if os.path.isdir(val_root):
            val_ds = datasets.ImageFolder(root=val_root, transform=vt)

    return train_ds, val_ds


__all__ = ["ImageListDataset", "build_imagenet_datasets"]
