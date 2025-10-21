from collections.abc import Sized
from pathlib import Path
from typing import cast

import pytest


from mfcl.data.imagenet1k import ImageListDataset, build_imagenet_datasets
from tests.helpers.data import make_synthetic_imagefolder


def test_imagelist_resolves_paths(tmp_path: Path):
    train, _ = make_synthetic_imagefolder(tmp_path)
    # Build a relative file list
    lst = tmp_path / "train_list.txt"
    rels = sorted([p.relative_to(tmp_path) for p in train.rglob("*.png")])
    lst.write_text("\n".join(str(p) for p in rels))
    ds = ImageListDataset(str(tmp_path), "train_list.txt")
    assert len(ds) == len(rels)
    sample, idx = ds[0]
    assert "img" in sample


def test_imagelist_missing_file_message(tmp_path: Path):
    lst = tmp_path / "train_list.txt"
    lst.write_text("missing.png")
    ds = ImageListDataset(str(tmp_path), str(lst))
    with pytest.raises(FileNotFoundError) as exc:
        ds[0]
    assert "index 0" in str(exc.value)
    assert "missing.png" in str(exc.value)


def test_imagelist_repeated_sampling_closes_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train, _ = make_synthetic_imagefolder(tmp_path)
    lst = tmp_path / "train_list.txt"
    rels = sorted([p.relative_to(tmp_path) for p in train.rglob("*.png")])
    lst.write_text("\n".join(str(p) for p in rels))

    opened = []

    class DummyImage:
        def __init__(self, path: str):
            self.path = path
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.closed = True
            return False

        def convert(self, mode: str):
            self.mode = mode
            return self

        def copy(self):
            return {"path": self.path, "mode": getattr(self, "mode", None)}

    def fake_open(path: str):
        img = DummyImage(path)
        opened.append(img)
        return img

    monkeypatch.setattr("mfcl.data.imagenet1k.Image.open", fake_open)

    ds = ImageListDataset(str(tmp_path), "train_list.txt")
    expected_path = str(tmp_path / rels[0])

    for _ in range(5):
        sample, _ = ds[0]
        assert sample["img"]["path"] == expected_path

    assert len(opened) == 5
    assert all(img.closed for img in opened)


def test_build_imagenet_datasets_imagefolder(tmp_path: Path):
    make_synthetic_imagefolder(tmp_path)

    def tf(img):
        return {"view1": img, "view2": img}

    train_ds, val_ds = build_imagenet_datasets(str(tmp_path), None, None, tf, tf)
    sized_train = cast(Sized, train_ds)
    assert len(sized_train) > 0
    x, idx = train_ds[0]
    assert set(x.keys()) == {"view1", "view2"}
