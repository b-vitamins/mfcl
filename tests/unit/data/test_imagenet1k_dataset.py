from pathlib import Path


from mfcl.data.imagenet1k import ImageListDataset, build_imagenet_datasets
from tests.helpers.data import make_synthetic_imagefolder


def test_imagelist_resolves_paths(tmp_path: Path):
    train, _ = make_synthetic_imagefolder(tmp_path)
    # Build a relative file list
    lst = tmp_path / "train_list.txt"
    rels = sorted([p.relative_to(tmp_path) for p in train.rglob("*.png")])
    lst.write_text("\n".join(str(p) for p in rels))
    ds = ImageListDataset(str(tmp_path), str(lst))
    assert len(ds) == len(rels)
    sample, idx = ds[0]
    assert isinstance(sample, (dict, object))


def test_build_imagenet_datasets_imagefolder(tmp_path: Path):
    make_synthetic_imagefolder(tmp_path)

    def tf(img):
        return {"view1": img, "view2": img}

    train_ds, val_ds = build_imagenet_datasets(str(tmp_path), None, None, tf, tf)
    assert len(train_ds) > 0
    x, idx = train_ds[0]
    assert set(x.keys()) == {"view1", "view2"}
