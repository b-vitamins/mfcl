from pathlib import Path

from mfcl.utils.checkpoint import save_checkpoint, load_checkpoint


def test_save_load_and_keepk(tmp_path: Path):
    d = tmp_path / "ckpts"
    d.mkdir()
    for e in range(1, 6):
        path = d / f"ckpt_ep{e:04d}.pt"
        save_checkpoint(str(path), {"epoch": e}, keep_k=3, make_latest=True)
    # Only last 3 should remain
    files = sorted(p.name for p in d.glob("ckpt_ep*.pt"))
    assert files == ["ckpt_ep0003.pt", "ckpt_ep0004.pt", "ckpt_ep0005.pt"]
    # latest exists
    assert (d / "latest.pt").exists()
    state = load_checkpoint(str(d / "ckpt_ep0004.pt"), strict=True)
    assert state["epoch"] == 4


def test_load_missing_strict_false(tmp_path: Path):
    p = tmp_path / "missing.pt"
    state = load_checkpoint(str(p), strict=False)
    assert state == {}
