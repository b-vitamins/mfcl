from pathlib import Path

import os
import time

from mfcl.utils.checkpoint import (
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


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


def test_latest_checkpoint_prefers_newest_by_mtime(tmp_path: Path):
    d = tmp_path / "ckpts"
    d.mkdir()
    p9 = d / "ckpt_ep9.pt"
    p10 = d / "ckpt_ep10.pt"
    save_checkpoint(str(p9), {"epoch": 9}, keep_k=5, make_latest=False)
    # Ensure p9 has an older timestamp than p10 regardless of filesystem resolution.
    older_time = p9.stat().st_mtime - 5
    os.utime(p9, (older_time, older_time))
    time.sleep(0.01)
    save_checkpoint(str(p10), {"epoch": 10}, keep_k=5, make_latest=False)
    newest = latest_checkpoint(str(d))
    assert newest is not None
    assert os.path.samefile(newest, p10)
