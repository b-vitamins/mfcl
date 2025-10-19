import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mfcl.utils.provenance import (
    append_event,
    collect_provenance,
    write_provenance,
    write_stable_manifest_once,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_roundtrip_json(tmp_path):
    cfg = {"train": {"seed": 123}, "data": {"root": str(tmp_path)}}
    snapshot = collect_provenance(cfg)
    payload = {"history": [snapshot]}
    out_path = tmp_path / "provenance" / "repro.json"
    write_provenance(out_path, payload)
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded == payload


def test_manifest_and_events(tmp_path):
    prov_dir = tmp_path / "provenance"
    snapshot = collect_provenance({})
    write_stable_manifest_once(prov_dir, snapshot)
    write_stable_manifest_once(prov_dir, {"different": True})
    repro_path = prov_dir / "repro.json"
    stored = json.loads(repro_path.read_text(encoding="utf-8"))
    assert stored == snapshot

    append_event(prov_dir, {"type": "start", "program": "train"})
    append_event(prov_dir, {"type": "start", "program": "train"})
    events_path = prov_dir / "events.jsonl"
    lines = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 2
    first_event = json.loads(lines[0])
    assert first_event["type"] == "start"
    assert first_event["program"] == "train"
    assert "time" in first_event


def test_git_info_present(tmp_path):
    cfg = {"train": {"seed": 0}, "data": {"root": str(tmp_path)}}
    snapshot = collect_provenance(cfg)
    git_info = snapshot.get("git", {})
    if not git_info.get("available"):
        pytest.skip("git unavailable in environment")
    assert git_info.get("sha")
    prov_path = tmp_path / "prov" / "repro.json"
    write_provenance(prov_path, {"history": [snapshot]})
    diff_path = prov_path.parent / "git.diff"
    assert diff_path.exists()

    repo_root = _repo_root()
    target = repo_root / "README.md"
    original = target.read_text(encoding="utf-8")
    try:
        target.write_text(original + "\n# provenance-test\n", encoding="utf-8")
        dirty_snapshot = collect_provenance(cfg)
        dirty_info = dirty_snapshot.get("git", {})
        if not dirty_info.get("available"):
            pytest.skip("git unavailable when collecting dirty snapshot")
        assert dirty_info.get("dirty") is True
        write_provenance(prov_path, {"history": [dirty_snapshot]})
        dirty_diff = diff_path.read_text(encoding="utf-8")
        assert dirty_diff.strip()
    finally:
        target.write_text(original, encoding="utf-8")


def test_seed_recorded(tmp_path):
    cfg = {"train": {"seed": 42}, "data": {"root": str(tmp_path)}}
    snapshot = collect_provenance(cfg)
    seeds = snapshot.get("seeds", {})
    for key in ("python", "numpy", "torch_cpu", "torch_cuda", "dataloader_workers"):
        assert key in seeds
    assert isinstance(seeds.get("dataloader_workers"), dict)


@pytest.mark.slow
def test_provenance_repro_hash(tmp_path):
    try:
        status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=_repo_root())
    except FileNotFoundError:
        pytest.skip("git not available for reproducibility check")
    if status.strip():
        pytest.skip("repository must be clean for reproducibility check")
    try:
        import torchvision  # noqa: F401
    except Exception:
        pytest.skip("torchvision is required for the provenance integration test")

    run_dir = tmp_path / "repro_run"
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    base_cmd = [
        sys.executable,
        "train.py",
        "data.name=synthetic",
        "data.synthetic_train_size=80",
        "data.synthetic_val_size=80",
        "data.batch_size=8",
        "train.epochs=1",
        "train.warmup_epochs=0",
        "train.log_interval=5",
        "train.amp=false",
        "optim.lr=0.01",
        "runtime.provenance=true",
        f"hydra.run.dir={run_dir}",
    ]

    subprocess.run(base_cmd, check=True, cwd=_repo_root(), env=env)
    repro_path = run_dir / "provenance" / "repro.json"
    assert repro_path.exists()
    first_bytes = repro_path.read_bytes()
    first_hash = hashlib.sha256(first_bytes).hexdigest()

    shutil.rmtree(run_dir)

    subprocess.run(base_cmd, check=True, cwd=_repo_root(), env=env)
    repro_path_second = run_dir / "provenance" / "repro.json"
    second_bytes = repro_path_second.read_bytes()
    second_hash = hashlib.sha256(second_bytes).hexdigest()

    assert second_hash == first_hash
