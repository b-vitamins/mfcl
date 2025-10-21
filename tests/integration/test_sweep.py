from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools import sweep


@pytest.mark.parametrize("node0_endpoint,node1_endpoint", [("node0.example:29500", "node1.example:29500")])
def test_multinode_grid_parsing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, node0_endpoint: str, node1_endpoint: str) -> None:
    output_dir = tmp_path / "multi_node"
    grid_payload = {
        "name": "multi_node",
        "entrypoint": "train.py",
        "output_dir": str(output_dir),
        "nnodes": 2,
        "rendezvous": {"backend": "c10d", "endpoint": node0_endpoint, "id": "mfcl-grid"},
        "grid": [
            {
                "name": "node0",
                "node_rank": 0,
                "params": {"world_size": 2},
            },
            {
                "name": "node1",
                "node_rank": 1,
                "params": {"world_size": 2},
                "rendezvous": {"endpoint": node1_endpoint},
            },
        ],
    }

    grid_path = tmp_path / "grid.json"
    grid_path.write_text(json.dumps(grid_payload), encoding="utf-8")

    captured: list[dict[str, object]] = []

    def _fake_run(cmd, cwd, env, stdout, stderr, text, check):
        captured.append(
            {
                "cmd": list(cmd),
                "env": {key: env.get(key) for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "NNODES", "NODE_RANK")},
            }
        )
        stdout.write("ok\n")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    result = sweep.main(["--enable-sweeps", str(grid_path)])
    assert result == 0

    assert len(captured) == 2

    first = captured[0]
    first_cmd = first["cmd"]
    assert "--nnodes=2" in first_cmd
    assert "--nproc_per_node=2" in first_cmd
    assert "--node_rank=0" in first_cmd
    assert "--rdzv_backend=c10d" in first_cmd
    assert f"--rdzv_endpoint={node0_endpoint}" in first_cmd
    assert "--rdzv_id=mfcl-grid" in first_cmd
    assert "--standalone" not in first_cmd
    first_env = first["env"]
    assert first_env == {"WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "2", "NNODES": "2", "NODE_RANK": "0"}

    second = captured[1]
    second_cmd = second["cmd"]
    assert f"--rdzv_endpoint={node1_endpoint}" in second_cmd
    assert "--node_rank=1" in second_cmd
    second_env = second["env"]
    assert second_env == {"WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "2", "NNODES": "2", "NODE_RANK": "1"}
