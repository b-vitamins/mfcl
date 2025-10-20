from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest

from tools import aggregate, sweep


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(str(item) for item in row) + "\n")


def _emit_stub_artifacts(run_dir: Path, params: dict[str, object], metadata: dict[str, object]) -> None:
    timings_header = [
        "step",
        "epoch",
        "t_data_ms",
        "t_fwd_ms",
        "t_bwd_ms",
        "t_opt_ms",
        "t_comm_ms",
        "t_assign_ms",
        "t_topr_ms",
        "t_beta_ctrl_ms",
        "t_misc_ms",
        "t_step_ms",
        "ips_step",
    ]
    timings_rows = [
        [1, 0, 1.0, 2.0, 3.0, 4.0, 1.5, 0.5, 0.2, 0.1, 0.7, 48.0, 210.0],
        [2, 0, 1.1, 2.1, 3.1, 4.1, 1.6, 0.6, 0.3, 0.2, 0.8, 52.0, 190.0],
    ]
    _write_csv(run_dir / "timings.csv", timings_header, timings_rows)

    comms_header = [
        "step",
        "epoch",
        "world_size",
        "bytes_all_reduce",
        "bytes_all_gather",
        "bytes_reduce_scatter",
        "bytes_broadcast",
        "bytes_total",
        "t_all_reduce_ms",
        "t_all_gather_ms",
        "t_reduce_scatter_ms",
        "t_broadcast_ms",
        "t_total_ms",
        "eff_bandwidth_MiBps",
        "eff_bandwidth_MBps",
        "bytes_features_allgather",
        "bytes_moments_mu",
        "bytes_moments_sigma_full",
        "bytes_moments_sigma_diag",
        "bytes_mixture_muK",
        "bytes_mixture_sigmaK",
        "bytes_third_moment_sketch",
        "bytes_topr_indices",
        "bytes_other",
    ]
    comms_rows = [
        [1, 0, params.get("world_size", 1), 1000.0, 500.0, 0.0, 0.0, 1500.0, 1.0, 0.5, 0.0, 0.0, 1.5, 1.0, 1.0, 400.0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, params.get("world_size", 1), 1100.0, 400.0, 0.0, 0.0, 1500.0, 1.1, 0.4, 0.0, 0.0, 1.5, 1.0, 1.0, 410.0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    _write_csv(run_dir / "comms.csv", comms_header, comms_rows)

    budget_payload = {
        "mode": metadata.get("budget_mode", "iso_time"),
        "limits": {
            metadata.get("budget_limit_key", "max_minutes"): metadata.get("budget_limit_value", 0.5),
        },
        "totals": {
            "tokens": 2048,
            "time_ms": 60000.0,
            "time_minutes": 1.0,
            "comm_bytes": 3000.0,
            "energy_Wh": 1.25,
            "steps": 2,
            "epochs": 0.1,
        },
        "steps_per_epoch": 20.0,
    }
    (run_dir / "budget.json").write_text(json.dumps(budget_payload, indent=2), encoding="utf-8")

    metric_name = str(metadata.get("target_metric", "linear.top1"))
    eval_header = ["step", metric_name, "time_minutes"]
    eval_rows = [
        [1, 65.0, 0.5],
        [2, 78.0, 0.8],
    ]
    _write_csv(run_dir / "linear_eval.csv", eval_header, eval_rows)

    energy_header = ["gpu_id", "energy_J_cum"]
    energy_rows = [[0, 100.0], [0, 450.0], [1, 90.0], [1, 360.0]]
    _write_csv(run_dir / "energy.csv", energy_header, energy_rows)

    fidelity_header = ["step", "epoch", "grad_cos"]
    fidelity_rows = [[1, 0, 0.9], [2, 0, 0.92]]
    _write_csv(run_dir / "fidelity.csv", fidelity_header, fidelity_rows)


def test_sweep_and_aggregate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    grid_payload = {
        "name": "spec9_smoke",
        "entrypoint": "train.py",
        "output_dir": str(tmp_path / "sweep_output"),
        "grid": [
            {
                "name": "combo_a",
                "params": {
                    "model": "resnet18",
                    "method": "simclr",
                    "batch_size": 128,
                    "amp_dtype": "bf16",
                    "world_size": 1,
                },
                "budget": {"mode": "iso_time", "limit": 1.0, "target_metric": "linear.top1", "target_value": 75.0},
            },
            {
                "name": "combo_b",
                "params": {
                    "model": "resnet50",
                    "method": "simclr",
                    "batch_size": 256,
                    "amp_dtype": "fp32",
                    "world_size": 1,
                },
                "budget": {"mode": "iso_time", "limit": 1.0, "target_metric": "linear.top1", "target_value": 75.0},
            },
        ],
    }
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(json.dumps(grid_payload), encoding="utf-8")

    def _fake_run(cmd, cwd, env, stdout, stderr, text, check):
        run_dir = Path(env["MFCL_SWEEP_RUN_DIR"])
        params = json.loads(env.get("MFCL_SWEEP_PARAMS", "{}"))
        metadata = json.loads(env.get("MFCL_SWEEP_METADATA", "{}"))
        _emit_stub_artifacts(run_dir, params, metadata)
        assert "MFCL_AMP_DTYPE" in env
        stdout.write("stub run\n")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    result = sweep.main(["--enable-sweeps", str(grid_path)])
    assert result == 0

    manifest_path = Path(grid_payload["output_dir"]) / "sweep_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["runs"]) == 2
    assert all(run["status"] == "completed" for run in manifest["runs"])
    first_run, second_run = manifest["runs"]
    assert "model=resnet18" in first_run["overrides"]
    assert "train.loss_fp32=false" in first_run["overrides"]
    assert first_run["env"].get("MFCL_AMP_DTYPE") == "bf16"
    assert "model=resnet50" in second_run["overrides"]
    assert "train.loss_fp32=true" in second_run["overrides"]
    assert second_run["env"].get("MFCL_AMP_DTYPE") == "fp32"

    agg_result = aggregate.main([
        "--enable-aggregator",
        "--root",
        str(grid_payload["output_dir"]),
    ])
    assert agg_result == 0

    reports_dir = Path(grid_payload["output_dir"]) / "reports"
    summary_md = reports_dir / "summary.md"
    assert summary_md.exists()
    content = summary_md.read_text(encoding="utf-8")
    assert "param_model" in content
    assert "ips_mean" in content

    summary_csv_candidates = sorted(reports_dir.glob("summary_*.csv"))
    assert summary_csv_candidates, "Expected summary CSV to be generated"
    summary_path = summary_csv_candidates[-1]
    rows = list(csv.DictReader(summary_path.read_text(encoding="utf-8").splitlines()))
    assert rows, "Summary CSV should contain data"
    assert "accuracy_value" in rows[0], "accuracy column missing"
    assert pytest.approx(float(rows[0]["energy_Wh"])) == pytest.approx((450.0 + 360.0) / 3600.0)
