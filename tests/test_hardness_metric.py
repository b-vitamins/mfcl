import csv
from pathlib import Path

import torch

from mfcl.telemetry.hardness import HardnessMonitor, get_active_monitor


def _read_csv(path: Path):
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def test_hardness_monitor_writes_quantiles(tmp_path):
    log_path = tmp_path / "hardness.csv"
    monitor = HardnessMonitor(
        enabled=True,
        log_path=log_path,
        is_main=True,
        max_anchors=8,
        max_negatives=None,
        topk=(1, 5),
        seed=42,
    )

    neg = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.0, -0.1, 0.6, 0.2, 0.3],
            [0.9, 0.8, 0.7, 0.6, 0.5],
        ],
        dtype=torch.float32,
    )

    monitor.begin_step(epoch=2, step=7)
    assert get_active_monitor() is monitor
    monitor.add_negatives(neg)
    monitor.end_step()
    assert get_active_monitor() is None

    rows = _read_csv(log_path)
    assert len(rows) == 1
    row = rows[0]
    assert int(row["step"]) == 7
    assert int(row["epoch"]) == 2
    assert int(row["anchors_seen"]) == 4
    assert int(row["anchors_sampled"]) == 4
    assert abs(float(row["top1_p50"]) - 0.55) < 1e-6
    assert abs(float(row["top1_p90"]) - 0.81) < 1e-6
    assert abs(float(row["top5_p50"]) - 0.1) < 1e-6
    assert abs(float(row["top5_p90"]) - 0.38) < 1e-6


def test_hardness_monitor_respects_caps(tmp_path):
    monitor = HardnessMonitor(
        enabled=True,
        log_path=None,
        is_main=True,
        max_anchors=2,
        max_negatives=2,
        topk=(1,),
        seed=0,
    )
    data = torch.tensor(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6, 0.7],
            [0.8, 0.9, 1.0, 1.1],
        ],
        dtype=torch.float32,
    )
    monitor.begin_step(epoch=0, step=1)
    monitor.add_negatives(data)
    monitor.end_step()
    # Only two anchors kept due to max_anchors=2
    assert monitor._reservoir is None  # internal buffers cleared after end_step
