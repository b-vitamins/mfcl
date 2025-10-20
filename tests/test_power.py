"""Tests for the NVML power monitor."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from mfcl.telemetry.power import PowerMonitor


class _FakeNvmlModule:
    handles = [0, 1]
    power_usage_mw = {0: 100_000, 1: 120_000}
    initialized = False

    @classmethod
    def nvmlInit(cls) -> None:  # pragma: no cover - trivial
        cls.initialized = True

    @classmethod
    def nvmlShutdown(cls) -> None:  # pragma: no cover - trivial
        cls.initialized = False

    @classmethod
    def nvmlDeviceGetHandleByIndex(cls, index: int) -> int:
        return cls.handles[index]

    @classmethod
    def nvmlDeviceGetPowerUsage(cls, handle: int) -> int:
        return cls.power_usage_mw[handle]


@pytest.fixture
def fake_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import mfcl.telemetry.power as power_mod

    monkeypatch.setattr(power_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(power_mod.torch.cuda, "device_count", lambda: 2)


def test_energy_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_cuda: None) -> None:
    import mfcl.telemetry.power as power_mod

    monkeypatch.setattr(power_mod, "pynvml", _FakeNvmlModule)

    log_path = tmp_path / "energy.csv"
    monitor = PowerMonitor(
        enabled=True,
        log_path=log_path,
        is_main=True,
        kwh_price_usd=0.25,
        start_sampler=False,
    )

    assert monitor.enabled

    monitor.update_step_context(epoch=1, step_index=1, global_step=1)
    monitor._process_sample(timestamp=0.0, dt=0.0, powers=[100.0, 120.0])
    monitor._process_sample(timestamp=1.0, dt=1.0, powers=[100.0, 120.0])
    monitor._process_sample(timestamp=2.5, dt=1.5, powers=[50.0, 50.0])

    total_wh, total_j = monitor.get_totals()
    assert pytest.approx(total_j, rel=1e-6) == 370.0
    assert pytest.approx(total_wh, rel=1e-6) == pytest.approx(370.0 / 3600.0, rel=1e-6)

    with log_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert len(rows) == 6
    gpu0 = [float(row["energy_J_cum"]) for row in rows if int(row["gpu_id"]) == 0]
    gpu1 = [float(row["energy_J_cum"]) for row in rows if int(row["gpu_id"]) == 1]
    assert gpu0 == sorted(gpu0)
    assert gpu1 == sorted(gpu1)
    assert pytest.approx(gpu0[-1], rel=1e-6) == 175.0
    assert pytest.approx(gpu1[-1], rel=1e-6) == 195.0

    monitor.close()
