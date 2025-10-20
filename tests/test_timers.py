import csv
import time
from contextlib import nullcontext
from pathlib import Path

from mfcl.telemetry.timers import StepTimer


def test_segments_sum_close_to_total(tmp_path):
    log_path = tmp_path / "timings.csv"
    timer = StepTimer(
        enabled=True,
        warmup_steps=0,
        sample_rate=1,
        log_path=log_path,
        nvtx_enabled=False,
        is_main=True,
    )
    try:
        timer.begin_step(epoch=1, step_index=1, global_step=1)
        step_start = time.perf_counter()
        time.sleep(0.005)
        data_elapsed = time.perf_counter() - step_start
        timer.record_data(data_elapsed)
        with timer.range_forward():
            time.sleep(0.002)
        with timer.range_backward():
            time.sleep(0.002)
        with timer.range_optimizer():
            time.sleep(0.001)
        with timer.range_assign():
            time.sleep(0.0005)
        with timer.range_topr():
            time.sleep(0.0004)
        with timer.range_beta_ctrl():
            time.sleep(0.0003)
        with timer.range_misc():
            time.sleep(0.0006)
        step_end = time.perf_counter()
        dt = step_end - step_start
        ips = 128 / dt
        timer.end_step(step_time_s=dt, ips=ips)
    finally:
        timer.close()

    with log_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == 1
    row = rows[0]
    total_ms = float(row["t_step_ms"])
    segments_sum = sum(
        float(row[name])
        for name in (
            "t_data_ms",
            "t_fwd_ms",
            "t_bwd_ms",
            "t_opt_ms",
            "t_comm_ms",
            "t_assign_ms",
            "t_topr_ms",
            "t_beta_ctrl_ms",
            "t_misc_ms",
        )
    )
    discrepancy = abs(segments_sum - total_ms)
    assert discrepancy <= 0.05 * total_ms + 1e-3
    assert "outlier_flags" in row


def test_named_ranges_exist():
    timer = StepTimer(
        enabled=False,
        warmup_steps=0,
        sample_rate=1,
        log_path=None,
        nvtx_enabled=False,
        is_main=True,
    )
    with timer.range_topr():
        pass
    with timer.range_beta_ctrl():
        pass
    timer.close()


def _run_synthetic_loop(base_dir: Path, enabled: bool) -> float:
    samples_per_step = 64
    steps = 200
    data_sleep = 0.002
    fwd_sleep = 0.001
    bwd_sleep = 0.001
    opt_sleep = 0.0008
    misc_sleep = 0.0005

    base_dir.mkdir(parents=True, exist_ok=True)
    log_path = base_dir / "timings.csv" if enabled else None
    timer = (
        StepTimer(
            enabled=True,
            warmup_steps=0,
            sample_rate=1,
            log_path=log_path,
            nvtx_enabled=False,
            is_main=True,
        )
        if enabled
        else None
    )

    total_start = time.perf_counter()
    try:
        for idx in range(steps):
            if timer is not None:
                timer.begin_step(epoch=1, step_index=idx + 1, global_step=idx + 1)
            step_start = time.perf_counter()
            time.sleep(data_sleep)
            data_elapsed = time.perf_counter() - step_start
            if timer is not None:
                timer.record_data(data_elapsed)
            ctx = timer.range_forward() if timer is not None else nullcontext()
            with ctx:
                time.sleep(fwd_sleep)
            ctx = timer.range_backward() if timer is not None else nullcontext()
            with ctx:
                time.sleep(bwd_sleep)
            ctx = timer.range_optimizer() if timer is not None else nullcontext()
            with ctx:
                time.sleep(opt_sleep)
            ctx = timer.range_misc() if timer is not None else nullcontext()
            with ctx:
                time.sleep(misc_sleep)
            step_end = time.perf_counter()
            dt = step_end - step_start
            if timer is not None:
                ips = samples_per_step / dt if dt > 0 else 0.0
                timer.end_step(step_time_s=dt, ips=ips)
    finally:
        if timer is not None:
            timer.close()
    total_time = time.perf_counter() - total_start
    return (steps * samples_per_step) / total_time


def test_overhead_small(tmp_path):
    ips_disabled = _run_synthetic_loop(tmp_path / "disabled", enabled=False)
    ips_enabled = _run_synthetic_loop(tmp_path / "enabled", enabled=True)
    assert ips_disabled > 0.0
    drop = (ips_disabled - ips_enabled) / ips_disabled
    assert drop < 0.02
