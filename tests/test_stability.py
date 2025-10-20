import json

import pytest
import torch

from mfcl.telemetry.stability import StabilityError, StabilitySentry


def _make_optimizer() -> torch.optim.Optimizer:
    param = torch.nn.Parameter(torch.ones(1))
    return torch.optim.SGD([param], lr=0.1)


def test_sentry_writes_artifacts_on_nan_loss(tmp_path):
    sentry = StabilitySentry(enabled=True, save_dir=tmp_path, is_main=True, max_history=5)
    optimizer = _make_optimizer()
    batch = {"x": torch.ones(2)}
    for step in range(3):
        sentry.record_step(
            step=step + 1,
            epoch=0,
            loss=0.1 * (step + 1),
            step_time_s=0.05,
            comm_bytes=32.0,
        )
    with pytest.raises(StabilityError):
        sentry.check_loss(
            torch.tensor(float("nan")),
            batch=batch,
            optimizer=optimizer,
            scaler=None,
            step=4,
            epoch=0,
        )

    crash_dir = tmp_path / "crash"
    assert (crash_dir / "last_batch.pt").exists()
    report_path = crash_dir / "stability_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["reason"] == "loss_non_finite"
    assert len(report["history"]) > 0


def test_sentry_flags_bad_gradients(tmp_path):
    sentry = StabilitySentry(enabled=True, save_dir=tmp_path, is_main=True, max_history=4)
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
    optimizer = torch.optim.SGD([param], lr=0.01)
    param.grad = torch.tensor([float("nan"), 0.0])

    with pytest.raises(StabilityError):
        sentry.check_gradients(
            optimizer,
            batch=None,
            step=1,
            epoch=0,
        )

    crash_dir = tmp_path / "crash"
    report_path = crash_dir / "stability_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["reason"] == "grad_non_finite"
