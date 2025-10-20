import json

from omegaconf import OmegaConf

from tools.oom_search import run_search, _simulate_trial


def test_binary_search_finds_threshold(tmp_path, monkeypatch):
    monkeypatch.setenv("MFCL_FAKE_OOM_THRESHOLD", "96")

    summary = run_search(
        min_batch=16,
        max_batch=192,
        steps=4,
        trial_runner=_simulate_trial,
        axes={"loss.covariance_mode": ["diag", "full"]},
        output_dir=tmp_path,
    )

    assert summary.best_batch_size == 96
    assert summary.overrides["train.batch_size"] == 96

    summary_path = tmp_path / "oom_search_summary.json"
    override_path = tmp_path / "oom_override.yaml"
    assert summary_path.exists()
    assert override_path.exists()

    payload = json.loads(summary_path.read_text())
    assert payload["best"]["train.batch_size"] == 96
    assert payload["throughput"] > 0

    override_cfg = OmegaConf.load(override_path)
    assert int(override_cfg.train.batch_size) == 96


def test_binary_search_handles_no_solution(tmp_path, monkeypatch):
    monkeypatch.setenv("MFCL_FAKE_OOM_THRESHOLD", "32")

    try:
        run_search(
            min_batch=64,
            max_batch=128,
            steps=3,
            trial_runner=_simulate_trial,
            output_dir=tmp_path,
        )
    except RuntimeError as exc:
        assert "No stable configuration" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected run_search to fail when no stable batch exists")
