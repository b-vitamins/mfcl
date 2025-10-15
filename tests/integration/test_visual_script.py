import subprocess
from pathlib import Path
import sys

import torch


def test_visual_script_generates_pdfs(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # Create a few fake checkpoints with metrics
    for e in range(1, 4):
        torch.save(
            {"metrics": {"loss": float(1.0 / e), "lr": 0.1, "time_per_batch": 0.01}},
            run_dir / f"ckpt_ep{e:04d}.pt",
        )
    script = Path(
        "examples/selfsupervised/imagenet1k/resnet18_160/simclr/visual.py"
    ).resolve()
    out_dir = tmp_path / "plots"
    subprocess.run(
        [sys.executable, str(script), "--runs", str(run_dir), "--out", str(out_dir)],
        check=True,
    )
    assert (out_dir / "loss.pdf").exists()
    assert (out_dir / "lr.pdf").exists()
