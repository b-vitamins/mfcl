#!/usr/bin/env python3
"""Plot training curves (loss, lr) from saved checkpoints.

Usage:
  python visual.py --runs /path/to/run_dir --out /path/to/out_dir

The script scans for files named ckpt_ep*.pt under --runs, extracts the
`metrics` dict saved at training time, and writes loss.pdf and lr.pdf to --out.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def _collect_metrics(run_dir: Path) -> Tuple[List[int], Dict[str, List[float]]]:
    epochs: List[int] = []
    series: Dict[str, List[float]] = {"loss": [], "lr": []}
    # Collect files matching ckpt_ep*.pt, sort by epoch
    ckpts = sorted(run_dir.glob("ckpt_ep*.pt"))
    for p in ckpts:
        try:
            state = torch.load(p, map_location="cpu")
        except Exception:
            continue
        metrics = state.get("metrics", {}) if isinstance(state, dict) else {}
        # Parse epoch number from filename
        try:
            ep = int(p.stem.split("ep")[-1])
        except Exception:
            ep = len(epochs) + 1
        if "loss" in metrics and "lr" in metrics:
            epochs.append(ep)
            series["loss"].append(float(metrics["loss"]))
            series["lr"].append(float(metrics["lr"]))
    # Sort by epoch just in case
    order = sorted(range(len(epochs)), key=lambda i: epochs[i])
    epochs = [epochs[i] for i in order]
    for k in list(series.keys()):
        series[k] = [series[k][i] for i in order]
    return epochs, series


def _save_plots(out_dir: Path, epochs: List[int], series: Dict[str, List[float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # Loss
    if series["loss"]:
        plt.figure()
        plt.plot(epochs, series["loss"], marker="o")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Training Loss")
        plt.grid(True, alpha=0.3)
        plt.savefig(out_dir / "loss.pdf", bbox_inches="tight")
        plt.close()

    # LR
    if series["lr"]:
        plt.figure()
        plt.plot(epochs, series["lr"], marker="o")
        plt.xlabel("epoch")
        plt.ylabel("learning rate")
        plt.title("Learning Rate")
        plt.grid(True, alpha=0.3)
        plt.savefig(out_dir / "lr.pdf", bbox_inches="tight")
        plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, required=True, help="Run directory")
    ap.add_argument("--out", type=str, required=True, help="Output directory")
    args = ap.parse_args()

    run_dir = Path(args.runs)
    out_dir = Path(args.out)
    epochs, series = _collect_metrics(run_dir)
    _save_plots(out_dir, epochs, series)


if __name__ == "__main__":
    main()
