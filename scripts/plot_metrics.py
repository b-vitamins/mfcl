from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import torch


def _load_metrics(run_dir: Path) -> Dict[str, List[tuple[int, float]]]:
    metrics: Dict[str, List[tuple[int, float]]] = {}
    for ckpt in sorted(run_dir.glob("ckpt_ep*.pt")):
        try:
            state = torch.load(ckpt, map_location="cpu")
        except Exception as exc:  # pragma: no cover - exercised in integration
            raise RuntimeError(f"Failed to load checkpoint {ckpt}: {exc}") from exc
        if not isinstance(state, dict):
            continue
        payload = state.get("metrics")
        if not isinstance(payload, dict):
            continue
        stem = ckpt.stem
        try:
            epoch = int(stem.split("ep")[-1])
        except Exception:
            continue
        for key, value in payload.items():
            try:
                scalar = float(value)
            except Exception:
                continue
            metrics.setdefault(key, []).append((epoch, scalar))
    return metrics


def _plot_metric(out_dir: Path, name: str, series: Iterable[tuple[int, float]]) -> None:
    points = sorted(series, key=lambda item: item[0])
    if not points:
        return
    epochs = [p[0] for p in points]
    values = [p[1] for p in points]
    plt.figure()
    plt.plot(epochs, values, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(name)
    plt.title(name)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"{name}.pdf")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render PDF plots for metrics stored in training checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runs", required=True, type=Path, help="Run directory containing ckpt_epXXXX.pt files.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for generated PDFs (defaults to <runs>/plots).",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=("loss", "lr", "time_per_batch", "imgs_per_sec", "knn_top1"),
        help="Subset of metric names to export. Defaults to common training metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.runs
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    out_dir = args.out or (run_dir / "plots")

    all_metrics = _load_metrics(run_dir)
    requested = set(args.metrics) if args.metrics else set(all_metrics.keys())
    missing = requested - set(all_metrics.keys())
    if missing:
        # Only warn; continue plotting available metrics.
        print(f"[plot_metrics] skipping missing metrics: {', '.join(sorted(missing))}")
    for name in sorted(requested & set(all_metrics.keys())):
        _plot_metric(out_dir, name, all_metrics[name])
    print(f"[plot_metrics] wrote PDFs to {out_dir}")


if __name__ == "__main__":
    main()
