import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def _load_metrics(run_dir: Path):
    metrics_by_epoch = {}
    for ckpt in sorted(run_dir.glob("ckpt_ep*.pt")):
        match = re.search(r"ckpt_ep(\d+)\.pt", ckpt.name)
        if not match:
            continue
        epoch = int(match.group(1))
        try:
            state = torch.load(str(ckpt), map_location="cpu")
        except Exception:
            continue
        metrics = state.get("metrics", {}) if isinstance(state, dict) else {}
        if isinstance(metrics, dict):
            metrics_by_epoch[epoch] = {
                key: float(value)
                if isinstance(value, (int, float))
                or (hasattr(value, "item") and callable(value.item))
                else None
                for key, value in metrics.items()
            }
    return metrics_by_epoch


def _looks_like_timestamp(name: str) -> bool:
    return (name.startswith("20") and any(ch.isdigit() for ch in name)) or all(
        ch in "0123456789-_:T" for ch in name
    )


def _plot_series(runs, metrics_by_run, out_dir: Path, key: str, ylabel: str, fmt: str) -> None:
    plt.figure(figsize=(6, 4))
    for run in runs:
        epochs_to_metrics = metrics_by_run.get(run, {})
        if not epochs_to_metrics:
            continue
        xs = sorted(epochs_to_metrics.keys())
        ys = [epochs_to_metrics[epoch].get(key, None) for epoch in xs]
        filtered = [(epoch, value) for epoch, value in zip(xs, ys) if value is not None]
        if not filtered:
            continue
        xs_plot, ys_plot = zip(*filtered)
        parent = run.parent.name if run.parent else ""
        label = (
            parent
            if parent
            and (_looks_like_timestamp(run.name) or not _looks_like_timestamp(parent))
            else run.name
        )
        plt.plot(xs_plot, ys_plot, label=label, linewidth=2)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle=":", linewidth=0.5)
    if len(runs) > 1:
        plt.legend(frameon=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{key}.{fmt}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--format", choices=["pdf", "svg"], default="pdf")
    args = parser.parse_args()

    run_dirs = [Path(run).resolve() for run in args.runs]
    out_dir = Path(args.out) if args.out else (run_dirs[0] / "plots")
    metrics_by_run = {run: _load_metrics(run) for run in run_dirs}

    series = [
        ("loss", "loss"),
        ("lr", "learning rate"),
        ("time_per_batch", "time per batch (s)"),
        ("pos_sim", "positive cosine"),
        ("neg_sim_mean", "negative cosine mean"),
        ("cos_sim", "cosine (BYOL/SimSiam)"),
        ("diag_mean", "diag mean (Barlow)"),
        ("offdiag_mean", "offdiag mean (Barlow)"),
        ("mse", "mse (VICReg)"),
        ("std_mean", "std mean (VICReg)"),
        ("cov_offdiag", "cov offdiag (VICReg)"),
        ("entropy", "code entropy (SwAV)"),
        ("q_max_mean", "Q max mean (SwAV)"),
    ]

    for key, ylabel in series:
        _plot_series(run_dirs, metrics_by_run, out_dir, key, ylabel, args.format)


if __name__ == "__main__":
    main()

