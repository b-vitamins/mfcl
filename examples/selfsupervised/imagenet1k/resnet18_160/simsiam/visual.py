import argparse
import re
from pathlib import Path
import torch
import matplotlib.pyplot as plt


def _load_metrics(run_dir: Path):
    metrics_by_epoch = {}
    for p in sorted(run_dir.glob("ckpt_ep*.pt")):
        m = re.search(r"ckpt_ep(\d+)\.pt", p.name)
        if not m:
            continue
        epoch = int(m.group(1))
        try:
            state = torch.load(str(p), map_location="cpu")
            md = state.get("metrics", {})
            if isinstance(md, dict):
                metrics_by_epoch[epoch] = {
                    k: float(v)
                    for k, v in md.items()
                    if isinstance(v, (int, float)) or hasattr(v, "item")
                }
        except Exception:
            continue
    return metrics_by_epoch


def _plot_series(runs, out_dir: Path, key: str, ylabel: str) -> None:
    plt.figure(figsize=(6, 4))
    for run in runs:
        metrics = _load_metrics(run)
        if not metrics:
            continue
        xs = sorted(metrics.keys())
        ys = [metrics[e].get(key, None) for e in xs]
        xs2, ys2 = (
            zip(*[(x, y) for x, y in zip(xs, ys) if y is not None])
            if any(v is not None for v in ys)
            else ([], [])
        )
        if xs2:
            label = run.parent.name if run.parent else run.name
            plt.plot(xs2, ys2, label=label, linewidth=2)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle=":", linewidth=0.5)
    if len(runs) > 1:
        plt.legend(frameon=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{key}.pdf")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_paths = [Path(r).resolve() for r in args.runs]
    out_dir = Path(args.out) if args.out else (run_paths[0] / "plots")
    keys = [
        ("loss", "loss"),
        ("lr", "learning rate"),
        ("time_per_batch", "time per batch (s)"),
        ("pos_sim", "pos sim"),
        ("neg_sim_mean", "neg sim mean"),
        ("cos_sim", "cos sim"),
        ("diag_mean", "diag mean"),
        ("offdiag_mean", "offdiag mean"),
        ("mse", "mse"),
        ("std_mean", "std mean"),
        ("cov_offdiag", "cov offdiag"),
        ("entropy", "entropy"),
        ("q_max_mean", "q max mean"),
    ]
    for key, ylabel in keys:
        _plot_series(run_paths, out_dir, key, ylabel)


if __name__ == "__main__":
    main()
