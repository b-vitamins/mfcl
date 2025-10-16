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


def _looks_like_timestamp(s: str) -> bool:
    return (s.startswith("20") and any(ch.isdigit() for ch in s)) or all(
        ch in "0123456789-_:T" for ch in s
    )


def _plot_series(runs, metrics_by_run, out_dir: Path, key: str, ylabel: str, fmt: str) -> None:
    plt.figure(figsize=(6, 4))
    for run in runs:
        metrics = metrics_by_run.get(run, {})
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
            parent = run.parent.name if run.parent else ""
            label = parent if parent and (
                _looks_like_timestamp(run.name) or not _looks_like_timestamp(parent)
            ) else run.name
            plt.plot(xs2, ys2, label=label, linewidth=2)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--format", choices=["pdf", "svg"], default="pdf")
    args = ap.parse_args()
    run_paths = [Path(r).resolve() for r in args.runs]
    out_dir = Path(args.out) if args.out else (run_paths[0] / "plots")
    metrics_by_run = {rp: _load_metrics(rp) for rp in run_paths}
    keys = [
        ("loss", "loss"),
        ("lr", "learning rate"),
        ("time_per_batch", "time per batch (s)"),
        # VICReg diagnostics
        ("mse", "mse (VICReg)"),
        ("std_mean", "std mean (VICReg)"),
        ("cov_offdiag", "cov offdiag (VICReg)"),
        # Common diagnostics across methods
        ("pos_sim", "positive cosine"),
        ("neg_sim_mean", "negative cosine mean"),
        ("cos_sim", "cosine (BYOL/SimSiam)"),
        ("diag_mean", "diag mean (Barlow)"),
        ("offdiag_mean", "offdiag mean (Barlow)"),
        ("entropy", "code entropy (SwAV)"),
        ("q_max_mean", "Q max mean (SwAV)"),
    ]
    for key, ylabel in keys:
        _plot_series(run_paths, metrics_by_run, out_dir, key, ylabel, args.format)


if __name__ == "__main__":
    main()
