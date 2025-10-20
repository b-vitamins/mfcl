from __future__ import annotations

import argparse
from typing import Sequence

import matplotlib.pyplot as plt

from .plot_utils import configure_style, ensure_output_dir, load_table, save_figure


def _select_key(row: dict[str, object], candidates: Sequence[str]) -> str | None:
    for key in candidates:
        if key in row:
            return key
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot accuracy vs wall-clock time")
    parser.add_argument("--input", required=True, help="CSV with accuracy/time curves")
    parser.add_argument("--output-dir", default=".", help="Directory for figures")
    parser.add_argument(
        "--output-prefix",
        default="accuracy_vs_time",
        help="Filename prefix for rendered plots",
    )
    args = parser.parse_args()

    rows = load_table(args.input)
    if not rows:
        raise SystemExit("Input CSV must contain at least one row")

    time_key = _select_key(rows[0], ["minutes", "time_min", "time_minutes"])
    if time_key is None:
        raise SystemExit("Expected minutes/time_min column in CSV")
    acc_key = _select_key(rows[0], ["top1", "accuracy", "acc"])
    if acc_key is None:
        raise SystemExit("Expected accuracy/top1 column in CSV")

    times = [float(row.get(time_key, 0.0)) for row in rows]
    accs = [float(row.get(acc_key, 0.0)) for row in rows]

    ci_key = _select_key(rows[0], ["ci95", "ci", "stderr"])
    lower_key = _select_key(rows[0], ["lower", "ci_lower", "ci95_lower"])
    upper_key = _select_key(rows[0], ["upper", "ci_upper", "ci95_upper"])
    lowers: list[float] = []
    uppers: list[float] = []
    if ci_key is not None:
        for acc, row in zip(accs, rows):
            delta = float(row.get(ci_key, 0.0))
            lowers.append(acc - delta)
            uppers.append(acc + delta)
    elif lower_key is not None and upper_key is not None:
        lowers = [float(row.get(lower_key, 0.0)) for row in rows]
        uppers = [float(row.get(upper_key, 0.0)) for row in rows]

    configure_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(times, accs, marker="o", color="C0", label="Top-1")
    if lowers and uppers:
        ax.fill_between(times, lowers, uppers, color="C0", alpha=0.2, label="95% CI")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. wall-clock time")
    ax.legend(loc="lower right", frameon=False)

    output_dir = ensure_output_dir(args.output_dir)
    save_figure(fig, output_dir, args.output_prefix)
    plt.close(fig)


if __name__ == "__main__":
    main()
