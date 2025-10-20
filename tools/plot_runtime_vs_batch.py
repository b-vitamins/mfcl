from __future__ import annotations

import argparse
from typing import Sequence

import matplotlib.pyplot as plt

from .plot_utils import configure_style, ensure_output_dir, load_table, save_figure


def _get_batch_sizes(rows: Sequence[dict[str, object]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get("batch_size") or row.get("global_batch")
        if isinstance(raw, (int, float)):
            values.append(float(raw))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot runtime breakdown vs. batch size")
    parser.add_argument("--input", required=True, help="CSV file with timing data")
    parser.add_argument(
        "--output-dir", default=".", help="Directory to store generated figures"
    )
    parser.add_argument(
        "--output-prefix",
        default="runtime_vs_batch",
        help="Filename prefix for the rendered figures",
    )
    args = parser.parse_args()

    rows = load_table(args.input)
    if not rows:
        raise SystemExit("No data rows found in input CSV")
    batch_sizes = _get_batch_sizes(rows)
    if not batch_sizes:
        raise SystemExit("Input CSV must include batch_size column")

    configure_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    stack_candidates = [
        ("t_data_ms", "Data"),
        ("t_fwd_ms", "Forward"),
        ("t_bwd_ms", "Backward"),
        ("t_opt_ms", "Optimizer"),
        ("t_misc_ms", "Misc"),
        ("t_assign_ms", "Assign"),
        ("t_topr_ms", "Top-R"),
        ("t_beta_ctrl_ms", "Beta Ctrl"),
        ("t_comm_ms", "Communication"),
    ]
    available = [cand for cand in stack_candidates if any(cand[0] in row for row in rows)]
    bottoms = [0.0 for _ in batch_sizes]
    bar_handles: list = []
    for key, label in available:
        heights = [float(row.get(key, 0.0)) for row in rows]
        handle = ax.bar(batch_sizes, heights, bottom=bottoms, width=0.6, label=label)
        bar_handles.append(handle)
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_xlabel("Batch size")
    ax.set_ylabel("Step time (ms)")
    ax.set_title("Runtime breakdown vs batch size")

    comm_mb = [float(row.get("comm_bytes", 0.0)) / 1e6 for row in rows]
    ax2 = ax.twinx()
    line, = ax2.plot(batch_sizes, comm_mb, marker="o", color="C7", label="Comm MB")
    ax2.set_ylabel("Communication (MB)")

    handles = []
    labels = []
    for handle in bar_handles:
        handles.append(handle[0])
        labels.append(handle.get_label())
    handles.append(line)
    labels.append("Comm MB")
    ax.legend(handles, labels, loc="upper left", frameon=False)

    output_dir = ensure_output_dir(args.output_dir)
    save_figure(fig, output_dir, args.output_prefix)
    plt.close(fig)


if __name__ == "__main__":
    main()
