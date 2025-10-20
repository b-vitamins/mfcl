from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from .plot_utils import configure_style, ensure_output_dir, load_table, save_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot peak memory vs batch size")
    parser.add_argument("--input", required=True, help="CSV with memory telemetry")
    parser.add_argument("--output-dir", default=".", help="Directory for figures")
    parser.add_argument(
        "--output-prefix",
        default="memory_vs_batch",
        help="Filename prefix for rendered plots",
    )
    args = parser.parse_args()

    rows = load_table(args.input)
    if not rows:
        raise SystemExit("Input CSV must contain at least one row")

    batch_sizes = [float(row.get("batch_size", 0.0)) for row in rows]
    if not any(batch_sizes):
        raise SystemExit("Input CSV must include batch_size column")
    peak_key = "peak_gb" if "peak_gb" in rows[0] else "max_gb"
    peak_values = [float(row.get(peak_key, 0.0)) for row in rows]

    configure_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(batch_sizes, peak_values, marker="o", color="C1")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Peak memory (GB)")
    ax.set_title("Memory footprint vs batch size")

    output_dir = ensure_output_dir(args.output_dir)
    save_figure(fig, output_dir, args.output_prefix)
    plt.close(fig)


if __name__ == "__main__":
    main()
