from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from .plot_utils import configure_style, ensure_output_dir, load_table, save_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot energy per image vs throughput")
    parser.add_argument("--input", required=True, help="CSV with energy metrics")
    parser.add_argument("--output-dir", default=".", help="Directory for figures")
    parser.add_argument(
        "--output-prefix",
        default="energy_per_image",
        help="Filename prefix for rendered plots",
    )
    args = parser.parse_args()

    rows = load_table(args.input)
    if not rows:
        raise SystemExit("Input CSV must contain at least one row")

    throughput_key = "images_per_s" if "images_per_s" in rows[0] else "throughput"
    energy_key = "energy_mj" if "energy_mj" in rows[0] else "energy_per_image_mj"
    if throughput_key not in rows[0] or energy_key not in rows[0]:
        raise SystemExit("CSV must provide throughput and energy columns")

    throughputs = [float(row.get(throughput_key, 0.0)) for row in rows]
    energies = [float(row.get(energy_key, 0.0)) for row in rows]
    labels = [str(row.get("method", f"config_{idx}")) for idx, row in enumerate(rows)]

    configure_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.scatter(throughputs, energies, c=range(len(rows)), cmap="viridis")
    for label, x, y in zip(labels, throughputs, energies):
        ax.text(x, y, label, fontsize=9, ha="left", va="bottom")
    ax.set_xlabel("Throughput (images / s)")
    ax.set_ylabel("Energy per image (mJ)")
    ax.set_title("Energy efficiency")

    output_dir = ensure_output_dir(args.output_dir)
    save_figure(fig, output_dir, args.output_prefix)
    plt.close(fig)


if __name__ == "__main__":
    main()
