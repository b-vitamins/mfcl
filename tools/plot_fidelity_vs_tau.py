from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from .plot_utils import configure_style, ensure_output_dir, load_table, save_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot fidelity metrics vs tau")
    parser.add_argument("--input", required=True, help="CSV with fidelity sweep results")
    parser.add_argument("--output-dir", default=".", help="Directory for figures")
    parser.add_argument(
        "--output-prefix",
        default="fidelity_vs_tau",
        help="Filename prefix for rendered plots",
    )
    args = parser.parse_args()

    rows = load_table(args.input)
    if not rows:
        raise SystemExit("Input CSV must contain at least one row")

    tau_values = [float(row.get("tau", 0.0)) for row in rows]
    fidelity_key = "fidelity" if "fidelity" in rows[0] else "distance"
    fidelities = [float(row.get(fidelity_key, 0.0)) for row in rows]
    ci_key = "ci95" if "ci95" in rows[0] else None
    lower_key = "lower" if "lower" in rows[0] else None
    upper_key = "upper" if "upper" in rows[0] else None

    yerr = None
    if ci_key is not None:
        yerr = [float(row.get(ci_key, 0.0)) for row in rows]
    elif lower_key and upper_key:
        lower = [float(row.get(lower_key, 0.0)) for row in rows]
        upper = [float(row.get(upper_key, 0.0)) for row in rows]
        yerr = [abs(f - l) for f, l in zip(fidelities, lower)]

    configure_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    if yerr is not None:
        ax.errorbar(tau_values, fidelities, yerr=yerr, fmt="-o", color="C2", capsize=4)
    else:
        ax.plot(tau_values, fidelities, marker="o", color="C2")
    ax.set_xlabel("Tau")
    ax.set_ylabel("Fidelity")
    ax.set_title("Fidelity vs tau")

    output_dir = ensure_output_dir(args.output_dir)
    save_figure(fig, output_dir, args.output_prefix)
    plt.close(fig)


if __name__ == "__main__":
    main()
