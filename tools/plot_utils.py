"""Shared utilities for MFCL plotting scripts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_HOUSE_STYLE = {
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "figure.dpi": 120,
}


def configure_style() -> None:
    """Apply a lightweight, reproducible plot style."""

    plt.rcParams.update(_HOUSE_STYLE)


def load_table(path: str | Path) -> list[dict[str, Any]]:
    """Load a CSV file into a list of dictionaries with numeric coercion."""

    rows: list[dict[str, Any]] = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if value is None:
                    row[key] = value
                    continue
                text = value.strip()
                if text == "":
                    row[key] = text
                    continue
                try:
                    number = float(text)
                except ValueError:
                    row[key] = text
                else:
                    row[key] = number
            rows.append(row)
    return rows


def ensure_output_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_figure(fig: plt.Figure, output_dir: Path, prefix: str) -> None:
    pdf_path = output_dir / f"{prefix}.pdf"
    png_path = output_dir / f"{prefix}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)


__all__ = ["configure_style", "load_table", "ensure_output_dir", "save_figure"]
