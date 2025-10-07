# pyright: reportGeneralTypeIssues=false
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, cast

import matplotlib.pyplot as plt
import pandas as pd

from mfcl.plotting.style import apply as apply_style


def _ensure_outdir(outdir: str | Path) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def fig_error_vs_tau(
    fidelity_errors_csv: str | Path,
    *,
    outdir: str | Path,
    dataset: Optional[str] = None,
    N: Optional[int] = None,
    objective: Optional[str] = None,
):
    apply_style()
    df = cast(pd.DataFrame, pd.read_csv(fidelity_errors_csv))
    if dataset:
        df = df[df["dataset"] == dataset]
    if objective:
        df = df[df["objective"] == objective]
    if N:
        df = df[df["N"] == N]

    base = cast(pd.DataFrame, df[df["method"].isin(["mf2", "hybrid"])].copy())
    if base.empty:
        raise ValueError("No mf2/hybrid rows in fidelity errors CSV")

    outdir = _ensure_outdir(outdir)
    covs: List[str] = [str(c) for c in sorted(base["covariance"].unique().tolist())]
    for cov in covs:
        sub = base[base["covariance"] == cov]
        grouped = cast(
            pd.DataFrame,
            (
                sub.groupby(["tau", "method", "shrinkage", "covariance", "N"])
                .agg(
                    mae_mean=("mae", "mean"),
                    mae_std=("mae", "std"),
                    count=("mae", "count"),
                )
                .reset_index()
            ),
        )
        grouped["mae_ci"] = 1.96 * grouped["mae_std"] / (grouped["count"] ** 0.5)

        plt.figure()
        for keys, data in grouped.groupby(["method", "shrinkage"]):
            method, shrink = cast(Tuple[str, float], keys)
            data = data.sort_values("tau")
            label = f"{method}, λ={shrink:.2f}" if cov == "full" else method
            plt.plot(data["tau"], data["mae_mean"], marker="o", label=label)
            plt.fill_between(
                data["tau"],
                data["mae_mean"] - data["mae_ci"],
                data["mae_mean"] + data["mae_ci"],
                alpha=0.2,
            )

        ttl = f"{dataset or ''} N={N or 'var'} | {objective or ''} | covariance={cov}"
        plt.title(ttl.strip())
        plt.xlabel("τ")
        plt.ylabel("MAE (approx − exact)")
        plt.legend()
        fname = f"fig_error_vs_tau_{cov}_N{N or 'var'}.png"
        plt.savefig(Path(outdir) / fname)
        plt.close()


def fig_pos_reentry_summary(
    pos_reentry_csv: str | Path,
    *,
    outdir: str | Path,
    dataset: Optional[str] = None,
    N: Optional[int] = None,
):
    apply_style()
    df = cast(pd.DataFrame, pd.read_csv(pos_reentry_csv))
    if dataset:
        df = df[df["dataset"] == dataset]
    if N:
        df = df[df["N"] == N]

    grouped = cast(
        pd.DataFrame,
        (
            df.groupby(["tau"]).agg(
                delta_med=("delta_median", "mean"),
                delta_q25=("delta_q25", "mean"),
                delta_q75=("delta_q75", "mean"),
                frac_pos_dom=("frac_pos_dominates", "mean"),
            )
        )
        .reset_index()
        .sort_values("tau"),
    )

    outdir = _ensure_outdir(outdir)

    plt.figure()
    plt.plot(grouped["tau"], grouped["delta_med"], marker="o", label="median Δ")
    plt.fill_between(
        grouped["tau"],
        grouped["delta_q25"],
        grouped["delta_q75"],
        alpha=0.2,
        label="IQR",
    )
    plt.xlabel("τ")
    plt.ylabel("Positive re-entry Δ")
    plt.title(f"{dataset or ''} N={N or 'var'} | Δ_pos summary".strip())
    plt.legend()
    plt.savefig(Path(outdir) / f"fig_pos_reentry_delta_N{N or 'var'}.png")
    plt.close()

    plt.figure()
    plt.plot(grouped["tau"], grouped["frac_pos_dom"], marker="o")
    plt.xlabel("τ")
    plt.ylabel("Frac(a_pos > m_neg)")
    plt.title(f"{dataset or ''} N={N or 'var'} | positive dominates".strip())
    plt.savefig(Path(outdir) / f"fig_pos_reentry_frac_N{N or 'var'}.png")
    plt.close()
