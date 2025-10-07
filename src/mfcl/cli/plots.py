from __future__ import annotations

from typing import Optional

import typer

from mfcl.plotting.plots import fig_error_vs_tau, fig_pos_reentry_summary

app = typer.Typer(
    no_args_is_help=True, add_completion=False, help="Generate paper-ready plots."
)


@app.command("fidelity")
def plots_fidelity(
    fidelity_errors_csv: str = typer.Option(
        ..., "--errors", help="Path to fidelity_errors.csv"
    ),
    outdir: str = typer.Option(
        "results/figures", "--outdir", help="Output directory for figures"
    ),
    dataset: Optional[str] = typer.Option(None, "--dataset"),
    N: Optional[int] = typer.Option(None, "--N"),
    objective: Optional[str] = typer.Option(None, "--objective"),
) -> None:
    fig_error_vs_tau(
        fidelity_errors_csv,
        outdir=outdir,
        dataset=dataset,
        N=N,
        objective=objective,
    )


@app.command("posreentry")
def plots_posreentry(
    pos_reentry_csv: str = typer.Option(
        ..., "--pos", help="Path to pos_reentry_stats.csv"
    ),
    outdir: str = typer.Option(
        "results/figures", "--outdir", help="Output directory for figures"
    ),
    dataset: Optional[str] = typer.Option(None, "--dataset"),
    N: Optional[int] = typer.Option(None, "--N"),
) -> None:
    fig_pos_reentry_summary(
        pos_reentry_csv,
        outdir=outdir,
        dataset=dataset,
        N=N,
    )


if __name__ == "__main__":
    app()
