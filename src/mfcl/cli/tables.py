from __future__ import annotations

from pathlib import Path

import typer

from mfcl.tables.builders import (
    TableSpec,
    table_fidelity_summary,
    table_gradient_summary,
    table_efficiency_summary,
)

app = typer.Typer(
    no_args_is_help=True, add_completion=False, help="Build paper tables from CSV logs."
)


@app.command("fidelity")
def build_fidelity(
    errors_csv: str = typer.Option(..., "--errors", help="Path to fidelity_errors.csv"),
    outdir: str = typer.Option("results/tables", "--outdir", help="Output directory"),
    dataset: str = typer.Option(..., "--dataset"),
    objective: str = typer.Option(..., "--objective"),
    N: int = typer.Option(8192, "--N"),
    taus: str = typer.Option(
        "0.10,0.25,0.50", "--taus", help="Comma-separated list of τ"
    ),
) -> None:
    tau_vals = [float(t.strip()) for t in taus.split(",") if t.strip()]
    spec = TableSpec(
        name="table_fidelity",
        caption=f"Fidelity summary on {dataset}, N={N}, {objective}. Lower is better.",
        label=f"tab:fidelity_{dataset}_{objective}",
        filters={
            "dataset": [dataset],
            "objective": [objective],
            "N": [N],
            "tau": tau_vals,
            "method": ["mf2", "hybrid"],
        },
        columns={
            "tau": "τ",
            "method": "method",
            "covariance": "cov",
            "shrinkage": "λ",
            "hybrid_k": "k",
            "mae": "MAE",
            "rmse": "RMSE",
            "bias": "Bias",
            "rel_mae": "Rel-MAE",
        },
        precision={"MAE": 4, "RMSE": 4, "Bias": 4, "Rel-MAE": 3},
        groupby=["dataset", "objective", "N", "tau"],
        select_by="mae",
        select_mode="min",
        bold_cols=["MAE"],
        sort_by=[("τ", True)],
    )
    df = table_fidelity_summary(errors_csv, spec)
    out_csv = Path(outdir) / f"table_fidelity_{dataset}_{objective}_N{N}.csv"
    out_tex = Path(outdir) / f"table_fidelity_{dataset}_{objective}_N{N}.tex"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    tex = df.to_latex(escape=False, index=False, caption=spec.caption, label=spec.label)
    out_tex.write_text(tex, encoding="utf-8")


@app.command("lowtau")
def build_lowtau(
    errors_csv: str = typer.Option(..., "--errors"),
    outdir: str = typer.Option("results/tables", "--outdir"),
    dataset: str = typer.Option(..., "--dataset"),
    objective: str = typer.Option("infonce", "--objective"),
    N: int = typer.Option(8192, "--N"),
    tau: float = typer.Option(0.05, "--tau"),
) -> None:
    spec = TableSpec(
        name="table_lowtau",
        caption=f"Low-τ regime on {dataset}, N={N}, τ={tau}.",
        label=f"tab:lowtau_{dataset}_{objective}",
        filters={
            "dataset": [dataset],
            "objective": [objective],
            "N": [N],
            "tau": [tau],
            "method": ["mf2", "hybrid"],
        },
        columns={
            "method": "method",
            "covariance": "cov",
            "shrinkage": "λ",
            "hybrid_k": "k",
            "mae": "MAE",
            "rmse": "RMSE",
            "bias": "Bias",
            "rel_mae": "Rel-MAE",
        },
        precision={"MAE": 4, "RMSE": 4, "Bias": 4, "Rel-MAE": 3},
        groupby=[
            "dataset",
            "objective",
            "N",
            "tau",
            "method",
            "covariance",
            "shrinkage",
            "hybrid_k",
        ],
        select_by="mae",
        select_mode="min",
        bold_cols=["MAE"],
        sort_by=[("MAE", True)],
    )
    df = table_fidelity_summary(errors_csv, spec)
    out_csv = Path(outdir) / f"table_lowtau_{dataset}_{objective}_N{N}.csv"
    out_tex = Path(outdir) / f"table_lowtau_{dataset}_{objective}_N{N}.tex"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    tex = df.to_latex(escape=False, index=False, caption=spec.caption, label=spec.label)
    out_tex.write_text(tex, encoding="utf-8")


@app.command("gradients")
def build_gradients(
    grad_csv: str = typer.Option(..., "--csv"),
    outdir: str = typer.Option("results/tables", "--outdir"),
    dataset: str = typer.Option(..., "--dataset"),
    objective: str = typer.Option("infonce", "--objective"),
    N: int = typer.Option(8192, "--N"),
) -> None:
    df = table_gradient_summary(
        grad_csv,
        filters={"dataset": [dataset], "objective": [objective], "N": [N]},
        columns={
            "covariance": "cov",
            "shrinkage": "λ",
            "cos_all": "cos(all)",
            "cos_X": "cos(X)",
            "cos_Y": "cos(Y)",
            "rel_norm_err_all": "rel-err",
            "sign_agree_topk_X": "sign@k(X)",
            "sign_agree_topk_Y": "sign@k(Y)",
        },
        precision={
            "cos(all)": 3,
            "cos(X)": 3,
            "cos(Y)": 3,
            "rel-err": 3,
            "sign@k(X)": 3,
            "sign@k(Y)": 3,
        },
        groupby=["covariance", "shrinkage"],
        sort_by=[("cos(all)", False)],
        bold_cols=["cos(all)"],
    )
    out_csv = Path(outdir) / f"table_gradients_{dataset}_{objective}_N{N}.csv"
    out_tex = Path(outdir) / f"table_gradients_{dataset}_{objective}_N{N}.tex"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    tex = df.to_latex(
        escape=False,
        index=False,
        caption=f"Gradient fidelity on {dataset}, N={N}.",
        label=f"tab:grad_{dataset}",
    )
    out_tex.write_text(tex, encoding="utf-8")


@app.command("efficiency")
def build_efficiency(
    eff_csv: str = typer.Option(..., "--csv"),
    outdir: str = typer.Option("results/tables", "--outdir"),
    dataset: str = typer.Option(..., "--dataset"),
    world_size: int = typer.Option(..., "--world-size"),
    d: int = typer.Option(..., "--d"),
) -> None:
    df = table_efficiency_summary(
        eff_csv,
        filters={"dataset": [dataset], "world_size": [world_size], "d": [d]},
        columns={
            "method": "method",
            "covariance": "cov",
            "shrinkage": "λ",
            "N_global": "N",
            "ms_median": "ms",
            "peak_mem_gb": "GB",
            "bytes_all_gather_theoretical": "gather_bytes",
            "bytes_all_reduce_theoretical": "reduce_bytes",
        },
        precision={"ms": 1, "GB": 2},
        groupby=["method", "covariance", "shrinkage", "N_global"],
        sort_by=[("N", True), ("method", True)],
        bold_cols=["ms"],
    )
    out_csv = Path(outdir) / f"table_efficiency_{dataset}_D{world_size}_d{d}.csv"
    out_tex = Path(outdir) / f"table_efficiency_{dataset}_D{world_size}_d{d}.tex"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    tex = df.to_latex(
        escape=False,
        index=False,
        caption=f"Efficiency on {dataset} (D={world_size}, d={d}).",
        label=f"tab:eff_{dataset}",
    )
    out_tex.write_text(tex, encoding="utf-8")


if __name__ == "__main__":
    app()
