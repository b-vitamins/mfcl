from __future__ import annotations

import typer

from mfcl import __version__

app = typer.Typer(
    no_args_is_help=True, add_completion=False, help="MFCL command line tools."
)
run_app = typer.Typer(
    no_args_is_help=True, add_completion=False, help="Run experiments."
)
app.add_typer(run_app, name="run")


@app.command("version")
def version() -> None:
    """Print version."""
    typer.echo(__version__)


@run_app.command("fidelity")
def run_fidelity(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config."),
) -> None:
    """Run normalization fidelity experiment."""
    from mfcl.cli.fidelity import main as fidelity_main

    fidelity_main(config_path=config)


@run_app.command("gradients")
def run_gradients(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config."),
) -> None:
    """Run gradient fidelity experiment."""
    from mfcl.cli.gradients import main as gradients_main

    gradients_main(config_path=config)


@run_app.command("efficiency")
def run_efficiency(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config."),
) -> None:
    """Run efficiency and scaling benchmarks (single- or multi-node)."""
    from mfcl.cli.efficiency import main as efficiency_main

    efficiency_main(config_path=config)


@run_app.command("diagnostics")
def run_diagnostics(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config."),
) -> None:
    """Run spectral and positive re-entry diagnostics."""
    from mfcl.cli.diagnostics import main as diag_main

    diag_main(config_path=config)


@app.command("plots")
def plots() -> None:
    """Plot helpers. Use subcommands like: mfcl plots fidelity ..."""
    from mfcl.cli.plots import app as plots_app

    plots_app()


@run_app.command("tables")
def run_tables() -> None:
    """Build tables from CSV logs. Use subcommands: fidelity, lowtau, gradients, efficiency."""
    from mfcl.cli.tables import app as tables_app

    tables_app()


if __name__ == "__main__":
    app()
