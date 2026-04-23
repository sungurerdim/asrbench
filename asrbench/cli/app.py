"""CLI entry point — aggregates all subcommands under ``asrbench``."""

from __future__ import annotations

import typer

from asrbench import __version__
from asrbench.cli import compare as compare_cmd
from asrbench.cli import config_cmd, datasets, doctor, models, optimize, power
from asrbench.cli import run as run_cmd
from asrbench.cli.serve import serve

app = typer.Typer(
    name="asrbench",
    help="ASRbench — fully local, multi-backend ASR benchmarking platform.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"asrbench {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print the package version and exit.",
    ),
) -> None:
    """Root callback — handles top-level options like ``--version``."""
    _ = version  # consumed by the eager callback


app.add_typer(config_cmd.app, name="config")
app.add_typer(datasets.app, name="datasets")
app.add_typer(models.app, name="models")
app.add_typer(optimize.app, name="optimize")
app.add_typer(power.app, name="power")
app.add_typer(run_cmd.app, name="run")
app.add_typer(compare_cmd.app, name="compare")
app.add_typer(doctor.app, name="doctor")
app.command("serve")(serve)

if __name__ == "__main__":
    app()
