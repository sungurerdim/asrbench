"""CLI entry point — aggregates all subcommands under ``asrbench``."""

from __future__ import annotations

import typer

from asrbench.cli import datasets, models, optimize, power
from asrbench.cli import run as run_cmd
from asrbench.cli.serve import serve

app = typer.Typer(
    name="asrbench",
    help="ASRbench — fully local, multi-backend ASR benchmarking platform.",
    no_args_is_help=True,
)

app.add_typer(datasets.app, name="datasets")
app.add_typer(models.app, name="models")
app.add_typer(optimize.app, name="optimize")
app.add_typer(power.app, name="power")
app.add_typer(run_cmd.app, name="run")
app.command("serve")(serve)

if __name__ == "__main__":
    app()
