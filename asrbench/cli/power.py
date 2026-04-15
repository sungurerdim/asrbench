"""
`asrbench power` — statistical-power helper for benchmark planning.

Use this to pick a sensible ``eps_min`` / convergence epsilon for an
optimization run before committing hours of GPU time to it. Given a dataset
cap (in seconds) and a few corpus assumptions, it reports the WER bootstrap
noise floor and recommends the safety-padded epsilon — the same value
``suggest_epsilon()`` returns from the library.

Typical invocation:

    asrbench power suggest --duration 2400
    asrbench power suggest --duration 900 --wpm 130 --base-wer 0.15

The command prints a single stanza; no files are written. It is safe to run
without a server process, since it only exercises pure-math helpers.
"""

from __future__ import annotations

import typer

from asrbench.engine.search.significance import suggest_epsilon

app = typer.Typer(help="Statistical-power helper for benchmark planning.")


@app.callback()
def _callback() -> None:  # noqa: F841 — registered by @app.callback()
    """
    Keep the ``suggest`` subcommand addressable.

    Typer collapses single-command apps into a flat command by default, which
    would break ``asrbench power suggest`` (the user would have to type just
    ``asrbench power`` with options). A no-op callback forces Typer to keep
    the subcommand structure.
    """


@app.command("suggest")
def suggest(
    duration: int = typer.Option(
        ...,
        "--duration",
        help="Dataset cap in seconds (e.g. 900 for 15 min, 2400 for 40 min).",
    ),
    wpm: float = typer.Option(
        150.0,
        "--wpm",
        help="Assumed words per minute for this corpus (default: 150).",
    ),
    base_wer: float = typer.Option(
        0.10,
        "--base-wer",
        help="Expected WER at the operating point (default: 0.10).",
    ),
    safety: float = typer.Option(
        2.0,
        "--safety",
        help="Safety multiplier on top of the raw SE (default: 2.0 = 2-sigma).",
    ),
) -> None:
    """Print the recommended convergence epsilon for a given dataset size."""
    eps = suggest_epsilon(duration, wpm=wpm, base_wer=base_wer, safety=safety)
    n_words = int(duration * wpm / 60)
    se = (base_wer * (1.0 - base_wer) / max(1, n_words)) ** 0.5
    typer.echo(f"Duration: {duration}s ({n_words} words, {wpm:.0f} wpm)")
    typer.echo(f"WER SE:   {se * 100:.2f}% (base WER {base_wer * 100:.0f}%)")
    typer.echo(f"Suggested epsilon: {eps}  ({safety}x margin)")
