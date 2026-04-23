"""``asrbench compare`` — side-by-side comparison of two or more runs."""

from __future__ import annotations

import httpx
import typer

app = typer.Typer(help="Compare two or more benchmark runs.")

_DEFAULT_BASE = "http://127.0.0.1:8765"


@app.callback(invoke_without_command=True)
def compare(
    ctx: typer.Context,
    run_ids: list[str] = typer.Argument(
        None,
        help="Two or more run UUIDs to compare.",
    ),
    base_url: str = typer.Option(_DEFAULT_BASE, "--base-url", hidden=True),
) -> None:
    """Print a table of shared vs differing params and per-run deltas."""
    if ctx.invoked_subcommand is not None:
        return
    if not run_ids or len(run_ids) < 2:
        typer.echo("Provide at least 2 run UUIDs.", err=True)
        raise typer.Exit(code=2)

    ids = ",".join(run_ids)
    with httpx.Client(base_url=base_url, timeout=10) as client:
        try:
            resp = client.get("/runs/compare", params={"ids": ids})
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            typer.echo(f"Error: {exc.response.status_code} — {exc.response.text}", err=True)
            raise typer.Exit(code=1) from exc
        except httpx.ConnectError:
            typer.echo(f"Cannot connect to server at {base_url}.", err=True)
            raise typer.Exit(code=1) from None

    data = resp.json()

    typer.echo("Comparing runs:")
    for r in data["runs"]:
        marker = " (baseline)" if r.get("is_baseline") else ""
        agg = r.get("aggregate") or {}
        wer = agg.get("wer_mean")
        delta = r.get("delta_wer_mean")
        wer_s = f"{wer:.4f}" if isinstance(wer, (int, float)) else "—"
        delta_s = (
            f"Δ{delta:+.4f}" if isinstance(delta, (int, float)) and not r.get("is_baseline") else ""
        )
        typer.echo(f"  {r['run_id']}  WER={wer_s}  {delta_s}{marker}")

    if data["params_same"]:
        typer.echo("\nShared params:")
        for key in data["params_same"]:
            value = data["runs"][0].get("params", {}).get(key)
            typer.echo(f"  {key} = {value}")

    if data["params_diff"]:
        typer.echo("\nDiffering params:")
        for key in data["params_diff"]:
            values = [str(r.get("params", {}).get(key)) for r in data["runs"]]
            typer.echo(f"  {key}: {' | '.join(values)}")

    if data.get("wilcoxon_p") is not None:
        p = data["wilcoxon_p"]
        note = "(significant at p<0.05)" if p < 0.05 else "(not significant)"
        typer.echo(f"\nWilcoxon p-value: {p:.4f} {note}")
