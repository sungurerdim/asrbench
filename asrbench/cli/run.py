"""``asrbench run`` — start, list, and inspect benchmark runs via the API."""

from __future__ import annotations

import httpx
import typer

app = typer.Typer(help="Manage benchmark runs.")

_DEFAULT_BASE = "http://127.0.0.1:8765"


@app.command("start")
def start(
    model_id: str = typer.Option(..., "--model", "-m", help="Model UUID."),
    dataset_id: str = typer.Option(..., "--dataset", "-d", help="Dataset UUID."),
    lang: str = typer.Option("en", "--lang", "-l", help="Language code (ISO 639-1)."),
    params: str = typer.Option("", "--params", help="JSON string of transcription params."),
    label: str = typer.Option("", "--label", "-L", help="Human-readable label for this run."),
    base_url: str = typer.Option(_DEFAULT_BASE, "--base-url", hidden=True),
) -> None:
    """Start a benchmark run."""
    import json

    payload: dict = {"model_id": model_id, "dataset_id": dataset_id, "lang": lang}
    if params:
        try:
            payload["params"] = json.loads(params)
        except json.JSONDecodeError as exc:
            typer.echo(f"Error: invalid --params JSON: {exc}", err=True)
            raise typer.Exit(code=2) from exc
    if label:
        payload["label"] = label

    with httpx.Client(base_url=base_url, timeout=10) as client:
        try:
            resp = client.post("/runs/start", json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            typer.echo(f"Error: {exc.response.status_code} — {exc.response.text}", err=True)
            raise typer.Exit(1)
        except httpx.ConnectError:
            typer.echo(f"Cannot connect to server at {base_url}.", err=True)
            raise typer.Exit(1)

    data = resp.json()
    typer.echo(f"Run started: {data['run_id']} (status: {data['status']})")


@app.command("status")
def status(
    run_id: str = typer.Argument(..., help="Run UUID."),
    base_url: str = typer.Option(_DEFAULT_BASE, "--base-url", hidden=True),
) -> None:
    """Check the status and metrics of a run."""
    with httpx.Client(base_url=base_url, timeout=10) as client:
        try:
            resp = client.get(f"/runs/{run_id}")
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            typer.echo(f"Error: {exc.response.status_code} — {exc.response.text}", err=True)
            raise typer.Exit(1)
        except httpx.ConnectError:
            typer.echo(f"Cannot connect to server at {base_url}.", err=True)
            raise typer.Exit(1)

    data = resp.json()
    typer.echo(f"Run:    {data['run_id']}")
    typer.echo(f"Status: {data['status']}")
    typer.echo(f"Lang:   {data['lang']}")
    agg = data.get("aggregate")
    if agg:
        typer.echo(f"WER:    {agg['wer_mean']:.4f}" if agg.get("wer_mean") is not None else "")
        typer.echo(f"CER:    {agg['cer_mean']:.4f}" if agg.get("cer_mean") is not None else "")
        typer.echo(f"RTFx:   {agg['rtfx_mean']:.2f}" if agg.get("rtfx_mean") is not None else "")
        typer.echo(
            f"Wall:   {agg['wall_time_s']:.1f}s" if agg.get("wall_time_s") is not None else ""
        )


@app.command("list")
def list_runs(
    run_status: str = typer.Option("", "--status", "-s", help="Filter by status."),
    base_url: str = typer.Option(_DEFAULT_BASE, "--base-url", hidden=True),
) -> None:
    """List all benchmark runs."""
    params = {}
    if run_status:
        params["status"] = run_status

    with httpx.Client(base_url=base_url, timeout=10) as client:
        try:
            resp = client.get("/runs", params=params)
            resp.raise_for_status()
        except httpx.ConnectError:
            typer.echo(f"Cannot connect to server at {base_url}.", err=True)
            raise typer.Exit(1)

    runs = resp.json()
    if not runs:
        typer.echo("No runs found.")
        return

    typer.echo(f"{'ID':<38} {'BACKEND':<20} {'LANG':<6} {'STATUS'}")
    typer.echo("-" * 75)
    for r in runs:
        typer.echo(f"{r['run_id']:<38} {r['backend']:<20} {r['lang']:<6} {r['status']}")
