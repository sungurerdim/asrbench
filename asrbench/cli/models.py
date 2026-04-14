"""asrbench models — register, list, load, and manage ASR models."""

from __future__ import annotations

import httpx
import typer

app = typer.Typer(help="Manage ASR models.")

_DEFAULT_BASE = "http://127.0.0.1:8765"


@app.command("list")
def list_models(
    base_url: str = typer.Option(_DEFAULT_BASE, "--base-url", hidden=True),
) -> None:
    """List all registered models."""
    with httpx.Client(base_url=base_url, timeout=10) as client:
        try:
            resp = client.get("/models")
            resp.raise_for_status()
        except httpx.ConnectError:
            typer.echo(f"Cannot connect to server at {base_url}.", err=True)
            raise typer.Exit(1)

    models = resp.json()
    if not models:
        typer.echo("No models registered. Use 'asrbench models register' to add one.")
        return

    # Simple table output
    typer.echo(f"{'ID':<38} {'NAME':<30} {'BACKEND':<20} {'PATH'}")
    typer.echo("-" * 100)
    for m in models:
        typer.echo(f"{m['model_id']:<38} {m['name']:<30} {m['backend']:<20} {m['local_path']}")


@app.command("register")
def register_model(
    family: str = typer.Option(..., "--family", help="Model family (e.g. whisper)."),
    name: str = typer.Option(..., "--name", help="Model name (e.g. large-v3)."),
    backend: str = typer.Option(..., "--backend", help="Backend name (e.g. faster-whisper)."),
    local_path: str = typer.Option(..., "--path", help="Path to the model directory."),
    base_url: str = typer.Option(_DEFAULT_BASE, "--base-url", hidden=True),
) -> None:
    """Register a new model."""
    payload = {"family": family, "name": name, "backend": backend, "local_path": local_path}
    with httpx.Client(base_url=base_url, timeout=10) as client:
        try:
            resp = client.post("/models", json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            typer.echo(f"Error: {exc.response.status_code} — {exc.response.text}", err=True)
            raise typer.Exit(1)
        except httpx.ConnectError:
            typer.echo(f"Cannot connect to server at {base_url}.", err=True)
            raise typer.Exit(1)

    m = resp.json()
    typer.echo(f"Registered: {m['model_id']} ({m['name']})")


@app.command("load")
def load_model(
    model_id: str = typer.Argument(..., help="Model UUID to load."),
    base_url: str = typer.Option(_DEFAULT_BASE, "--base-url", hidden=True),
) -> None:
    """Load a model into memory."""
    with httpx.Client(base_url=base_url, timeout=60) as client:
        try:
            resp = client.post(f"/models/{model_id}/load")
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            typer.echo(f"Error: {exc.response.status_code} — {exc.response.text}", err=True)
            raise typer.Exit(1)
        except httpx.ConnectError:
            typer.echo(f"Cannot connect to server at {base_url}.", err=True)
            raise typer.Exit(1)

    r = resp.json()
    typer.echo(
        f"Loaded: {r['model_id']} — VRAM {r['vram_used_mb']:.0f}/{r['vram_total_mb']:.0f} MB"
    )


@app.command("unload")
def unload_model(
    model_id: str = typer.Argument(..., help="Model UUID to unload."),
    base_url: str = typer.Option(_DEFAULT_BASE, "--base-url", hidden=True),
) -> None:
    """Unload a model from memory."""
    with httpx.Client(base_url=base_url, timeout=30) as client:
        try:
            resp = client.post(f"/models/{model_id}/unload")
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            typer.echo(f"Error: {exc.response.status_code} — {exc.response.text}", err=True)
            raise typer.Exit(1)
        except httpx.ConnectError:
            typer.echo(f"Cannot connect to server at {base_url}.", err=True)
            raise typer.Exit(1)

    typer.echo(f"Unloaded: {model_id}")
