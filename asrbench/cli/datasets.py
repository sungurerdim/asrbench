"""asrbench datasets — list and fetch ASR benchmark datasets."""

from __future__ import annotations

import httpx
import typer

app = typer.Typer(help="Manage benchmark datasets.")

_DEFAULT_BASE = "http://127.0.0.1:8765"


@app.command("list")
def list_datasets(
    lang: str = typer.Option("", "--lang", help="Filter by language code."),
    base_url: str = typer.Option(_DEFAULT_BASE, "--base-url", hidden=True),
) -> None:
    """List all available datasets."""
    params = {}
    if lang:
        params["lang"] = lang

    with httpx.Client(base_url=base_url, timeout=10) as client:
        try:
            resp = client.get("/datasets", params=params)
            resp.raise_for_status()
        except httpx.ConnectError:
            typer.echo(f"Cannot connect to server at {base_url}.", err=True)
            raise typer.Exit(1)

    datasets = resp.json()
    if not datasets:
        typer.echo("No datasets. Use 'asrbench datasets fetch' to download one.")
        return

    typer.echo(f"{'ID':<38} {'NAME':<35} {'LANG':<6} {'SPLIT':<12} {'DUR (s)':<10} {'OK'}")
    typer.echo("-" * 110)
    for d in datasets:
        dur = f"{d['duration_s']:.0f}" if d.get("duration_s") else "?"
        ok = "✓" if d.get("verified") else "✗"
        typer.echo(
            f"{d['dataset_id']:<38} {d['name']:<35} {d['lang']:<6} {d['split']:<12} {dur:<10} {ok}"
        )


@app.command("fetch")
def fetch_dataset(
    source: str = typer.Argument(
        ..., help="Dataset source (common_voice, fleurs, yodas, ted_lium, custom)."
    ),
    lang: str = typer.Option("en", "--lang", help="Language code."),
    split: str = typer.Option("test", "--split", help="Split: train, validation, test."),
    local_path: str = typer.Option(
        "", "--local-path", help="For source=custom: path to audio directory."
    ),
    base_url: str = typer.Option(_DEFAULT_BASE, "--base-url", hidden=True),
) -> None:
    """Download and preprocess a dataset."""
    payload: dict = {"source": source, "lang": lang, "split": split}
    if local_path:
        payload["local_path"] = local_path

    with httpx.Client(base_url=base_url, timeout=30) as client:
        try:
            resp = client.post("/datasets/fetch", json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            typer.echo(f"Error: {exc.response.status_code} — {exc.response.text}", err=True)
            raise typer.Exit(1)
        except httpx.ConnectError:
            typer.echo(f"Cannot connect to server at {base_url}.", err=True)
            raise typer.Exit(1)

    data = resp.json()
    typer.echo(f"Fetch started: {data['name']} — follow progress at {base_url}{data['stream_url']}")
