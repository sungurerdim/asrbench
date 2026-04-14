"""
`asrbench optimize` — drive the IAMS parameter optimizer from the command line.

Typical invocation:

    asrbench optimize \\
        --model whisper-large-v3 \\
        --dataset tr-common-voice-test \\
        --space optimize_space.yaml \\
        --objective wer \\
        --mode maximum \\
        --budget 200 \\
        --output study.json

The command:

    1. Loads the parameter space from YAML
    2. Calls `POST /optimize/start` on the local server (via httpx)
    3. Streams study progress and writes the final study.json locally

This CLI is a thin wrapper over the REST API — it does NOT import IAMS code
directly. Keeping the CLI and the engine decoupled means the same study can
also be driven from a web UI, another script, or a different host.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import typer

app = typer.Typer(help="Run an IAMS parameter optimization study.")

_DEFAULT_BASE = "http://127.0.0.1:8765"


@app.command()
def optimize(
    model_id: str = typer.Option(
        ..., "--model", "-m", help="Model UUID (registered via POST /models)."
    ),
    dataset_id: str = typer.Option(
        ..., "--dataset", "-d", help="Dataset UUID (fetched via POST /datasets)."
    ),
    space_file: Path = typer.Option(
        ..., "--space", "-s", help="Path to YAML parameter space declaration."
    ),
    lang: str = typer.Option("en", "--lang", "-l", help="Language code (ISO 639-1)."),
    objective: str = typer.Option(
        "wer",
        "--objective",
        "-o",
        help="Objective metric: wer|cer|rtfx|vram or 'weighted'.",
    ),
    weights: str | None = typer.Option(
        None,
        "--weights",
        help="Comma-separated weights for --objective weighted (e.g. 'wer=1.0,rtfx=-0.1').",
    ),
    maximize: bool = typer.Option(
        False, "--maximize", help="Maximize the single metric instead of minimizing."
    ),
    mode: str = typer.Option(
        "maximum",
        "--mode",
        help="Accuracy mode: fast (L1-2), balanced (L1-5), maximum (L1-7).",
    ),
    budget: int = typer.Option(200, "--budget", "-b", help="Hard cap on total trial count."),
    eps_min: float = typer.Option(
        0.005,
        "--epsilon",
        "-e",
        help="Minimum significant score difference (practical threshold).",
    ),
    convergence_window: int = typer.Option(
        3,
        "--convergence-window",
        help="Consecutive non-improving trials before early stop. 0 disables.",
    ),
    top_k_pairs: int = typer.Option(
        4,
        "--top-k-pairs",
        help="Top-K most sensitive parameters to include in pairwise 2D scan.",
    ),
    multistart_candidates: int = typer.Option(
        3,
        "--multistart-candidates",
        help="Maximum number of promising points to use as multi-start seeds.",
    ),
    validation_runs: int = typer.Option(
        3,
        "--validation-runs",
        help="Number of fresh re-evaluations for confidence certification.",
    ),
    deep_ablation: bool = typer.Option(
        False,
        "--deep-ablation",
        help="Enable leave-two-out ablation (expensive, off by default).",
    ),
    prior_study: str | None = typer.Option(
        None,
        "--prior-study",
        help="Study UUID to warm-start from (reuse trial cache + screening).",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-O", help="Write study.json here when done."
    ),
    base_url: str = typer.Option(_DEFAULT_BASE, "--base-url", help="asrbench server URL."),
) -> None:
    """Run an IAMS parameter optimization study and write the result to JSON."""
    if not space_file.exists():
        typer.echo(f"Error: space file not found: {space_file}", err=True)
        raise typer.Exit(code=2)

    if mode not in ("fast", "balanced", "maximum"):
        typer.echo(
            f"Error: invalid --mode {mode!r}. Use fast, balanced, or maximum.",
            err=True,
        )
        raise typer.Exit(code=2)

    # Parse the space YAML and validate locally before sending to the server
    try:
        import yaml

        with space_file.open("r", encoding="utf-8") as f:
            space_data = yaml.safe_load(f)
        from asrbench.engine.search.space import ParameterSpace

        space = ParameterSpace.from_dict(space_data)
        typer.echo(f"Loaded space with {len(space.names)} parameters: {', '.join(space.names)}")
    except Exception as exc:
        typer.echo(f"Error: failed to load space file: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    # Build the objective payload
    if objective == "weighted":
        if not weights:
            typer.echo(
                "Error: --objective weighted requires --weights "
                "(e.g. --weights 'wer=1.0,rtfx=-0.1').",
                err=True,
            )
            raise typer.Exit(code=2)
        try:
            weight_map: dict[str, float] = {}
            for pair in weights.split(","):
                key, value = pair.split("=")
                weight_map[key.strip()] = float(value.strip())
        except Exception as exc:
            typer.echo(f"Error: invalid --weights format: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        objective_payload: dict = {"type": "weighted", "weights": weight_map}
    else:
        direction = "maximize" if maximize else None
        objective_payload = {
            "type": "single",
            "metric": objective,
            "direction": direction,
        }

    # Build the request
    payload = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "lang": lang,
        "space": space_data,
        "objective": objective_payload,
        "mode": mode,
        "budget": {
            "hard_cap": budget,
            "convergence_eps": eps_min,
            "convergence_window": convergence_window,
        },
        "eps_min": eps_min,
        "top_k_pairs": top_k_pairs,
        "multistart_candidates": multistart_candidates,
        "validation_runs": validation_runs,
        "enable_deep_ablation": deep_ablation,
        "prior_study_id": prior_study,
    }

    url = f"{base_url}/optimize/start"
    typer.echo(f"POST {url}")
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(url, json=payload)
    except httpx.ConnectError:
        typer.echo(
            f"Error: cannot reach asrbench server at {base_url}. "
            "Is it running? Start with `asrbench serve`.",
            err=True,
        )
        raise typer.Exit(code=1) from None

    if response.status_code >= 400:
        typer.echo(
            f"Error: server returned {response.status_code}: {response.text}",
            err=True,
        )
        raise typer.Exit(code=1)

    data = response.json()
    study_id = data.get("study_id")
    typer.echo(f"Study started: {study_id}")
    typer.echo(
        "Poll `GET /optimize/{study_id}` for status, "
        "or watch the server logs for per-trial progress."
    )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump({"study_id": study_id, **data}, f, indent=2)
        typer.echo(f"Wrote initial study metadata to {output}")

    # Print a final hint about polling
    typer.echo(
        f"\nTo fetch the final study result once complete, run:\n"
        f"  curl {base_url}/optimize/{study_id}"
    )


if __name__ == "__main__":
    try:
        app()
    except typer.Exit as exc:
        sys.exit(exc.exit_code)
