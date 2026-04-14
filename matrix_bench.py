"""matrix_bench.py — Toplu ASRbench test runner.

Kullanım:
    python matrix_bench.py matrix.json
    python matrix_bench.py matrix.json --csv results.csv
    python matrix_bench.py matrix.json --dry-run

Config format: matrix.json (bkz. matrix_example.json)
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_POLL_INTERVAL = 5  # saniye


def _get(client: httpx.Client, path: str) -> dict:
    resp = client.get(path)
    resp.raise_for_status()
    return resp.json()


def _post(client: httpx.Client, path: str, body: dict) -> dict:
    resp = client.post(path, json=body)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def ensure_server(base_url: str) -> bool:
    """Return True if server is reachable, False if it couldn't be started."""
    try:
        with httpx.Client(base_url=base_url, timeout=3) as c:
            c.get("/system/health").raise_for_status()
        print("  [server] already running")
        return True
    except Exception:
        pass

    print("  [server] not running — starting in background...")
    subprocess.Popen(
        [sys.executable, "-m", "asrbench.cli.app", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(20):
        time.sleep(2)
        try:
            with httpx.Client(base_url=base_url, timeout=3) as c:
                c.get("/system/health").raise_for_status()
            print("  [server] ready")
            return True
        except Exception:
            pass

    print("  [server] ERROR: failed to start after 40s", file=sys.stderr)
    return False


# ---------------------------------------------------------------------------
# Model registration
# ---------------------------------------------------------------------------


def register_model(client: httpx.Client, model_cfg: dict) -> str:
    """Register model if not already registered; return model_id."""
    name = model_cfg["name"]
    backend = model_cfg["backend"]
    family = model_cfg.get("family", "whisper")
    local_path = model_cfg.get("local_path", name)

    # Check existing by name + backend
    models = _get(client, "/models")
    for m in models:
        if m["name"] == name and m["backend"] == backend:
            print(f"  [model] reusing existing: {m['model_id'][:8]}…  ({name})")
            return m["model_id"]

    body = {
        "family": family,
        "name": name,
        "backend": backend,
        "local_path": local_path,
    }
    data = _post(client, "/models", body)
    model_id = data["model_id"]
    print(f"  [model] registered: {model_id[:8]}…  ({name})")
    return model_id


def load_model(client: httpx.Client, model_id: str) -> None:
    """Load model into memory (no-op if already loaded)."""
    client.post(f"/models/{model_id}/load").raise_for_status()
    print("  [model] loaded into memory")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def fetch_dataset(client: httpx.Client, run_cfg: dict) -> str:
    """Fetch dataset (idempotent by source+lang+split+duration); return dataset_id."""
    body: dict[str, Any] = {
        "source": run_cfg["dataset"],
        "lang": run_cfg["lang"],
        "split": run_cfg.get("split", "test"),
    }
    if run_cfg.get("max_duration_s"):
        body["max_duration_s"] = run_cfg["max_duration_s"]

    data = _post(client, "/datasets/fetch", body)
    dataset_id = data["dataset_id"]
    return dataset_id


def wait_dataset(client: httpx.Client, dataset_id: str) -> None:
    """Poll until dataset is verified."""
    while True:
        ds = _get(client, f"/datasets/{dataset_id}")
        if ds.get("verified"):
            print(f"  [dataset] ready: {ds['name']}")
            return
        print(f"  [dataset] loading {ds['name']}…")
        time.sleep(_POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def start_run(client: httpx.Client, model_id: str, dataset_id: str, run_cfg: dict) -> str:
    """Start a benchmark run; return run_id."""
    body: dict[str, Any] = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "lang": run_cfg["lang"],
        "label": run_cfg.get("label"),
        "params": run_cfg.get("params") or {},
    }
    data = _post(client, "/runs/start", body)
    return data["run_id"]


def wait_run(client: httpx.Client, run_id: str) -> dict:
    """Poll until run is completed or failed; return full run dict."""
    dots = 0
    while True:
        run = _get(client, f"/runs/{run_id}")
        status = run["status"]
        if status == "completed":
            print("  [run] done")
            return run
        if status == "failed":
            print("  [run] FAILED", file=sys.stderr)
            return run
        dots += 1
        print(f"  [run] running{'.' * (dots % 4)}   ", end="\r")
        time.sleep(_POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_TABLE_COLS = [
    ("label", "LABEL", 24),
    ("lang", "LANG", 5),
    ("dataset", "DATASET", 18),
    ("status", "STATUS", 9),
    ("wer_mean", "WER", 7),
    ("cer_mean", "CER", 7),
    ("rtfx_mean", "RTFx", 6),
    ("rtfx_p95", "RTFx-p95", 9),
    ("vram_peak", "VRAM-MB", 8),
    ("wall_s", "WALL-S", 7),
    ("wer_ci", "WER-CI", 14),
]


def _fmt(val: Any, key: str) -> str:
    if val is None:
        return "—"
    if key in ("wer_mean", "cer_mean"):
        return f"{val:.4f}"
    if key in ("rtfx_mean", "rtfx_p95"):
        return f"{val:.2f}"
    if key == "vram_peak":
        return f"{val:.0f}"
    if key == "wall_s":
        return f"{val:.1f}"
    return str(val)


def build_row(run_cfg: dict, run_data: dict) -> dict:
    agg = run_data.get("aggregate") or {}
    wer_lo = agg.get("wer_ci_lower")
    wer_hi = agg.get("wer_ci_upper")
    ci = f"[{wer_lo:.4f},{wer_hi:.4f}]" if wer_lo is not None else "—"
    return {
        "label": run_cfg.get("label", ""),
        "lang": run_cfg["lang"],
        "dataset": run_cfg["dataset"],
        "status": run_data.get("status", "?"),
        "wer_mean": agg.get("wer_mean"),
        "cer_mean": agg.get("cer_mean"),
        "rtfx_mean": agg.get("rtfx_mean"),
        "rtfx_p95": agg.get("rtfx_p95"),
        "vram_peak": agg.get("vram_peak_mb"),
        "wall_s": agg.get("wall_time_s"),
        "wer_ci": ci,
        "run_id": run_data.get("run_id", ""),
        "params": json.dumps(run_cfg.get("params") or {}),
    }


def print_table(rows: list[dict]) -> None:
    header = "  ".join(h.ljust(w) for _, h, w in _TABLE_COLS)
    sep = "  ".join("-" * w for _, _, w in _TABLE_COLS)
    print()
    print(header)
    print(sep)
    for row in rows:
        line = "  ".join(_fmt(row[k], k).ljust(w) for k, _, w in _TABLE_COLS)
        print(line)
    print()


def write_csv(rows: list[dict], path: str) -> None:
    fields = [k for k, _, _ in _TABLE_COLS] + ["run_id", "params"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = {k: (_fmt(row[k], k) if row[k] is not None else "") for k in fields}
            writer.writerow(flat)
    print(f"CSV saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="ASRbench matrix runner")
    parser.add_argument("config", help="Path to matrix JSON config file")
    parser.add_argument("--csv", metavar="FILE", help="Export results to CSV")
    parser.add_argument("--dry-run", action="store_true", help="Print plan, don't run")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"Config not found: {cfg_path}")

    cfg: dict = json.loads(cfg_path.read_text(encoding="utf-8"))
    base_url: str = cfg.get("server", {}).get("base_url", "http://127.0.0.1:8765")
    model_cfg: dict = cfg["model"]
    runs_cfg: list[dict] = cfg["runs"]

    print(f"\n{'=' * 60}")
    print("  ASRbench Matrix Runner")
    print(f"  Config: {cfg_path.name}  |  Runs: {len(runs_cfg)}")
    print(f"  Model:  {model_cfg['name']} ({model_cfg['backend']})")
    print(f"{'=' * 60}\n")

    if args.dry_run:
        print("DRY RUN — plan:\n")
        for i, r in enumerate(runs_cfg, 1):
            dur = f"{int(r.get('max_duration_s', 0) / 60)}m" if r.get("max_duration_s") else "full"
            params_str = json.dumps(r.get("params") or {})
            print(f"  {i:2}. [{r.get('label', '?')}]")
            print(
                f"      lang={r['lang']}  dataset={r['dataset']}  "
                f"split={r.get('split', 'test')}  dur={dur}"
            )
            print(f"      params={params_str}")
        print()
        return

    # Server
    print("[1/4] Server check")
    if not ensure_server(base_url):
        sys.exit(1)

    with httpx.Client(base_url=base_url, timeout=60) as client:
        # Model
        print("\n[2/4] Model registration")
        model_id = register_model(client, model_cfg)
        load_model(client, model_id)

        # Datasets (deduplicate: same source+lang+split+duration → one fetch)
        print("\n[3/4] Dataset prefetch")
        ds_cache: dict[str, str] = {}  # key → dataset_id
        for run_cfg in runs_cfg:
            key = (
                f"{run_cfg['dataset']}/{run_cfg['lang']}/"
                f"{run_cfg.get('split', 'test')}/{run_cfg.get('max_duration_s', '')}"
            )
            if key not in ds_cache:
                print(f"  fetching {run_cfg['dataset']} [{run_cfg['lang']}]…")
                dataset_id = fetch_dataset(client, run_cfg)
                wait_dataset(client, dataset_id)
                ds_cache[key] = dataset_id
            else:
                print(f"  reusing {run_cfg['dataset']} [{run_cfg['lang']}] (cached)")

        # Runs
        print(f"\n[4/4] Benchmark runs  ({len(runs_cfg)} total)")
        result_rows: list[dict] = []

        for i, run_cfg in enumerate(runs_cfg, 1):
            label = run_cfg.get("label") or f"run-{i}"
            key = (
                f"{run_cfg['dataset']}/{run_cfg['lang']}/"
                f"{run_cfg.get('split', 'test')}/{run_cfg.get('max_duration_s', '')}"
            )
            dataset_id = ds_cache[key]

            print(f"\n  [{i}/{len(runs_cfg)}] {label}")
            run_id = start_run(client, model_id, dataset_id, run_cfg)
            print(f"  run_id: {run_id[:8]}…")
            run_data = wait_run(client, run_id)
            result_rows.append(build_row(run_cfg, run_data))

    # Results
    print(f"\n{'=' * 60}")
    print("  Results")
    print(f"{'=' * 60}")
    print_table(result_rows)

    if args.csv:
        write_csv(result_rows, args.csv)


if __name__ == "__main__":
    main()
