"""optimize_matrix.py — IAMS optimizer tabanlı 2-aşamalı toplu parametre arama.

Her (model, lang, dataset, condition) kombinasyonu için asrbench'in
7-katmanlı IAMS optimizer'ını çalıştırır.

  L1  Screening      → hangi parametreler WER'i gerçekten etkiliyor? (bağımsız)
  L2  Local descent  → hassas parametrelerde koordinat inişi (birlikte)
  L3  Pairwise grid  → parametre çiftleri etkileşimleri (ikili kombinasyon)
  L4  Multi-start    → lokal minimumdan kaçış (global kombinasyon)
  L5  Ablation       → toksik parametre tespiti (leave-k-out)
  L6  Refinement     → ince ayar
  L7  Validation     → confidence + CV ile sertifikasyon

2-aşamalı varsayılan akış (successive-halving tabanlı):
  Stage 1  kısa dataset (15 dk) + gevşek epsilon (0.01) → kaba tarama
  Stage 2  uzun dataset (60 dk) + sıkı epsilon (0.005) → warm-start ile
           sadece sensitive parametreleri ve rafinasyonu test eder
           (prior_study_id = Stage 1 study_id). Sensitivity metadata
           taşınır; ham skorlar context mismatch nedeniyle cache'e
           yüklenmez — Stage 2 her config'i yeni datasetinde yeniden ölçer.

Kullanım:
    python optimize_matrix.py optimize_matrix.json
    python optimize_matrix.py optimize_matrix.json --csv results.csv
    python optimize_matrix.py optimize_matrix.json --dry-run
    python optimize_matrix.py optimize_matrix.json --single-stage
    python optimize_matrix.py optimize_matrix.json \
        --stage1-duration 900 --stage1-budget 120 --stage1-epsilon 0.01 \
        --stage2-duration 3600 --stage2-budget 80 --stage2-epsilon 0.005
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

_POLL_INTERVAL = 10  # saniye — optimizer trialler uzun sürer

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _get(client: httpx.Client, path: str) -> dict:
    resp = client.get(path)
    resp.raise_for_status()
    return resp.json()


def _post(client: httpx.Client, path: str, body: dict) -> dict:
    resp = client.post(path, json=body)
    resp.raise_for_status()
    return resp.json()


def _load_yaml(path: Path) -> dict:
    """Load YAML without adding PyYAML as a hard dependency."""
    try:
        import yaml  # type: ignore[import-untyped]

        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except ImportError:
        pass
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        pass
    sys.exit(
        f"[ERROR] Cannot read {path}.\n"
        "  Install PyYAML:  pip install pyyaml\n"
        "  or convert the space file to JSON."
    )


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def ensure_server(base_url: str) -> bool:
    try:
        with httpx.Client(base_url=base_url, timeout=3) as c:
            c.get("/system/health").raise_for_status()
        print("  [server] already running")
        return True
    except Exception:
        pass

    print("  [server] not running -- starting in a new window...")
    subprocess.Popen(
        [sys.executable, "-m", "asrbench.cli.app", "serve"],
        creationflags=subprocess.CREATE_NEW_CONSOLE,
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
# Model
# ---------------------------------------------------------------------------


def register_model(client: httpx.Client, model_cfg: dict) -> str:
    name = model_cfg["name"]
    backend = model_cfg["backend"]
    family = model_cfg.get("family", "whisper")
    local_path = model_cfg.get("local_path", name)

    for m in _get(client, "/models"):
        if m["name"] == name and m["backend"] == backend:
            print(f"  [model] reusing {m['model_id'][:8]}…  ({name})")
            return m["model_id"]

    data = _post(
        client,
        "/models",
        {
            "family": family,
            "name": name,
            "backend": backend,
            "local_path": local_path,
        },
    )
    model_id = data["model_id"]
    print(f"  [model] registered {model_id[:8]}…  ({name})")
    return model_id


def load_model(client: httpx.Client, model_id: str) -> None:
    client.post(f"/models/{model_id}/load").raise_for_status()
    print("  [model] loaded into memory")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def fetch_dataset(client: httpx.Client, study_cfg: dict) -> str:
    body: dict[str, Any] = {
        "source": study_cfg["dataset"],
        "lang": study_cfg["lang"],
        "split": study_cfg.get("split", "test"),
    }
    if study_cfg.get("max_duration_s"):
        body["max_duration_s"] = study_cfg["max_duration_s"]
    return _post(client, "/datasets/fetch", body)["dataset_id"]


def wait_dataset(client: httpx.Client, dataset_id: str) -> None:
    while True:
        ds = _get(client, f"/datasets/{dataset_id}")
        if ds.get("verified"):
            print(f"  [dataset] ready: {ds['name']}")
            return
        print(f"  [dataset] loading {ds['name']}…")
        time.sleep(5)


def prefetch_all_datasets(client: httpx.Client, studies_cfg: list[dict]) -> dict[str, str]:
    """Fetch all unique datasets once; return key → dataset_id map."""
    ds_cache: dict[str, str] = {}
    for study_cfg in studies_cfg:
        key = (
            f"{study_cfg['dataset']}/{study_cfg['lang']}/"
            f"{study_cfg.get('split', 'test')}/{study_cfg.get('max_duration_s', '')}"
        )
        if key in ds_cache:
            label = f"{study_cfg['dataset']} [{study_cfg['lang']}]"
            print(f"  reusing {label} (cached)")
            continue
        print(f"  fetching {study_cfg['dataset']} [{study_cfg['lang']}]…")
        dataset_id = fetch_dataset(client, study_cfg)
        wait_dataset(client, dataset_id)
        ds_cache[key] = dataset_id
    return ds_cache


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def build_optimize_request(
    model_id: str,
    dataset_id: str,
    study_cfg: dict,
    space_dict: dict,
    opt_cfg: dict,
    prior_study_id: str | None,
) -> dict:
    objective_type = opt_cfg.get("objective", "wer")
    if objective_type in ("wer", "cer", "rtfx", "vram"):
        objective: dict[str, Any] = {"type": "single", "metric": objective_type}
    elif objective_type == "weighted":
        objective = {"type": "weighted", "weights": opt_cfg.get("weights", {"wer": 1.0})}
    else:
        objective = {"type": "single", "metric": "wer"}

    return {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "lang": study_cfg["lang"],
        "space": space_dict,
        "objective": objective,
        "mode": opt_cfg.get("mode", "maximum"),
        "budget": {
            "hard_cap": opt_cfg.get("budget", 150),
            "convergence_eps": opt_cfg.get("epsilon", 0.005),
            "convergence_window": opt_cfg.get("convergence_window", 3),
        },
        "eps_min": opt_cfg.get("epsilon", 0.005),
        "top_k_pairs": opt_cfg.get("top_k_pairs", 4),
        "multistart_candidates": opt_cfg.get("multistart_candidates", 3),
        "validation_runs": opt_cfg.get("validation_runs", 3),
        "enable_deep_ablation": opt_cfg.get("deep_ablation", False),
        "prior_study_id": prior_study_id,
    }


def _kill_server(base_url: str) -> None:
    """
    Kill the server process holding the configured port.

    Uses netstat to find the PID listening on the port, then terminates it.
    The background task (zombie model load) dies with the process.
    """
    import re
    import urllib.parse

    port = urllib.parse.urlparse(base_url).port or 8765
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                pid = parts[-1]
                if pid.isdigit():
                    subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
                    print(f"  [reset] killed server PID {pid}")
                    return
        # Fallback: try TCP_WAIT / ESTABLISHED lines too
        for line in result.stdout.splitlines():
            m = re.search(rf":{port}\s", line)
            if m:
                parts = line.split()
                pid = parts[-1]
                if pid.isdigit() and pid != "0":
                    subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
                    print(f"  [reset] killed server PID {pid}")
                    return
    except Exception as exc:
        print(f"  [reset] could not kill server: {exc}")


def _restart_server(base_url: str) -> bool:
    """Kill existing server, start fresh, wait for readiness."""
    _kill_server(base_url)
    time.sleep(2)
    return ensure_server(base_url)


def _cancel_stuck_studies(client: httpx.Client) -> None:
    """Cancel running studies left in DB from a previous interrupted run."""
    try:
        r = client.get("/optimize/", params={"status": "running"})
        if r.status_code != 200:
            return
        for s in r.json():
            sid = s.get("study_id", "")
            if not sid:
                continue
            cr = client.post(f"/optimize/{sid}/cancel")
            if cr.status_code == 200:
                print(f"  [reset] cancelled stuck study {sid[:8]}...")
    except Exception as exc:
        print(f"  [reset] warning: {exc}")


def start_study(client: httpx.Client, request_body: dict, base_url: str = "") -> str:
    resp = client.post("/optimize/start", json=request_body)
    if resp.status_code == 409:
        print(
            "  [409] Stuck study blocking start -- restarting server"
            " + cancelling stuck DB entries..."
        )
        if base_url and _restart_server(base_url):
            with httpx.Client(base_url=base_url, timeout=60) as fresh:
                _cancel_stuck_studies(fresh)  # clear DB entries on fresh server
                resp = fresh.post("/optimize/start", json=request_body)
        else:
            resp = client.post("/optimize/start", json=request_body)
    resp.raise_for_status()
    return resp.json()["study_id"]


_PHASE_LABELS: dict[str, str] = {
    "screening": "L1 Screening",
    "local_1d": "L2 Local descent",
    "pairwise_grid": "L3 Pairwise grid",
    "multistart": "L4 Multi-start",
    "ablation": "L5 Ablation",
    "refinement": "L6 Refinement",
    "validation": "L7 Validation",
}


def _fmt_phase(phase: str) -> str:
    return _PHASE_LABELS.get(phase, phase)


def _fetch_trial_snapshot(client: httpx.Client, study_id: str) -> tuple[int, str, float | None]:
    """
    Query the live trials table.

    Returns (count, last_phase, best_score_so_far).
    Falls back to (0, "", None) on any error.
    """
    try:
        resp = client.get(f"/optimize/{study_id}/trials", params={"page_size": 500})
        if resp.status_code != 200:
            return 0, "", None
        trials: list[dict] = resp.json()
        count = len(trials)
        last_phase = trials[-1].get("phase", "") if trials else ""
        scores = [t["score"] for t in trials if t.get("score") is not None]
        best = min(scores) if scores else None
        return count, last_phase, best
    except Exception:
        return 0, "", None


def wait_study(client: httpx.Client, study_id: str, budget: int = 150) -> dict:
    last_count = 0
    last_phase = ""
    last_best: float | None = None
    _start_time = time.time()

    while True:
        study = _get(client, f"/optimize/{study_id}")
        status = study.get("status", "?")

        if status in ("completed", "failed"):
            # Print a blank line to clear the progress bar
            print(f"\r{' ' * 80}", end="\r")
            if status == "completed":
                total = study.get("total_trials") or last_count
                best = study.get("best_score")
                best_str = f"  WER: {best:.4f}" if best is not None else ""
                conf = study.get("confidence") or ""
                conf_str = f"  conf: {conf}" if conf else ""
                print(f"  done - {total} trials{best_str}{conf_str}")
            else:
                err = study.get("error_message") or ""
                # Show first line of the error (type + message) for quick diagnosis
                first_line = err.split("\n")[0] if err else "unknown error"
                print(f"  FAILED after {last_count} trials -- {first_line}", file=sys.stderr)
            return study

        # Live trial data — refreshed every poll cycle (budget <= 500 trials)
        count, phase, best = _fetch_trial_snapshot(client, study_id)
        if count > 0:
            last_count = count
        if phase:
            last_phase = phase
        if best is not None:
            last_best = best

        if last_count == 0:
            # First trial not done yet: model load + full baseline transcription
            elapsed = int(time.time() - _start_time)
            m, s = divmod(elapsed, 60)
            print(
                f"\r  [initializing]  {m:02d}:{s:02d} elapsed"
                f"  (model load + baseline trial - may take 15-30 min)   ",
                end="",
                flush=True,
            )
        else:
            # Progress bar
            pct = min(last_count / budget, 1.0) if budget > 0 else 0.0
            bar_w = 28
            filled = int(pct * bar_w)
            bar = "#" * filled + "-" * (bar_w - filled)

            best_str = f"  WER: {last_best:.4f}" if last_best is not None else "  WER: ..."
            phase_str = f"  {_fmt_phase(last_phase)}" if last_phase else ""

            line = f"  [{bar}] {last_count}/{budget}{best_str}{phase_str}"
            print(f"\r{line:<80}", end="", flush=True)

        time.sleep(_POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_TABLE_COLS = [
    ("model", "MODEL", 16),
    ("label", "STUDY", 22),
    ("lang", "LANG", 5),
    ("dataset", "DATASET", 13),
    ("condition", "COND", 6),
    ("status", "STATUS", 9),
    ("best_score_ci", "WER* [95% CI]", 22),
    ("wilcoxon_p", "p", 8),
    ("confidence", "CONF", 8),
    ("total_trials", "TRIALS", 7),
]


def _fetch_best_run_stats(client: httpx.Client, run_id: str | None) -> dict:
    """
    Pull (wer_ci_lower, wer_ci_upper, wilcoxon_p) for a completed run.

    The optimizer's StudyResponse only exposes a scalar best_score; the
    bootstrap CI and Wilcoxon p-value live on the underlying `runs` row, which
    /runs/{id} serializes. Missing / failed / pre-CI runs return empty dict.
    """
    if not run_id:
        return {}
    try:
        resp = client.get(f"/runs/{run_id}")
        if resp.status_code != 200:
            return {}
        data = resp.json()
        return {
            "wer_ci_lower": data.get("wer_ci_lower"),
            "wer_ci_upper": data.get("wer_ci_upper"),
            "wilcoxon_p": data.get("wilcoxon_p"),
        }
    except Exception:
        return {}


def build_result_row(
    model_name: str,
    study_cfg: dict,
    study_data: dict,
    run_stats: dict | None = None,
) -> dict:
    """
    Convert a completed /optimize/{id} response + its /runs/{best_run_id} stats
    into the flat row consumed by print_table / write_csv. ``run_stats`` carries
    the bootstrap CI and Wilcoxon p-value fetched from the backing run; it is
    optional so legacy callers (dry-run, tests) still work.
    """
    best_cfg = study_data.get("best_config") or {}
    score = study_data.get("best_score")
    condition = "clean" if "clean" in study_cfg.get("label", "") else "noisy"
    run_stats = run_stats or {}
    ci_lo = run_stats.get("wer_ci_lower")
    ci_hi = run_stats.get("wer_ci_upper")
    if score is not None and ci_lo is not None and ci_hi is not None:
        # WER reported to 4 decimals so half-width comparisons against eps_min
        # (0.005) are visible; CI bounds written beside the point estimate
        # make the reader stop at a glance and NOT over-read a tight number.
        best_score_str = f"{score:.4f}"
        best_score_ci_str = f"{score:.4f} [{ci_lo:.4f}-{ci_hi:.4f}]"
    elif score is not None:
        best_score_str = f"{score:.4f}"
        best_score_ci_str = f"{score:.4f}"
    else:
        best_score_str = "-"
        best_score_ci_str = "-"
    wilcoxon_p = run_stats.get("wilcoxon_p")
    wilcoxon_str = f"{wilcoxon_p:.3f}" if wilcoxon_p is not None else "-"
    return {
        "model": model_name,
        "label": study_cfg.get("label", ""),
        "lang": study_cfg["lang"],
        "dataset": study_cfg["dataset"],
        "condition": condition,
        "status": study_data.get("status", "?"),
        "best_score": best_score_str,
        "best_score_ci": best_score_ci_str,
        "wer_ci_lower": ci_lo,
        "wer_ci_upper": ci_hi,
        "wilcoxon_p": wilcoxon_str,
        "confidence": study_data.get("confidence") or "-",
        "total_trials": str(study_data.get("total_trials") or "-"),
        "study_id": study_data.get("study_id", ""),
        "best_run_id": study_data.get("best_run_id") or "",
        "best_config": json.dumps(best_cfg),
        "space_file": study_cfg.get("space_file", ""),
    }


def print_table(rows: list[dict]) -> None:
    header = "  ".join(h.ljust(w) for _, h, w in _TABLE_COLS)
    sep = "  ".join("-" * w for _, _, w in _TABLE_COLS)
    print()
    print(header)
    print(sep)
    # Group by study label for easy model comparison
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    for _, label_rows in by_label.items():
        for row in label_rows:
            line = "  ".join(str(row.get(k, "-")).ljust(w) for k, _, w in _TABLE_COLS)
            print(line)
        print()  # blank line between study groups


def print_best_configs(rows: list[dict]) -> None:
    print("Best configs per study:")
    print("-" * 70)
    for row in rows:
        ci_lo = row.get("wer_ci_lower")
        ci_hi = row.get("wer_ci_upper")
        if ci_lo is not None and ci_hi is not None:
            half_width = (float(ci_hi) - float(ci_lo)) / 2.0
            ci_str = f"  95%CI=[{ci_lo:.4f}-{ci_hi:.4f}] (±{half_width:.4f})"
        else:
            ci_str = ""
        p_str = f"  p={row['wilcoxon_p']}" if row.get("wilcoxon_p", "-") != "-" else ""
        print(
            f"\n  [{row['model']}] {row['label']}"
            f"  WER*={row['best_score']}{ci_str}{p_str}  conf={row['confidence']}"
            f"  trials={row['total_trials']}"
        )
        try:
            cfg = json.loads(row["best_config"])
            for k, v in sorted(cfg.items()):
                print(f"    {k}: {v}")
        except (json.JSONDecodeError, TypeError):
            print(f"    {row['best_config']}")


def write_csv(rows: list[dict], path: str) -> None:
    # CSV gets the flat scalar columns (not the pretty [lo-hi] string) so
    # downstream pandas/Excel can filter by numeric CI bounds directly.
    fields = [
        "model",
        "label",
        "lang",
        "dataset",
        "condition",
        "status",
        "best_score",
        "wer_ci_lower",
        "wer_ci_upper",
        "wilcoxon_p",
        "confidence",
        "total_trials",
        "study_id",
        "best_run_id",
        "best_config",
        "space_file",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {path}")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _parse_models(cfg: dict) -> list[dict]:
    """Support both "models": [...] (multi) and "model": {...} (legacy single)."""
    if "models" in cfg:
        return cfg["models"]
    if "model" in cfg:
        return [cfg["model"]]
    sys.exit("[ERROR] Config must have a 'models' list or a 'model' object.")


def _ds_key(study_cfg: dict) -> str:
    return (
        f"{study_cfg['dataset']}/{study_cfg['lang']}/"
        f"{study_cfg.get('split', 'test')}/{study_cfg.get('max_duration_s', '')}"
    )


def _compound_label(model_name: str, study_label: str) -> str:
    """Unique label for a (model, study) pair used in prior_study resolution."""
    return f"{model_name}/{study_label}"


# ---------------------------------------------------------------------------
# Dry-run output
# ---------------------------------------------------------------------------


def _estimate_trial_seconds(study_cfg: dict, backend: str) -> float:
    """
    Rough wall-clock estimate for one IAMS trial against this study.

    Baseline assumption: each trial runs the whole PreparedDataset once at the
    current max_duration_s through the backend, with a realtime factor (RTFx)
    that depends on model + batch mode. Empirical RTFx values below are pulled
    from published faster-whisper benchmarks on a single modern consumer GPU
    (RTX 4090-class), used only as order-of-magnitude figures — real
    measurements in the aggregates table will be far more accurate once the
    first trial lands. The estimate is intentionally pessimistic (favors the
    slower sequential path) so users are not surprised by longer runs.
    """
    dataset_seconds = float(study_cfg.get("max_duration_s") or 3600)
    batch_size = int(study_cfg.get("batch_size") or 0)

    # Published RTFx ranges for faster-whisper large-v3 on a 4090:
    #   sequential  ~15-25× realtime
    #   batched 5   ~40-60× realtime
    # We invert to get seconds per second of audio, then multiply by corpus.
    if backend == "faster-whisper":
        rtfx = 45.0 if batch_size > 0 else 18.0
    elif backend == "whisper-cpp":
        rtfx = 8.0
    elif backend == "parakeet":
        rtfx = 30.0
    elif backend == "qwen-asr":
        rtfx = 6.0
    else:
        rtfx = 15.0

    return dataset_seconds / rtfx


def _format_duration(seconds: float) -> str:
    """Pretty short duration: 5h42m / 12m30s / 45s."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, s = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{s:02d}s"
    hours, m = divmod(minutes, 60)
    return f"{hours}h{m:02d}m"


def _stage_estimate_for_model(
    model_cfg: dict,
    studies_cfg: list[dict],
    budget: int,
    duration_override: int | None,
) -> float:
    """Sum likely wall-clock across all studies for one (model, stage) pass."""
    backend = model_cfg.get("backend", "faster-whisper")
    total = 0.0
    for s in studies_cfg:
        effective = dict(s)
        if duration_override is not None:
            effective["max_duration_s"] = duration_override
        per_trial = _estimate_trial_seconds(effective, backend)
        # IAMS typically converges at ~60% of the hard cap due to insensitive
        # param pruning + convergence_window; Stage 2 warm-start usually uses
        # even less, but we keep a single multiplier for simplicity.
        total += budget * per_trial * 0.6
    return total


def _dry_run(
    models: list[dict],
    studies_cfg: list[dict],
    opt_cfg: dict,
    *,
    two_stage: bool = False,
    stage1_duration: int | None = None,
    stage1_budget: int | None = None,
    stage2_duration: int | None = None,
    stage2_budget: int | None = None,
) -> None:
    total = len(models) * len(studies_cfg)
    print(f"DRY RUN - plan  ({total} total studies per stage)\n")

    total_wall_time_s = 0.0

    for model_cfg in models:
        print(f"  Model: {model_cfg['name']} ({model_cfg['backend']})")
        print(f"  Path:  {model_cfg.get('local_path', model_cfg['name'])}\n")
        for i, s in enumerate(studies_cfg, 1):
            dur_json = s.get("max_duration_s", 0)
            space_p = Path(s["space_file"])
            param_count: int | str = "?"
            if space_p.exists():
                space_dict = _load_yaml(space_p)
                param_count = len(space_dict.get("parameters", {}))
            l1 = f"{1 + 2 * int(param_count)}" if param_count != "?" else "?"
            print(f"    {i:2}. [{s.get('label', '?')}]")
            print(
                f"        lang={s['lang']}  dataset={s['dataset']}  "
                f"split={s.get('split', 'test')}  json_dur={int(dur_json / 60)}m"
            )
            print(f"        space={s['space_file']}  ({param_count} params)  L1~{l1} trials")
            prior = s.get("prior_study")
            if prior:
                print(f"        warm-start: {prior}")
        print()

        if two_stage:
            s1_total = _stage_estimate_for_model(
                model_cfg, studies_cfg, int(stage1_budget or 120), stage1_duration
            )
            s2_total = _stage_estimate_for_model(
                model_cfg, studies_cfg, int(stage2_budget or 80), stage2_duration
            )
            model_total = s1_total + s2_total
            print(
                f"  Model subtotal: S1 ~{_format_duration(s1_total)}  "
                f"+ S2 ~{_format_duration(s2_total)}  "
                f"= ~{_format_duration(model_total)}"
            )
        else:
            model_total = _stage_estimate_for_model(
                model_cfg, studies_cfg, int(opt_cfg.get("budget", 150)), None
            )
            print(f"  Model subtotal (likely): ~{_format_duration(model_total)}")
        total_wall_time_s += model_total
        print()

    print(
        f"Optimizer: mode={opt_cfg.get('mode', 'maximum')}  "
        f"objective={opt_cfg.get('objective', 'wer').upper()}"
    )
    if two_stage:
        print(
            f"Total wall-clock (likely, 2-stage): "
            f"~{_format_duration(total_wall_time_s)}  "
            f"({_format_duration(total_wall_time_s * 5 / 3)} upper cap)"
        )
    else:
        print(
            f"Total wall-clock (likely): ~{_format_duration(total_wall_time_s)}  "
            f"({_format_duration(total_wall_time_s * 5 / 3)} upper cap)"
        )
    print(
        "Note: estimates use published RTFx ranges (faster-whisper large-v3 "
        "on an RTX 4090-class GPU: ~18× sequential, ~45× batched). Real "
        "trial times land in the `aggregates` table once the first trial "
        "completes and will refine subsequent estimates."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _merge_opt_overrides(
    base: dict,
    *,
    budget: int | None,
    epsilon: float | None,
) -> dict:
    """Copy ``base`` and apply stage-level budget/epsilon overrides if set."""
    merged = dict(base)
    if budget is not None:
        merged["budget"] = budget
    if epsilon is not None:
        merged["epsilon"] = epsilon
    return merged


def _apply_duration_override(
    studies_cfg: list[dict],
    duration_override: int | None,
) -> list[dict]:
    """Return a copy of ``studies_cfg`` with ``max_duration_s`` patched to ``duration_override``."""
    if duration_override is None:
        return studies_cfg
    out = []
    for s in studies_cfg:
        c = dict(s)
        c["max_duration_s"] = duration_override
        out.append(c)
    return out


def run_stage(
    *,
    stage_label: str,
    studies_cfg: list[dict],
    models: list[dict],
    opt_cfg: dict,
    client: httpx.Client,
    ds_cache: dict[str, str],
    base_url: str,
    cfg_path: Path,
    prior_study_map: dict[tuple[str, str], str] | None,
    model_id_cache: dict[str, str],
    label_to_study_id: dict[str, str],
) -> list[dict]:
    """
    Run one optimization stage across all (model × study) combinations.

    Parameters:
        stage_label: short tag printed in logs ("S1", "S2", "single").
        studies_cfg: already has ``max_duration_s`` patched to the stage's value.
        opt_cfg:     already has ``budget``/``epsilon`` patched to the stage's values.
        prior_study_map: {(model_name, study_label) -> prior_study_id} for warm-start;
                         None for stage 1 or single-stage runs.
        model_id_cache:  shared across stages so models load + register only once.
        label_to_study_id: shared across stages so compound labels resolve
                           across stage boundaries when user specifies manual prior refs.

    Returns:
        list of result row dicts (one per (model, study) evaluation).
    """
    result_rows: list[dict] = []

    for m_idx, model_cfg in enumerate(models, 1):
        model_name = model_cfg["name"]
        cached_model_id = model_id_cache.get(model_name)
        if cached_model_id is None:
            print(f"\n[{stage_label}] Model {m_idx}/{len(models)}: {model_name}")
            model_id = register_model(client, model_cfg)
            load_model(client, model_id)
            model_id_cache[model_name] = model_id
        else:
            print(f"\n[{stage_label}] Model {m_idx}/{len(models)}: {model_name} (reusing)")
            model_id = cached_model_id

        study_count = len(studies_cfg)
        print(f"[{stage_label}] Studies for {model_name}  ({study_count} studies)\n")

        for i, study_cfg in enumerate(studies_cfg, 1):
            study_label = study_cfg.get("label") or f"study-{i}"
            compound = _compound_label(model_name, study_label)
            dataset_id = ds_cache[_ds_key(study_cfg)]

            space_path = Path(study_cfg["space_file"])
            if not space_path.exists():
                space_path = cfg_path.parent / study_cfg["space_file"]
            if not space_path.exists():
                print(
                    f"  [SKIP] space file not found: {study_cfg['space_file']}",
                    file=sys.stderr,
                )
                continue

            space_dict = _load_yaml(space_path)

            batch_size: int = int(study_cfg.get("batch_size", 0))
            if batch_size > 0:
                space_dict.setdefault("parameters", {})["batch_size"] = {
                    "type": "enum",
                    "values": [batch_size],
                    "default": batch_size,
                }

            param_count = len(space_dict.get("parameters", {}))

            # Resolve prior_study: stage-scoped map wins, then legacy raw field
            prior_study_id: str | None = None
            if prior_study_map is not None:
                prior_study_id = prior_study_map.get((model_name, study_label))
            if prior_study_id is None:
                raw_prior: str | None = study_cfg.get("prior_study")
                if raw_prior:
                    same_model_prior = _compound_label(model_name, raw_prior)
                    if same_model_prior in label_to_study_id:
                        prior_study_id = label_to_study_id[same_model_prior]
                    elif raw_prior in label_to_study_id:
                        prior_study_id = label_to_study_id[raw_prior]
                    else:
                        prior_study_id = raw_prior  # assume UUID

            batch_tag = f"  batch={batch_size}" if batch_size > 0 else "  sequential"
            duration_min = int(study_cfg.get("max_duration_s", 0) / 60)
            print(
                f"  [{stage_label}][{i}/{study_count}] [{model_name}] "
                f"{study_label}{batch_tag}  dur={duration_min}m"
            )
            print(
                f"  space: {space_path.name}  {param_count} params  L1~{1 + 2 * param_count} trials"
            )

            params_list = list(space_dict.get("parameters", {}).keys())
            params_preview = ", ".join(params_list[:10])
            if len(params_list) > 10:
                params_preview += f" ... (+{len(params_list) - 10} more)"
            print(f"  params: {params_preview}")

            if prior_study_id:
                print(f"  warm-start: {prior_study_id[:8]}...")

            req_body = build_optimize_request(
                model_id=model_id,
                dataset_id=dataset_id,
                study_cfg=study_cfg,
                space_dict=space_dict,
                opt_cfg=opt_cfg,
                prior_study_id=prior_study_id,
            )

            budget = int(opt_cfg.get("budget", 150))
            study_id = start_study(client, req_body, base_url=base_url)
            label_to_study_id[compound] = study_id
            print(f"  study_id: {study_id[:8]}...")

            study_data = wait_study(client, study_id, budget=budget)
            run_stats = _fetch_best_run_stats(client, study_data.get("best_run_id"))
            row = build_result_row(model_name, study_cfg, study_data, run_stats)
            row["stage"] = stage_label
            result_rows.append(row)

            # Inline best config summary (non-default params only)
            best_cfg: dict = study_data.get("best_config") or {}
            defaults = space_dict.get("parameters", {})
            non_default = {
                k: v
                for k, v in best_cfg.items()
                if str(v) != str(defaults.get(k, {}).get("default", ""))
            }
            if non_default:
                cfg_str = "  ".join(f"{k}={v}" for k, v in sorted(non_default.items()))
                print(f"  best (non-default): {cfg_str}")
            print()

    return result_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="ASRbench IAMS optimizer matrix")
    parser.add_argument("config", help="Path to optimize matrix JSON config")
    parser.add_argument("--csv", metavar="FILE", help="Export results to CSV")
    parser.add_argument("--dry-run", action="store_true", help="Print plan, don't run")
    parser.add_argument(
        "--single-stage",
        action="store_true",
        help=(
            "Disable 2-stage default; run every study once using the "
            "JSON's max_duration_s + optimizer.budget + optimizer.epsilon."
        ),
    )
    parser.add_argument(
        "--stage1-duration",
        type=int,
        default=900,
        help="Stage 1 dataset cap in seconds (default: 900 = 15 min).",
    )
    parser.add_argument(
        "--stage1-budget",
        type=int,
        default=120,
        help="Stage 1 trial budget per study (default: 120).",
    )
    parser.add_argument(
        "--stage1-epsilon",
        type=float,
        default=0.02,
        help=(
            "Stage 1 convergence epsilon — coarse (default: 0.02). "
            "Calibrated to the actual WER measurement noise floor at 900 s on "
            "FLEURS-tr (~750 words → SE ~1.5%%). Setting this tighter than the "
            "noise floor causes spurious sensitivity attributions during "
            "screening; 0.02 keeps the AND-gate honest. See Li et al. 2018 "
            "(Hyperband, JMLR) on resource/noise calibration and Bisani & Ney "
            "2004 on bootstrap CI width vs. corpus size."
        ),
    )
    parser.add_argument(
        "--stage2-duration",
        type=int,
        default=3600,
        help="Stage 2 dataset cap in seconds (default: 3600 = 60 min).",
    )
    parser.add_argument(
        "--stage2-budget",
        type=int,
        default=80,
        help=(
            "Stage 2 trial budget per study (default: 80). Lower than stage 1 "
            "because warm-start skips L1 screening."
        ),
    )
    parser.add_argument(
        "--stage2-epsilon",
        type=float,
        default=0.005,
        help="Stage 2 convergence epsilon — fine (default: 0.005).",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"Config not found: {cfg_path}")

    cfg: dict = json.loads(cfg_path.read_text(encoding="utf-8"))
    base_url: str = cfg.get("server", {}).get("base_url", "http://127.0.0.1:8765")
    models = _parse_models(cfg)
    opt_cfg: dict = cfg.get("optimizer", {})
    studies_cfg: list[dict] = cfg["studies"]
    total_studies = len(models) * len(studies_cfg)

    two_stage = not args.single_stage
    mode_label = "2-STAGE (S1 coarse -> S2 refine)" if two_stage else "SINGLE-STAGE"

    print(f"\n{'=' * 65}")
    print("  ASRbench IAMS Optimizer Matrix")
    print(f"  Config:  {cfg_path.name}")
    print(f"  Mode:    {mode_label}")
    print(f"  Models:  {', '.join(m['name'] for m in models)}")
    print(f"  Studies: {len(studies_cfg)} x {len(models)} models = {total_studies} per stage")
    print(
        f"  Opt:     mode={opt_cfg.get('mode', 'maximum')}  "
        f"obj={opt_cfg.get('objective', 'wer').upper()}"
    )
    if two_stage:
        print(
            f"  S1:      dur={args.stage1_duration}s  "
            f"budget={args.stage1_budget}  eps={args.stage1_epsilon}"
        )
        print(
            f"  S2:      dur={args.stage2_duration}s  "
            f"budget={args.stage2_budget}  eps={args.stage2_epsilon}  "
            f"(warm-start from S1)"
        )
    else:
        print(
            f"  Single:  dur={'(JSON)' if not opt_cfg.get('budget') else 'json'}  "
            f"budget={opt_cfg.get('budget', 150)}  eps={opt_cfg.get('epsilon', 0.005)}"
        )
    print(f"{'=' * 65}\n")

    if args.dry_run:
        _dry_run(
            models,
            studies_cfg,
            opt_cfg,
            two_stage=two_stage,
            stage1_duration=args.stage1_duration if two_stage else None,
            stage1_budget=args.stage1_budget if two_stage else None,
            stage2_duration=args.stage2_duration if two_stage else None,
            stage2_budget=args.stage2_budget if two_stage else None,
        )
        return

    # Server
    print("[1/4] Server check")
    if not ensure_server(base_url):
        sys.exit(1)

    with httpx.Client(base_url=base_url, timeout=60) as client:
        # --- Dataset prefetch (fetch each unique (dataset, lang, duration) once) ----
        print("\n[2/4] Dataset prefetch (shared across models + stages)")
        if two_stage:
            s1_studies = _apply_duration_override(studies_cfg, args.stage1_duration)
            s2_studies = _apply_duration_override(studies_cfg, args.stage2_duration)
            ds_cache = prefetch_all_datasets(client, s1_studies)
            ds_cache.update(prefetch_all_datasets(client, s2_studies))
        else:
            s1_studies = studies_cfg  # unused
            s2_studies = studies_cfg
            ds_cache = prefetch_all_datasets(client, studies_cfg)

        # --- Stage execution ----------------------------------------------------
        model_id_cache: dict[str, str] = {}
        label_to_study_id: dict[str, str] = {}

        if two_stage:
            s1_opt = _merge_opt_overrides(
                opt_cfg, budget=args.stage1_budget, epsilon=args.stage1_epsilon
            )
            s2_opt = _merge_opt_overrides(
                opt_cfg, budget=args.stage2_budget, epsilon=args.stage2_epsilon
            )

            print("\n[3/4] Stage 1 — coarse screening on short dataset")
            stage1_rows = run_stage(
                stage_label="S1",
                studies_cfg=s1_studies,
                models=models,
                opt_cfg=s1_opt,
                client=client,
                ds_cache=ds_cache,
                base_url=base_url,
                cfg_path=cfg_path,
                prior_study_map=None,
                model_id_cache=model_id_cache,
                label_to_study_id=label_to_study_id,
            )

            # Build (model, study_label) -> study_id map for completed S1 runs only.
            # Failed / cancelled studies get NO warm-start reference; Stage 2 will
            # re-run them from scratch rather than inherit broken screening metadata.
            prior_map: dict[tuple[str, str], str] = {}
            for row in stage1_rows:
                if row.get("status") == "completed" and row.get("study_id"):
                    prior_map[(row["model"], row["label"])] = row["study_id"]

            skipped = [
                (row["model"], row["label"])
                for row in stage1_rows
                if row.get("status") != "completed"
            ]
            if skipped:
                print(
                    f"\n[warn] {len(skipped)} Stage-1 studies did not complete — "
                    f"their Stage-2 runs will start cold (no warm-start)."
                )

            print("\n[4/4] Stage 2 — fine refinement on full dataset (warm-start from S1)")
            stage2_rows = run_stage(
                stage_label="S2",
                studies_cfg=s2_studies,
                models=models,
                opt_cfg=s2_opt,
                client=client,
                ds_cache=ds_cache,
                base_url=base_url,
                cfg_path=cfg_path,
                prior_study_map=prior_map,
                model_id_cache=model_id_cache,
                label_to_study_id=label_to_study_id,
            )
            result_rows = stage2_rows  # final = S2 outcomes; S1 kept in DB for audit
        else:
            print("\n[3/4] Single-stage run")
            result_rows = run_stage(
                stage_label="single",
                studies_cfg=studies_cfg,
                models=models,
                opt_cfg=opt_cfg,
                client=client,
                ds_cache=ds_cache,
                base_url=base_url,
                cfg_path=cfg_path,
                prior_study_map=None,
                model_id_cache=model_id_cache,
                label_to_study_id=label_to_study_id,
            )

    # Results
    print(f"\n{'=' * 65}")
    header_tag = "Stage 2 final" if two_stage else "Single-stage"
    print(f"  Results — {header_tag} best configs (grouped by study)")
    print(f"{'=' * 65}")
    print_table(result_rows)
    print_best_configs(result_rows)

    if args.csv:
        print()
        write_csv(result_rows, args.csv)


if __name__ == "__main__":
    main()
