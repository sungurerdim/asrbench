# API Reference

ASRbench REST API. Default: `http://localhost:8765`.

## System

### `GET /system/health`

Returns server status and version.

**Response:** `{"status": "ok", "version": "0.1.0"}`

### `GET /system/vram`

Returns current GPU VRAM usage.

**Response:**
```json
{"used_mb": 2048.0, "total_mb": 8192.0, "pct": 25.0, "available": true}
```

## Models

### `GET /models`

List registered models. Query params: `family`, `backend` (optional filters).

### `POST /models`

Register a new model.

**Request:**
```json
{
  "family": "whisper",
  "name": "large-v3",
  "backend": "faster-whisper",
  "local_path": "/path/to/model",
  "default_params": {"beam_size": 5},
  "size_bytes": null
}
```

### `POST /models/{model_id}/load`

Load model into memory (VRAM). Returns VRAM snapshot after loading.

### `POST /models/{model_id}/unload`

Unload model from memory.

## Datasets

### `GET /datasets`

List datasets. Query params: `lang`, `source` (optional filters).

### `POST /datasets/fetch`

Download a dataset from HuggingFace.

**Request:**
```json
{
  "source": "common_voice",
  "lang": "en",
  "split": "test",
  "local_path": null
}
```

Sources: `common_voice`, `fleurs`, `yodas`, `ted_lium`, `custom`.

### `DELETE /datasets/{dataset_id}`

Delete a dataset. Query param: `delete_files=true` to also remove audio files.

## Runs

### `POST /runs/start`

Start one or more benchmark runs (with matrix expansion).

**Request:**
```json
{
  "mode": "param_compare",
  "model_id": "<uuid>",
  "dataset_id": "<uuid>",
  "lang": "en",
  "matrix": {"beam_size": [1, 2, 4, 8]},
  "baseline_run_id": null
}
```

**Response (202):**
```json
{
  "run_ids": ["<uuid>", ...],
  "baseline_run_id": "<uuid>",
  "matrix_size": 4,
  "estimated_runs": 4
}
```

### `GET /runs`

List runs. Query params: `status`, `model_id`, `lang`, `limit` (default 50, max 500).

### `GET /runs/{run_id}`

Get run details including aggregate metrics.

**Aggregate fields:** `wer_mean`, `cer_mean`, `mer_mean`, `rtfx_mean`, `rtfx_p95`, `vram_peak_mb`, `wall_time_s`, `word_count`, `wilcoxon_p`, `data_leakage_warning`, `wer_ci_lower`, `wer_ci_upper`.

### `GET /runs/{run_id}/segments`

Paginated segment-level results. Query params: `page` (default 1), `page_size` (default 100, max 1000).

### `GET /runs/compare?ids=uuid1,uuid2,...`

Compare N completed runs side-by-side with delta metrics and parameter diff.

### `POST /runs/{run_id}/cancel`

Cancel a running benchmark.

### `POST /runs/{run_id}/retry`

Clone a failed/cancelled run for re-execution.

### `DELETE /runs/{run_id}`

Delete a run and its segments/aggregates. Cannot delete running runs.

### `GET /runs/{run_id}/export?fmt=json|csv`

Export run results. Supported formats: `json`, `csv`. (`pdf` planned.)

## Parameter Optimization (IAMS)

### `POST /optimize/start`

Start an IAMS parameter optimization study.

**Request:**
```json
{
  "model_id": "<uuid>",
  "dataset_id": "<uuid>",
  "lang": "en",
  "space": {
    "parameters": {
      "beam_size": {"type": "int", "min": 1, "default": 5, "max": 20},
      "temperature": {"type": "float", "min": 0.0, "default": 0.0, "max": 1.0},
      "vad_filter": {"type": "bool", "default": true}
    }
  },
  "objective": {"type": "single", "metric": "wer", "direction": null},
  "mode": "maximum",
  "budget": {"hard_cap": 200, "convergence_eps": 0.005, "convergence_window": 3},
  "eps_min": 0.005,
  "top_k_pairs": 4,
  "multistart_candidates": 3,
  "validation_runs": 3,
  "enable_deep_ablation": false
}
```

**Objective types:**
- Single metric: `{"type": "single", "metric": "wer|cer|rtfx|vram", "direction": "minimize|maximize"}`
- Weighted: `{"type": "weighted", "weights": {"wer": 1.0, "rtfx": -0.1}}`

**Accuracy modes:**
- `fast` — Layers 1-2 only (screening + coordinate descent)
- `balanced` — Layers 1-5 (adds interaction detection + ablation)
- `maximum` — Layers 1-7 (full pipeline with validation)

**Parameter types:** `int` (min/max/step), `float` (min/max/step), `bool`, `enum` (values list).

**Response (202):**
```json
{
  "study_id": "<uuid>",
  "status": "running",
  "mode": "maximum",
  "hard_cap": 200
}
```

### `GET /optimize/{study_id}`

Get study status and results.

**Response fields:** `study_id`, `status`, `best_score`, `best_config`, `confidence` (HIGH/MEDIUM/LOW), `total_trials`, `reasoning` (per-layer explanations).

### `GET /optimize/{study_id}/trials`

Paginated trial log for audit. Query params: `page`, `page_size`, `phase` (filter by layer).

**Trial fields:** `trial_id`, `run_id`, `phase`, `config`, `score`, `score_ci_lower`, `score_ci_upper`, `reasoning`.

## WebSocket Streams

### `WS /ws/activity`

Live activity log stream. Sends JSON messages with `type: "activity"`.

### `WS /ws/vram`

Live VRAM usage stream (2-second interval). Sends `{"type": "vram", "used_mb": ..., "total_mb": ..., "pct": ...}`.

### `WS /ws/runs/{run_id}/live`

Live benchmark progress for a specific run. Message types:
- `segment` — per-segment progress with WER, RTFx, ETA
- `complete` — final aggregate metrics
- `error` — run failure
- `cancelled` — run cancellation
- `ping` — keepalive (every 30s)
