# asrbench — Technical Specification

**Version:** 1.0.0
**Date:** 2026-04-06
**Status:** Approved

---

## Overview

asrbench is a fully local, offline-first ASR benchmarking platform. It exposes a FastAPI REST + WebSocket server consumed equally by a Svelte web UI and a Typer CLI. All benchmark data is persisted in a single DuckDB file. No cloud services, no authentication, no external GPU APIs.

**Runtime environment:** Python 3.11+, conda or venv, CUDA optional (CPU fallback for all backends).

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                       asrbench                          │
├───────────────────┬─────────────────────────────────────┤
│   Typer CLI       │      Svelte Web UI (localhost)      │
│   asrbench <cmd>  │      http://localhost:8765          │
├───────────────────┴─────────────────────────────────────┤
│                FastAPI Core                             │
│         REST endpoints + WebSocket streams              │
│   /runs  /models  /datasets  /system  /ws              │
├──────────────────────────────────────────────────────────┤
│  BenchmarkEngine │ MatrixBuilder  │ CompareEngine       │
│  WEREngine       │ DatasetManager │ ActivityLogger      │
│  VRAMMonitor     │                │                     │
├──────────────────────────────────────────────────────────┤
│              Backend Plugin Interface                   │
│   FasterWhisperBackend │ WhisperCppBackend              │
│   ParakeetBackend      │ QwenASRBackend                 │
│   [plugin: TRTLLMBackend, ...]                         │
├──────────────────────────────────────────────────────────┤
│         DuckDB (benchmark.db)  +  Filesystem            │
│         ~/.asrbench/           +  user-configured paths │
└──────────────────────────────────────────────────────────┘
```

**Key architectural decisions:**
- CLI and Web UI are equal first-class interfaces — every operation reachable from both.
- FastAPI is the single source of truth for all business logic. CLI calls the same HTTP endpoints.
- WebSocket streams (`/ws/logs`, `/ws/vram`, `/ws/runs/{id}/live`) push real-time data to both UI and any connected CLI listener.
- Plugin backends loaded via Python entry points (`asrbench.backends` group).

---

## Data Model

**Storage:** Single DuckDB file at `~/.asrbench/benchmark.db`. Zstd compression enabled. JSON columns for flexible parameter storage.

### Table: `runs`
```sql
CREATE TABLE runs (
    run_id       UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id     UUID        NOT NULL REFERENCES models(model_id),
    backend      VARCHAR     NOT NULL,
    params       JSON        NOT NULL,
    dataset_id   UUID        NOT NULL REFERENCES datasets(dataset_id),
    lang         VARCHAR(8)  NOT NULL,
    mode         VARCHAR(16) NOT NULL
                             CHECK (mode IN ('model_compare', 'param_compare')),
    baseline_run_id UUID     REFERENCES runs(run_id),
    status       VARCHAR(16) NOT NULL DEFAULT 'pending'
                             CHECK (status IN ('pending','running','completed','failed','cancelled')),
    started_at   TIMESTAMPTZ,
    finished_at  TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### Table: `segments`
```sql
CREATE TABLE segments (
    segment_id   UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id       UUID        NOT NULL REFERENCES runs(run_id),
    offset_s     DOUBLE      NOT NULL,
    duration_s   DOUBLE      NOT NULL,
    ref_text     TEXT        NOT NULL,
    hyp_text     TEXT        NOT NULL,
    wer          DOUBLE,
    cer          DOUBLE,
    mer          DOUBLE,
    wil          DOUBLE,
    rtfx         DOUBLE
);
```

### Table: `aggregates`
```sql
CREATE TABLE aggregates (
    run_id               UUID    PRIMARY KEY REFERENCES runs(run_id),
    wer_mean             DOUBLE,
    cer_mean             DOUBLE,
    mer_mean             DOUBLE,
    rtfx_mean            DOUBLE,
    rtfx_p95             DOUBLE,
    vram_peak_mb         DOUBLE,
    wall_time_s          DOUBLE,
    word_count           INTEGER,
    wilcoxon_p           DOUBLE,
    data_leakage_warning BOOLEAN NOT NULL DEFAULT FALSE
);
```

### Table: `datasets`
```sql
CREATE TABLE datasets (
    dataset_id    UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    name          VARCHAR     NOT NULL,
    source        VARCHAR     NOT NULL
                              CHECK (source IN ('common_voice','fleurs','yodas','ted_lium','custom')),
    lang          VARCHAR(8)  NOT NULL,
    split         VARCHAR(16) NOT NULL
                              CHECK (split IN ('train','validation','test')),
    duration_s    DOUBLE,
    size_bytes    BIGINT,
    source_url    VARCHAR,
    local_path    VARCHAR     NOT NULL,
    downloaded_at TIMESTAMPTZ,
    checksum      VARCHAR(64),
    verified      BOOLEAN     NOT NULL DEFAULT FALSE
);
```

### Table: `models`
```sql
CREATE TABLE models (
    model_id       UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    family         VARCHAR NOT NULL,
    name           VARCHAR NOT NULL,
    backend        VARCHAR NOT NULL,
    local_path     VARCHAR NOT NULL,
    default_params JSON    NOT NULL DEFAULT '{}',
    size_bytes     BIGINT,
    registered_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### Table: `baselines`
```sql
CREATE TABLE baselines (
    model_id   UUID NOT NULL REFERENCES models(model_id),
    run_id     UUID NOT NULL REFERENCES runs(run_id),
    set_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (model_id)
);
```

---

## API Design

**Base URL:** `http://localhost:8765`
**Auth:** None (localhost only).
**CORS:** `allow_origins=["http://localhost:*"]`.

### Runs

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/runs/start` | Start a new benchmark run or matrix |
| `GET` | `/runs` | List runs (`?status=`, `?model_id=`, `?lang=`, `?limit=`) |
| `GET` | `/runs/{id}` | Run detail + aggregate |
| `DELETE` | `/runs/{id}` | Delete run and its segments |
| `POST` | `/runs/{id}/cancel` | Cancel a running benchmark |
| `POST` | `/runs/{id}/retry` | Retry a failed run with same params |
| `GET` | `/runs/{id}/segments` | Paginated segment results (`?page=`, `?page_size=`) |
| `GET` | `/runs/{id}/export` | Export run (`?fmt=json\|csv\|pdf`) |
| `GET` | `/runs/compare` | Compare N runs (`?ids=uuid1,uuid2,...`) |

**POST /runs/start — request body:**
```json
{
  "mode": "param_compare",
  "model_id": "uuid",
  "dataset_id": "uuid",
  "lang": "en",
  "matrix": {
    "beam_size": [1, 2, 4, 8],
    "compute_type": ["float16"]
  },
  "baseline_run_id": "uuid | null"
}
```

### Models

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/models` | List registered models |
| `POST` | `/models` | Register a model |
| `GET` | `/models/{id}` | Model detail + default params |
| `DELETE` | `/models/{id}` | Unregister model |
| `POST` | `/models/{id}/load` | Load model into GPU/CPU memory |
| `POST` | `/models/{id}/unload` | Release model from memory |
| `POST` | `/models/{id}/set-baseline` | Mark a run as baseline (`{"run_id": "uuid"}`) |

### Datasets

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/datasets` | List datasets with metadata |
| `POST` | `/datasets/fetch` | Download a dataset (`{"source": "common_voice", "lang": "tr", "split": "test"}`) |
| `DELETE` | `/datasets/{id}` | Remove dataset record (optionally delete files) |

### System

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/system/vram` | Current VRAM usage snapshot |
| `GET` | `/system/health` | Server health + version |

### WebSocket Streams

| Path | Payload | Description |
|------|---------|-------------|
| `WS /ws/logs` | `{"level": "INFO", "msg": "...", "ts": "..."}` | Activity log stream |
| `WS /ws/vram` | `{"used_mb": 4096, "total_mb": 8192, "pct": 50.0}` | VRAM polled every 500ms |
| `WS /ws/runs/{id}/live` | `{"segment": {...}, "progress": 0.42, "eta_s": 120}` | Per-segment live results |

---

## Security

- **Network:** FastAPI bound to `127.0.0.1` only.
- **CORS:** `allow_origins=["http://localhost:*"]`.
- **Auth:** None (single-user local tool).
- **Path sanitization:** All user-supplied paths resolved with `pathlib.Path.resolve()` and validated against configured base directories. Prevents path traversal.
- **No secrets:** No API keys, tokens, or credentials stored or transmitted.

---

## Scalability

asrbench targets latency and resource efficiency for single-user local use.

- DuckDB handles 10,000+ run history with sub-100ms analytical queries (columnar + Zstd).
- Benchmark runs are sequential by design — GPU is a shared resource.
- WebSocket fans out to all connected clients via FastAPI background tasks.
- Audio cache: preprocessed audio stored on disk; re-runs skip re-preprocessing if checksum matches.
- Model memory: explicit load/unload only — one model loaded at a time unless VRAM permits multiple.
