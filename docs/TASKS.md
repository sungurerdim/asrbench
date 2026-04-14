# asrbench — Tasks

**Version:** 1.0.0
**Date:** 2026-04-06
**Status:** Approved

---

## Milestones

| # | Milestone | Tasks | Go/No-Go Criteria |
|---|-----------|-------|-------------------|
| MS1 | Foundation | T1–T4 | `asrbench serve` starts; `GET /system/health` responds `200` |
| MS2 | WER + First Backend | T5–T8 | Single faster-whisper run completes; WER/CER/RTFx values correct vs. known reference |
| MS3 | Datasets + All Backends | T9–T13 | All 4 backends transcribe; Common Voice TR downloads and passes checksum |
| MS4 | Matrix + Compare | T14–T16 | 4-run param matrix completes via CLI; `GET /runs/compare` returns delta fields |
| MS5 | Web UI + Live Streams | T17–T19 | Full dashboard functional; live log/VRAM/progress streams update in real time |
| MS6 | Complete | T20 | JSON/CSV/PDF export works; all unit + integration tests pass |

---

## Tasks

### T1 — Project Scaffold

**Milestone:** MS1
**Estimate:** 2–3 h
**Depends on:** —

Set up the project skeleton.

- Initialize `pyproject.toml` with all pinned dependencies (see ImplGuide stack table)
- Create `asrbench/` package with empty `__init__.py` files for all sub-packages
- Register entry point: `asrbench = "asrbench.cli.app:app"` and `asrbench.backends` group
- Create `tests/unit/` and `tests/integration/` directories with `conftest.py`
- Verify: `pip install -e .` succeeds; `asrbench --help` prints usage

---

### T2 — Config Loader

**Milestone:** MS1
**Estimate:** 1–2 h
**Depends on:** T1

Implement `asrbench/config.py`.

- Parse `~/.asrbench/config.toml` using `tomllib` (Python 3.11) or `tomli` (<3.11)
- Create file with defaults if absent (see ImplGuide config.toml)
- Expand all paths via `pathlib.Path.expanduser().resolve()`
- Expose `get_config() -> Config` singleton; cached after first call
- Verify: unit test confirms default values when no config file exists

---

### T3 — DuckDB Init

**Milestone:** MS1
**Estimate:** 2–3 h
**Depends on:** T2

Implement `asrbench/db.py`.

- Open/create `benchmark.db` at path from config
- Execute all `CREATE TABLE IF NOT EXISTS` + `CREATE INDEX IF NOT EXISTS` statements from ImplGuide DDL
- Expose `get_conn() -> duckdb.DuckDBPyConnection` (thread-safe via connection pool)
- Verify: schema creation is idempotent (run twice, no error, no duplicate tables)

---

### T4 — FastAPI App Factory

**Milestone:** MS1
**Estimate:** 2–3 h
**Depends on:** T3

Implement `asrbench/main.py`.

- Create FastAPI app with lifespan: DB init (T3) + NVML init (T6 stubs)
- Mount `asrbench/static/` at `/` for Svelte UI (placeholder index.html for now)
- Register routers: `/runs`, `/models`, `/datasets`, `/system`, `/ws`
- Bind to `127.0.0.1:{config.server.port}` via uvicorn
- `GET /system/health` returns `{"status": "ok", "version": "1.0.0"}`
- CORS: `allow_origins=["http://localhost:*"]`
- Verify: `asrbench serve` starts; `curl http://localhost:8765/system/health` returns 200

---

### T5 — WEREngine

**Milestone:** MS2
**Estimate:** 3–4 h
**Depends on:** T1

Implement `asrbench/engine/wer.py` (full class from ImplGuide).

- EN normalization: `EnglishTextNormalizer → lowercase`
- TR normalization: `BasicTextNormalizer → trnorm → lowercase`; graceful fallback if trnorm absent
- `compute()`: WER + CER + MER + WIL via jiwer `process_words` / `process_characters`
- Wilcoxon p-value when `len(refs) >= 100` (scipy.stats.wilcoxon)
- Data leakage detection for Whisper × LibriSpeech/FLEURS pairs
- Unit tests: symmetric normalization; empty ref (hallucination); mixed-language edge cases

---

### T6 — VRAMMonitor

**Milestone:** MS2
**Estimate:** 1–2 h
**Depends on:** T1

Implement `asrbench/engine/vram.py` (full class from ImplGuide).

- pynvml init; GPU handle for device 0
- `snapshot()` — never raises; returns `available=False` on any pynvml failure
- `warn_threshold_pct()` — returns True if usage ≥ configured threshold (default 85%)
- Unit test: mock pynvml unavailable → `VRAMSnapshot(available=False)` returned

---

### T7 — BaseBackend + FasterWhisperBackend

**Milestone:** MS2
**Estimate:** 4–6 h
**Depends on:** T1

Implement `asrbench/backends/base.py` and `asrbench/backends/faster_whisper.py`.

- `BaseBackend` ABC: `default_params()`, `load()`, `unload()`, `transcribe()` (see ImplGuide)
- `FasterWhisperBackend`:
  - `family = "whisper"`, `name = "faster-whisper"`
  - `default_params()`: `{"beam_size": 5, "compute_type": "float16", "language": null}`
  - `load()`: `WhisperModel(model_path, device="auto", compute_type=params["compute_type"])`
  - `transcribe()`: iterate segments; RTFx = `audio_duration_s / wall_time_s`; return `list[Segment]`
  - `unload()`: delete model reference; call `gc.collect()` + CUDA cache clear
- `GET /models/{id}` returns `default_params` from registered model record
- Verify: load `faster-whisper-tiny` on a 10s WAV; WER computed against known transcript

---

### T8 — BenchmarkEngine

**Milestone:** MS2
**Estimate:** 4–6 h
**Depends on:** T5, T6, T7

Implement `asrbench/engine/benchmark.py`.

- `run_single(run_id, model, dataset, params, lang)`:
  1. Set `runs.status = 'running'`, `started_at = now()`
  2. Load audio from dataset (via AudioCache — T9 stub: load directly for now)
  3. Call `backend.transcribe(audio, lang, params)`
  4. For each segment: call `WEREngine.compute()` for per-segment metrics
  5. INSERT all segments in one DuckDB transaction
  6. Compute aggregates (mean WER/CER/MER, RTFx mean+p95, peak VRAM, wall_time)
  7. INSERT into `aggregates`; set `runs.status = 'completed'`, `finished_at = now()`
  8. On exception: set `runs.status = 'failed'`; re-raise
- Cancellation: check `asyncio.Event` per run_id between segments
- Verify: complete run inserts rows in `segments` + `aggregates`; status transitions correct

---

### T9 — MatrixBuilder

**Milestone:** MS3
**Estimate:** 2–3 h
**Depends on:** T8

Implement `asrbench/engine/matrix.py` (full class from ImplGuide).

- `build_matrix(matrix, default_params, mode)` — cartesian product; baseline-first
- `mode = "param_compare"`: baseline = `{default_params ∪ {k: first_value for k in matrix}}`
- `mode = "model_compare"`: baseline = first model's run (handled by caller, builder just orders)
- Validate: empty matrix → ValueError; empty value list → ValueError
- Unit tests: 2×2 matrix → 4 runs, baseline first; single-value matrix → 1 run marked baseline

---

### T10 — /runs API

**Milestone:** MS3
**Estimate:** 5–7 h
**Depends on:** T8, T9

Implement `asrbench/api/runs.py`.

| Endpoint | Implementation Notes |
|----------|---------------------|
| `POST /runs/start` | Validate model/dataset exist; call MatrixBuilder; enqueue runs via asyncio.Queue; return run_ids |
| `GET /runs` | Filter by status/model_id/lang; paginate by limit (default 50) |
| `GET /runs/{id}` | Join runs + aggregates; return combined detail |
| `DELETE /runs/{id}` | Cascade delete segments + aggregate; reject if status = 'running' |
| `POST /runs/{id}/cancel` | Set cancel Event for run_id; set status = 'cancelled' |
| `POST /runs/{id}/retry` | Clone run record with status = 'pending'; re-enqueue same params |
| `GET /runs/{id}/segments` | Paginate segments (page + page_size, default 50/page) |
| `GET /runs/{id}/export` | `fmt=json`: full run JSON; `fmt=csv`: segments CSV; `fmt=pdf`: rendered HTML → PDF via `weasyprint` |
| `GET /runs/compare` | Accept `?ids=` CSV; return CompareEngine.compare(run_ids) |

- All 404/409/422/507 errors from ImplGuide API contracts
- Integration test: start → cancel → retry lifecycle

---

### T11 — /models API

**Milestone:** MS3
**Estimate:** 3–4 h
**Depends on:** T7

Implement `asrbench/api/models.py`.

| Endpoint | Implementation Notes |
|----------|---------------------|
| `GET /models` | List all registered models |
| `POST /models` | Validate `local_path` exists; insert into `models` table |
| `GET /models/{id}` | Return model record + default_params |
| `DELETE /models/{id}` | Reject if model currently loaded |
| `POST /models/{id}/load` | Call backend.load(); VRAM check first; 409 if another model loaded |
| `POST /models/{id}/unload` | Call backend.unload(); no-op if not loaded |
| `POST /models/{id}/set-baseline` | Validate run belongs to model; upsert into `baselines` |

- Path validation: `local_path` must be within allowed dirs (from config or exist as absolute path)
- VRAM estimate per family (whisper-tiny ≈ 300MB, whisper-large ≈ 3000MB) — warn before OOM

---

### T12 — DatasetManager

**Milestone:** MS3
**Estimate:** 4–5 h
**Depends on:** T3, T14 (ActivityLogger stub OK)

Implement `asrbench/data/dataset_manager.py`.

- `fetch(source, lang, split)`:
  1. Load via HuggingFace `datasets.load_dataset(source_name, lang, split=split)`
  2. Stream progress to ActivityLogger
  3. Save audio files to `config.storage.cache_dir/datasets/<name>/`
  4. Compute SHA-256 of each audio file
  5. Insert dataset record with `verified=True`, `downloaded_at=now()`
- `verify(dataset_id)`: recompute checksums; return `True` if all match
- `_resolve_hf_name(source)`: `"common_voice"` → `"mozilla-foundation/common_voice_17_0"`
- AudioCache: on dataset access, resample to 16kHz float32 mono; save as `.npy`; key = SHA-256
- Verify: Common Voice TR test split downloads; `verified=True` in DB; re-run skips download

---

### T13 — /datasets API

**Milestone:** MS3
**Estimate:** 2–3 h
**Depends on:** T12

Implement `asrbench/api/datasets.py`.

| Endpoint | Implementation Notes |
|----------|---------------------|
| `GET /datasets` | List all datasets with metadata (duration_s, size_bytes, verified) |
| `POST /datasets/fetch` | Async: start DatasetManager.fetch in background task; 202 with stream_url |
| `DELETE /datasets/{id}` | Remove DB record; optionally delete files (`?delete_files=true`) |

- Custom dataset import: `POST /datasets/fetch` with `source=custom` + `local_path` field
- Progress streamed via ActivityLogger → `/ws/logs`

---

### T14 — ActivityLogger

**Milestone:** MS4
**Estimate:** 2–3 h
**Depends on:** T4

Implement `asrbench/activity/logger.py`.

- Structured JSON log: `{"level": "INFO", "msg": "...", "ts": "ISO8601", "run_id": null | "uuid"}`
- Levels: DEBUG / INFO / WARN / ERROR — filtered by `config.server.log_level`
- Broadcast via `asyncio.Queue` → all connected `/ws/logs` WebSocket clients
- Expose `log(level, msg, run_id=None)` function; import anywhere
- Replace all `print()` / `logging.basicConfig()` calls with this logger
- Verify: `asrbench serve` + connect to `ws://localhost:8765/ws/logs`; server startup message received

---

### T15 — WebSocket Streams

**Milestone:** MS4
**Estimate:** 3–4 h
**Depends on:** T6, T14

Implement `asrbench/api/ws.py`.

| Stream | Implementation |
|--------|---------------|
| `WS /ws/logs` | Fan out ActivityLogger queue to all connected clients |
| `WS /ws/vram` | Background task: poll `VRAMMonitor.snapshot()` every 500ms; broadcast to all |
| `WS /ws/runs/{id}/live` | Per-run queue; BenchmarkEngine pushes segment + progress events; `type=complete` on done |

- Segment messages must include `segments_done: int` and `segments_total: int` alongside `progress` and `eta_s`
- `segments_total` determined before run starts (total segments in dataset); sent in every message so clients can render a counter and progress bar without computing it themselves
- Disconnect handling: remove client from broadcast set on WebSocket close
- `/ws/vram` sends `{"used_mb": ..., "total_mb": ..., "pct": ..., "available": ...}`
- Verify: integration test connects to all 3 streams; segment messages include `segments_done`/`segments_total`

---

### T16 — Typer CLI

**Milestone:** MS4
**Estimate:** 4–5 h
**Depends on:** T10, T11, T13, T15

Implement `asrbench/cli/`.

All commands call FastAPI via `httpx.Client(base_url=config.server.url)`. Auto-start server if not running.

| Command | Behavior |
|---------|----------|
| `asrbench serve` | Start uvicorn; open browser to dashboard |
| `asrbench run --config <yaml>` | POST /runs/start; subscribe to `/ws/runs/{id}/live`; render `rich` progress bar (segments done/total + %; ETA) |
| `asrbench run --model-id <id> --dataset-id <id> --lang en` | Inline run without YAML |
| `asrbench models list` | Print registered models table |
| `asrbench models register --path <path> --backend faster-whisper` | POST /models |
| `asrbench models load <id>` | POST /models/{id}/load |
| `asrbench datasets list` | Print datasets table |
| `asrbench datasets fetch --source common_voice --lang tr --split test` | POST /datasets/fetch; tail /ws/logs |
| `asrbench runs list` | Print recent runs table |
| `asrbench runs compare <id1> <id2>` | GET /runs/compare; print delta table |
| `asrbench runs export <id> --fmt csv` | GET /runs/{id}/export; save file |

- YAML config schema: mirrors `POST /runs/start` body
- Error output: stderr; exit code 1 on API error
- Verify: `asrbench run --config bench.yaml` completes a faster-whisper run without browser

---

### T17 — WhisperCppBackend

**Milestone:** MS5
**Estimate:** 3–4 h
**Depends on:** T7

Implement `asrbench/backends/whisper_cpp.py`.

- Adapter for `pywhispercpp` (wraps whisper.cpp C++ binding)
- `family = "whisper"`, `name = "whisper-cpp"`
- `default_params()`: `{"n_threads": 4, "language": null, "translate": false}`
- Convert float32 array → format accepted by pywhispercpp
- Map segment timestamps to `Segment` dataclass
- Register via entry points in `pyproject.toml`

---

### T18 — ParakeetBackend + QwenASRBackend

**Milestone:** MS5
**Estimate:** 4–6 h
**Depends on:** T7

Implement `asrbench/backends/parakeet.py` and `asrbench/backends/qwen_asr.py`.

**ParakeetBackend** (NeMo):
- `family = "parakeet"`, `name = "parakeet"`
- Load via `nemo.collections.asr.models.ASRModel.restore_from(model_path)`
- `default_params()`: `{"batch_size": 1}`

**QwenASRBackend** (Transformers):
- `family = "qwen"`, `name = "qwen-asr"`
- Load via `transformers.AutoModelForSpeechSeq2Seq` + `AutoProcessor`
- `default_params()`: `{"num_beams": 1, "language": null}`

Both: graceful `ImportError` with install instructions; register via entry points.

---

### T19 — Svelte Web UI

**Milestone:** MS5
**Estimate:** 10–15 h
**Depends on:** T15, T16

Build `ui/` and compile to `asrbench/static/`.

**Pages:**

| Page | Components | Data Sources |
|------|-----------|-------------|
| Dashboard | Run history table, VRAM bar, activity log | `GET /runs`, `WS /ws/vram`, `WS /ws/logs` |
| Run Detail | Progress bar (segments done/total + %), segment table, aggregate stats, export buttons | `GET /runs/{id}`, `GET /runs/{id}/segments`, `WS /ws/runs/{id}/live` |
| Compare | Side-by-side table, delta highlighting, Wilcoxon badge | `GET /runs/compare` |
| Models | Registered models, load/unload controls, baseline marker | `GET /models`, `POST /models/{id}/load` |
| Datasets | Dataset list, download progress, custom import | `GET /datasets`, `POST /datasets/fetch`, `WS /ws/logs` |

**MatrixBuilder component:** Drag-add params → value lists → preview matrix size → POST /runs/start.

**Live run view:** Progress bar showing `segments_done / segments_total` and percentage, updated on every WS segment message. ETA derived from `eta_s` field. Segments stream into the table in real time below the bar.

Build pipeline: `npm run build` outputs to `asrbench/static/`; included in Python package via `package_data`.

---

### T20 — Export + Integration Tests

**Milestone:** MS6
**Estimate:** 4–6 h
**Depends on:** T10, T19

**Export (GET /runs/{id}/export):**
- `fmt=json`: full run JSON (run record + aggregate + all segments)
- `fmt=csv`: segments as CSV with header row
- `fmt=pdf`: HTML template → PDF via `weasyprint`; include run params, aggregate table, segment sample

**Integration test suite** (all tests use real DuckDB + real localhost FastAPI):

| Test | Verifies |
|------|---------|
| `test_full_run_lifecycle` | start → running → complete; segments + aggregate in DB |
| `test_cancel_and_retry` | cancel mid-run; status = cancelled; retry → new run_id; completes |
| `test_matrix_run` | 4-run matrix; baseline_run_id set; all 4 complete |
| `test_compare_returns_deltas` | 2 completed runs; compare returns delta_wer/cer/rtfx |
| `test_dataset_fetch_and_verify` | fetch common_voice TR test; verified=True in DB |
| `test_ws_live_stream` | connect before run start; receive ≥1 segment message with `segments_done`/`segments_total` fields; receive complete message |
| `test_export_all_formats` | JSON/CSV/PDF all return 200 with correct content-type |

---

## Dependency Graph

```
T1 → T2 → T3 → T4
               T4 → T14 → T15 → T16
T1 → T5 → T8 → T9 → T10 → T16 → T19 → T20
T1 → T6 ↗          ↗
T1 → T7 ↗    T11 ↗
             T12 → T13 → T16
T7 → T17 → T19
T7 → T18 → T19
```

**Critical Path (longest dependent chain):**

`T1 → T2 → T3 → T4 → T14 → T15 → T16 → T19 → T20`
`T1 → T5 → T8 → T9 → T10 → T16 → T19 → T20`

Both paths converge at T16 (CLI) and T19 (UI). The BenchmarkEngine path (T5→T8→T9→T10) is typically the bottleneck due to backend integration complexity.

---

## Risk Register

| # | Risk | Probability | Impact | Category | Mitigation | Contingency |
|---|------|-------------|--------|----------|-----------|-------------|
| R1 | Scope creep: features added before MS4 completes | High | High | Scope | MS1–MS3 must pass go/no-go before MS4 begins; each PR references a task | Revert to task list; defer addition to backlog |
| R2 | faster-whisper API change (v2.x) | Medium | High | Technical | Pin `faster-whisper>=1.1,<2.0`; integration test in CI against pinned version | Adapter isolation — only `faster_whisper.py` needs update |
| R3 | NeMo / Qwen API instability (T18) | Medium | Medium | Technical | T18 is isolated in its own adapter; other backends unaffected | Skip T18 for MS3 go/no-go; ship as optional plugin post-MS3 |
| R4 | pynvml unavailable (Linux NVML driver, WSL) | Low | Medium | Technical | VRAMMonitor graceful fallback; VRAM bar shows "N/A" in UI | No impact on benchmark correctness; VRAM fields null in aggregates |
| R5 | weasyprint PDF rendering failures | Low | Low | Technical | PDF export is optional format; JSON/CSV unaffected | Remove PDF from T20 scope; ship as follow-up |
| R6 | HuggingFace datasets library rate limits / network errors | Medium | Medium | External | Retry with exponential backoff (3 retries); stream download progress to UI | Cache partial downloads; resume on retry |
| R7 | Svelte UI complexity exceeds estimate (T19) | Medium | High | Scope | T19 not started until T16 (CLI) is validated; all logic lives in FastAPI | Ship minimal UI (run table + live log); defer charts to post-MS5 |

---

## Definition of Done

A task is **done** when:
1. All listed sub-items implemented
2. No `TODO` or `FIXME` left in touched files
3. Relevant tests pass (unit or integration as specified)
4. `ruff check .` passes (zero errors)
5. `mypy asrbench/` passes (zero errors on modified files)
6. Milestone go/no-go criteria met (for final task in each milestone)
