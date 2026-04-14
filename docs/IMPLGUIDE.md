# asrbench — Implementation Guide

**Version:** 1.0.0
**Date:** 2026-04-06
**Status:** Approved

---

## Quick Start

```bash
# Install
pip install asrbench

# Start server + open dashboard
asrbench serve

# Or: run a benchmark from CLI only
asrbench run --config bench.yaml
```

Server starts at `http://localhost:8765`. Dashboard opens automatically in the default browser.

---

## Technology Stack

| Component | Package | Pinned Version | Purpose |
|-----------|---------|---------------|---------|
| API server | `fastapi` | `>=0.115,<0.116` | REST + WebSocket core |
| ASGI server | `uvicorn[standard]` | `>=0.34,<0.35` | Production ASGI runner |
| CLI | `typer[all]` | `>=0.15,<0.16` | CLI framework |
| Database | `duckdb` | `>=1.2,<1.3` | Analytical benchmark storage |
| Metrics | `jiwer` | `>=4.0,<4.1` | WER / CER / MER / WIL |
| Turkish norm | `trnorm` | `>=0.2,<0.3` | Turkish text normalization |
| VRAM monitor | `pynvml` | `>=11.5,<12.0` | NVIDIA GPU memory via NVML |
| Data validation | `pydantic` | `>=2.7,<3.0` | Request/response schemas |
| Audio I/O | `soundfile` | `>=0.12,<0.13` | Audio loading |
| Resampling | `soxr` | `>=0.5,<0.6` | Sample rate conversion |
| Datasets | `datasets` | `>=3.3,<4.0` | HuggingFace dataset downloads |
| HTTP client | `httpx` | `>=0.28,<0.29` | CLI → FastAPI calls |
| Config | `tomli` | `>=2.2,<3.0` (py<3.11) | TOML config parser |
| Frontend | `svelte` | `5.x` | Web dashboard (compiled, bundled) |

**Backend adapters (optional, installed on demand):**

| Backend | Package | Version |
|---------|---------|---------|
| faster-whisper | `faster-whisper` | `>=1.1,<2.0` |
| whisper.cpp | `pywhispercpp` | `>=1.3,<2.0` |
| Parakeet | `nemo_toolkit[asr]` | `>=2.2,<3.0` |
| Qwen-ASR | `transformers` | `>=4.47,<5.0` |

**Python:** 3.11+ required.

---

## Project Layout

```
asrbench/
├── __init__.py
├── main.py                  # FastAPI app factory + lifespan
├── config.py                # Config loader (~/.asrbench/config.toml)
├── db.py                    # DuckDB connection pool + schema init
│
├── api/
│   ├── __init__.py
│   ├── runs.py              # /runs endpoints
│   ├── models.py            # /models endpoints
│   ├── datasets.py          # /datasets endpoints
│   ├── system.py            # /system endpoints
│   └── ws.py                # WebSocket streams
│
├── engine/
│   ├── __init__.py
│   ├── benchmark.py         # BenchmarkEngine — orchestrates a run
│   ├── matrix.py            # MatrixBuilder — cartesian product of params
│   ├── compare.py           # CompareEngine — side-by-side N-run analysis
│   ├── wer.py               # WEREngine — metric computation + normalization
│   └── vram.py              # VRAMMonitor — pynvml wrapper
│
├── backends/
│   ├── __init__.py
│   ├── base.py              # BaseBackend ABC
│   ├── faster_whisper.py    # FasterWhisperBackend
│   ├── whisper_cpp.py       # WhisperCppBackend
│   ├── parakeet.py          # ParakeetBackend
│   └── qwen_asr.py          # QwenASRBackend
│
├── data/
│   ├── __init__.py
│   ├── dataset_manager.py   # DatasetManager — download + verify
│   └── audio_cache.py       # AudioCache — checksum-based preprocessing cache
│
├── activity/
│   ├── __init__.py
│   └── logger.py            # ActivityLogger — structured log + WS broadcast
│
└── cli/
    ├── __init__.py
    ├── app.py               # Typer root app
    ├── run.py               # asrbench run
    ├── models.py            # asrbench models
    ├── datasets.py          # asrbench datasets
    └── serve.py             # asrbench serve

ui/                          # Svelte frontend (compiled to asrbench/static/)
├── src/
│   ├── App.svelte
│   ├── routes/
│   │   ├── Dashboard.svelte
│   │   ├── RunDetail.svelte
│   │   ├── Compare.svelte
│   │   ├── Models.svelte
│   │   └── Datasets.svelte
│   ├── components/
│   │   ├── VRAMBar.svelte
│   │   ├── LiveLog.svelte
│   │   ├── WERChart.svelte
│   │   └── MatrixBuilder.svelte
│   └── lib/
│       ├── api.ts            # Typed API client
│       └── ws.ts             # WebSocket client

tests/
├── unit/
│   ├── test_wer_engine.py
│   ├── test_matrix_builder.py
│   └── test_vram_monitor.py
└── integration/
    ├── test_runs_api.py
    ├── test_datasets_api.py
    └── test_ws_streams.py

~/.asrbench/
├── config.toml
├── benchmark.db
└── cache/                   # Preprocessed audio cache
```

---

## Database Schema

```sql
-- Run: create schema on first startup
CREATE TABLE IF NOT EXISTS runs (
    run_id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id        UUID        NOT NULL REFERENCES models(model_id),
    backend         VARCHAR     NOT NULL,
    params          JSON        NOT NULL,
    dataset_id      UUID        NOT NULL REFERENCES datasets(dataset_id),
    lang            VARCHAR(8)  NOT NULL,
    mode            VARCHAR(16) NOT NULL
                                CHECK (mode IN ('model_compare', 'param_compare')),
    baseline_run_id UUID        REFERENCES runs(run_id),
    status          VARCHAR(16) NOT NULL DEFAULT 'pending'
                                CHECK (status IN ('pending','running','completed','failed','cancelled')),
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS segments (
    segment_id  UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id      UUID    NOT NULL REFERENCES runs(run_id),
    offset_s    DOUBLE  NOT NULL,
    duration_s  DOUBLE  NOT NULL,
    ref_text    TEXT    NOT NULL,
    hyp_text    TEXT    NOT NULL,
    wer         DOUBLE,
    cer         DOUBLE,
    mer         DOUBLE,
    wil         DOUBLE,
    rtfx        DOUBLE
);

CREATE TABLE IF NOT EXISTS aggregates (
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

CREATE TABLE IF NOT EXISTS datasets (
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

CREATE TABLE IF NOT EXISTS models (
    model_id       UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    family         VARCHAR NOT NULL,
    name           VARCHAR NOT NULL,
    backend        VARCHAR NOT NULL,
    local_path     VARCHAR NOT NULL,
    default_params JSON    NOT NULL DEFAULT '{}',
    size_bytes     BIGINT,
    registered_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS baselines (
    model_id UUID NOT NULL REFERENCES models(model_id),
    run_id   UUID NOT NULL REFERENCES runs(run_id),
    set_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (model_id)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_runs_status    ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_model_id  ON runs(model_id);
CREATE INDEX IF NOT EXISTS idx_runs_created   ON runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_segments_run   ON segments(run_id);
CREATE INDEX IF NOT EXISTS idx_datasets_lang  ON datasets(lang);
```

---

## Configuration

`~/.asrbench/config.toml` — created on first `asrbench serve` if absent:

```toml
[server]
host     = "127.0.0.1"
port     = 8765
log_level = "info"

[storage]
db_path    = "~/.asrbench/benchmark.db"
cache_dir  = "~/.asrbench/cache"

[limits]
max_concurrent_runs = 1
vram_warn_pct       = 85
```

All paths resolved via `pathlib.Path.expanduser().resolve()`. User-supplied paths validated against `storage.cache_dir` prefix to prevent path traversal.

---

## API Contracts

### POST /runs/start

**Request:**
```json
{
  "mode": "param_compare",
  "model_id": "550e8400-e29b-41d4-a716-446655440000",
  "dataset_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
  "lang": "en",
  "matrix": {
    "beam_size": [1, 2, 4, 8],
    "compute_type": ["float16"]
  },
  "baseline_run_id": null
}
```

**Response `202 Accepted`:**
```json
{
  "run_ids": ["uuid1", "uuid2", "uuid3", "uuid4"],
  "baseline_run_id": "uuid1",
  "matrix_size": 4,
  "estimated_runs": 4
}
```

**Errors:**
- `404` — `{"detail": "Model 'uuid' not found. Register it first via POST /models."}`
- `404` — `{"detail": "Dataset 'uuid' not found. Add it via POST /datasets/fetch."}`
- `409` — `{"detail": "A benchmark run is already in progress. Cancel it first via POST /runs/{id}/cancel."}`
- `422` — `{"detail": "Invalid matrix: 'beam_size' values must be positive integers."}`

---

### GET /runs/compare

**Request:** `GET /runs/compare?ids=uuid1,uuid2,uuid3`

**Response `200 OK`:**
```json
{
  "runs": [
    {
      "run_id": "uuid1",
      "params": {"beam_size": 1, "compute_type": "float16"},
      "wer_mean": 0.082,
      "cer_mean": 0.041,
      "rtfx_mean": 42.3,
      "vram_peak_mb": 1840,
      "wall_time_s": 18.2,
      "wilcoxon_p": null,
      "is_baseline": true
    },
    {
      "run_id": "uuid2",
      "params": {"beam_size": 4, "compute_type": "float16"},
      "wer_mean": 0.071,
      "cer_mean": 0.036,
      "rtfx_mean": 28.1,
      "vram_peak_mb": 1952,
      "wall_time_s": 26.7,
      "delta_wer": -0.011,
      "delta_cer": -0.005,
      "delta_rtfx": -14.2,
      "wilcoxon_p": 0.023,
      "is_baseline": false
    }
  ],
  "params_diff": ["beam_size"],
  "params_same": ["compute_type"]
}
```

**Error:**
- `400` — `{"detail": "Comparison requires at least 2 run IDs. Got: 1."}`
- `400` — `{"detail": "Run 'uuid3' has status 'running'. Only completed runs can be compared."}`

---

### POST /models/{id}/load

**Request:** `POST /models/550e8400.../load` (no body)

**Response `200 OK`:**
```json
{
  "model_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "loaded",
  "vram_used_mb": 1840,
  "vram_total_mb": 8192
}
```

**Errors:**
- `404` — `{"detail": "Model '550e8400...' not found."}`
- `409` — `{"detail": "Another model is already loaded. Unload it first via POST /models/{id}/unload."}`
- `507` — `{"detail": "Insufficient VRAM. Required: ~3800 MB, Available: 2100 MB. Try reducing compute_type to 'int8'."}`

---

### POST /datasets/fetch

**Request:**
```json
{
  "source": "common_voice",
  "lang": "tr",
  "split": "test"
}
```

**Response `202 Accepted`:**
```json
{
  "dataset_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
  "name": "common_voice_tr_test",
  "status": "downloading",
  "stream_url": "/ws/logs"
}
```

**Errors:**
- `400` — `{"detail": "Source 'librispeech' is not supported. Valid sources: common_voice, fleurs, yodas, ted_lium, custom."}`
- `409` — `{"detail": "Dataset 'common_voice_tr_test' already exists. Delete it first if you want to re-download."}`

---

### WS /ws/runs/{id}/live

**Server → Client messages:**
```json
{ "type": "segment", "segment": { "segment_id": "uuid", "offset_s": 12.3, "duration_s": 4.1, "ref_text": "hello world", "hyp_text": "hello world", "wer": 0.0, "cer": 0.0, "rtfx": 38.7 }, "segments_done": 42, "segments_total": 100, "progress": 0.42, "eta_s": 87 }
```
```json
{ "type": "complete", "run_id": "uuid", "aggregate": { "wer_mean": 0.071, "cer_mean": 0.036, "rtfx_mean": 28.1, "vram_peak_mb": 1952, "wall_time_s": 26.7 } }
```
```json
{ "type": "error", "run_id": "uuid", "message": "CUDA out of memory. Reduce beam_size or compute_type." }
```

---

## Core Class Implementations

### BaseBackend

```python
# asrbench/backends/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class Segment:
    offset_s: float
    duration_s: float
    ref_text: str
    hyp_text: str

class BaseBackend(ABC):
    """
    Contract for all ASR backend adapters.

    Preconditions:
    - load() must be called before transcribe()
    - audio must be float32 mono, 16kHz
    - lang must be ISO 639-1 code (e.g., "en", "tr")
    - params must not contain unknown keys (validated by caller)

    Side effects:
    - load() allocates GPU/CPU memory
    - unload() releases it; no-op if not loaded
    """
    family: str   # e.g., "whisper"
    name: str     # e.g., "faster-whisper"

    @abstractmethod
    def default_params(self) -> dict:
        """Return backend-specific default transcription params."""
        ...

    @abstractmethod
    def load(self, model_path: str, params: dict) -> None:
        """
        Load model into memory.
        Raises RuntimeError if model_path does not exist.
        Raises MemoryError if insufficient VRAM/RAM.
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model from memory. No-op if not currently loaded."""
        ...

    @abstractmethod
    def transcribe(self, audio: np.ndarray, lang: str, params: dict) -> list[Segment]:
        """
        Transcribe audio array.
        Returns list of Segment with hyp_text populated; ref_text is empty string.
        Raises RuntimeError if model not loaded.
        """
        ...
```

---

### WEREngine

```python
# asrbench/engine/wer.py
from jiwer import process_words, process_characters
from whisper_normalizer.english import EnglishTextNormalizer
from whisper_normalizer.basic import BasicTextNormalizer

try:
    from trnorm import normalize as tr_normalize
    _TRNORM_AVAILABLE = True
except ImportError:
    _TRNORM_AVAILABLE = False

_en_normalizer = EnglishTextNormalizer()
_basic_normalizer = BasicTextNormalizer()

class WEREngine:
    """
    Compute WER/CER/MER/WIL for a list of (reference, hypothesis) pairs.

    Normalization is always applied symmetrically to BOTH ref and hyp.
    EN pipeline: whisper_normalizer → lowercase → remove punctuation
    TR pipeline: whisper_normalizer → trnorm → Turkish-safe lowercase
    """

    _DATA_LEAKAGE_MODELS = {"whisper", "openai-whisper"}
    _DATA_LEAKAGE_DATASETS = {"librispeech", "fleurs"}

    def compute(
        self,
        refs: list[str],
        hyps: list[str],
        lang: str,
        model_family: str | None = None,
        dataset_source: str | None = None,
    ) -> dict:
        """
        Params:
        - refs: reference transcripts (ground truth)
        - hyps: hypothesis transcripts (model output)
        - lang: ISO 639-1 language code
        - model_family: for data leakage detection (optional)
        - dataset_source: for data leakage detection (optional)

        Returns dict with keys: wer, cer, mer, wil, wilcoxon_p, data_leakage_warning
        Raises ValueError if len(refs) != len(hyps) or both are empty.
        """
        if len(refs) != len(hyps):
            raise ValueError(
                f"refs and hyps length mismatch: {len(refs)} vs {len(hyps)}."
            )
        if not refs:
            raise ValueError("Cannot compute WER on empty input.")

        norm_refs = [self._normalize(t, lang) for t in refs]
        norm_hyps = [self._normalize(t, lang) for t in hyps]

        wer_out = process_words(norm_refs, norm_hyps)
        cer_out = process_characters(norm_refs, norm_hyps)

        wilcoxon_p = self._wilcoxon(norm_refs, norm_hyps) if len(refs) >= 100 else None
        leakage = self._check_leakage(model_family, dataset_source)

        return {
            "wer":  wer_out.wer,
            "cer":  cer_out.cer,
            "mer":  wer_out.mer,
            "wil":  wer_out.wil,
            "wilcoxon_p": wilcoxon_p,
            "data_leakage_warning": leakage,
        }

    def _normalize(self, text: str, lang: str) -> str:
        if lang == "tr":
            text = _basic_normalizer(text)
            if _TRNORM_AVAILABLE:
                text = tr_normalize(text)
            return text.lower()
        # Default EN pipeline
        text = _en_normalizer(text)
        return text.lower()

    def _wilcoxon(self, refs: list[str], hyps: list[str]) -> float | None:
        """Wilcoxon signed-rank test on per-segment WER. Returns p-value."""
        try:
            from scipy.stats import wilcoxon
            seg_wers_ref = [process_words([r], [r]).wer for r in refs]
            seg_wers_hyp = [process_words([r], [h]).wer for r, h in zip(refs, hyps)]
            _, p = wilcoxon(seg_wers_ref, seg_wers_hyp, zero_method="wilcox")
            return float(p)
        except Exception:
            return None

    def _check_leakage(self, model_family: str | None, dataset_source: str | None) -> bool:
        if model_family is None or dataset_source is None:
            return False
        return (
            model_family.lower() in self._DATA_LEAKAGE_MODELS
            and dataset_source.lower() in self._DATA_LEAKAGE_DATASETS
        )
```

---

### VRAMMonitor

```python
# asrbench/engine/vram.py
import threading
from dataclasses import dataclass, field

@dataclass
class VRAMSnapshot:
    used_mb: float
    total_mb: float
    pct: float
    available: bool = True  # False if pynvml unavailable or no GPU

class VRAMMonitor:
    """
    Polls NVIDIA GPU memory via pynvml.
    Graceful fallback: returns available=False snapshot when pynvml is
    unavailable, no NVIDIA GPU is present, or NVML call fails.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._handle = None
        self._init_nvml()

    def _init_nvml(self) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = pynvml
        except Exception:
            self._handle = None

    def snapshot(self) -> VRAMSnapshot:
        """
        Returns current VRAM snapshot.
        Never raises — returns available=False on any error.
        """
        if self._handle is None:
            return VRAMSnapshot(used_mb=0, total_mb=0, pct=0.0, available=False)
        try:
            with self._lock:
                info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            used_mb  = info.used  / 1024**2
            total_mb = info.total / 1024**2
            pct      = (info.used / info.total) * 100 if info.total else 0.0
            return VRAMSnapshot(used_mb=used_mb, total_mb=total_mb, pct=pct)
        except Exception:
            return VRAMSnapshot(used_mb=0, total_mb=0, pct=0.0, available=False)

    def warn_threshold_pct(self, threshold: float = 85.0) -> bool:
        """Returns True if current VRAM usage exceeds threshold percent."""
        snap = self.snapshot()
        return snap.available and snap.pct >= threshold
```

---

### MatrixBuilder

```python
# asrbench/engine/matrix.py
import itertools
from dataclasses import dataclass

@dataclass
class MatrixRun:
    params: dict
    is_baseline: bool

class MatrixBuilder:
    """
    Builds a parameter matrix for benchmark runs.

    Baseline-first rule:
    - model_compare: one run per model, same params → baseline = first model's run
    - param_compare: cartesian product of param values → baseline = default_params × 1 run
      The default_params run is always inserted first if not already in matrix.
    """

    def build_matrix(
        self,
        matrix: dict[str, list],
        default_params: dict,
        mode: str,
    ) -> list[MatrixRun]:
        """
        Params:
        - matrix: {"beam_size": [1,2,4], "compute_type": ["float16"]}
        - default_params: model's registered default parameters
        - mode: "model_compare" | "param_compare"

        Returns ordered list with baseline run first.
        Raises ValueError if matrix is empty or any value list is empty.
        """
        if not matrix:
            raise ValueError("Matrix must contain at least one parameter.")
        for key, values in matrix.items():
            if not values:
                raise ValueError(
                    f"Matrix parameter '{key}' has no values. Provide at least one."
                )

        keys = list(matrix.keys())
        combinations = [
            dict(zip(keys, combo))
            for combo in itertools.product(*[matrix[k] for k in keys])
        ]

        baseline_params = {**default_params, **{k: matrix[k][0] for k in keys}}

        runs: list[MatrixRun] = []
        baseline_added = False

        for combo in combinations:
            merged = {**default_params, **combo}
            is_baseline = (merged == baseline_params)
            if is_baseline:
                baseline_added = True
                runs.insert(0, MatrixRun(params=merged, is_baseline=True))
            else:
                runs.append(MatrixRun(params=merged, is_baseline=False))

        if not baseline_added:
            runs.insert(0, MatrixRun(params=baseline_params, is_baseline=True))

        return runs
```

---

## Implementation Order

Complete in this exact order. Each task depends on all prior tasks unless noted.

| # | Task | Key Output | Notes |
|---|------|-----------|-------|
| T1 | Project scaffold | `pyproject.toml`, `asrbench/` skeleton | Entry point `asrbench.main:app` |
| T2 | Config loader | `config.py` | `~/.asrbench/config.toml`, expand user paths |
| T3 | DuckDB init | `db.py` | Schema + indexes on first startup; connection pool |
| T4 | FastAPI app factory | `main.py` | Lifespan: DB init + NVML init; mount static UI |
| T5 | WEREngine | `engine/wer.py` | EN + TR normalization; jiwer 4.0 process_words/chars |
| T6 | VRAMMonitor | `engine/vram.py` | pynvml; graceful fallback; polling via asyncio task |
| T7 | BaseBackend + FasterWhisperBackend | `backends/` | Full transcribe loop; RTFx = audio_duration / wall_time |
| T8 | BenchmarkEngine | `engine/benchmark.py` | Single run: load → transcribe → WER → aggregate → save |
| T9 | MatrixBuilder | `engine/matrix.py` | Cartesian product; baseline-first ordering |
| T10 | `/runs` API | `api/runs.py` | start, list, detail, cancel, retry, delete, segments, export, compare |
| T11 | `/models` API | `api/models.py` | list, register, detail, delete, load, unload, set-baseline |
| T12 | DatasetManager | `data/dataset_manager.py` | HF `datasets` lib download; checksum verify; progress log |
| T13 | `/datasets` API | `api/datasets.py` | list, fetch, delete |
| T14 | ActivityLogger | `activity/logger.py` | Structured JSON log; WS broadcast via asyncio.Queue |
| T15 | WebSocket streams | `api/ws.py` | `/ws/logs`, `/ws/vram`, `/ws/runs/{id}/live`; segment messages include `segments_done`/`segments_total` |
| T16 | Typer CLI | `cli/` | All commands call FastAPI via httpx; `run` command shows `rich` progress bar (done/total + %) |
| T17 | WhisperCppBackend | `backends/whisper_cpp.py` | pywhispercpp adapter |
| T18 | ParakeetBackend + QwenASRBackend | `backends/parakeet.py`, `backends/qwen_asr.py` | NeMo + Transformers adapters |
| T19 | Svelte UI | `ui/` | Dashboard, RunDetail, Compare, Models, Datasets; RunDetail shows progress bar + segments done/total during live run |
| T20 | Export + tests | `api/runs.py`, `tests/` | JSON/CSV/PDF export; integration test suite |

---

## Error Handling Matrix

| Scenario | Detection | Response | Recovery |
|----------|-----------|----------|----------|
| Model file not found | `Path.exists()` before load | `404` with expected path | Prompt: re-register with correct `local_path` |
| VRAM OOM | `MemoryError` from backend | `507` with used/available MB | Suggest: reduce `compute_type` to `int8` or lower `beam_size` |
| Dataset download failure | `datasets` library exception | `502` with source URL | Offer retry via `POST /datasets/fetch` |
| Run cancelled mid-flight | Cancel flag checked per segment | `runs.status = 'cancelled'` | Partial segments retained; re-run via `POST /runs/{id}/retry` |
| DuckDB write error | Exception on INSERT | `500` with error detail | Log to activity; run marked `failed` |
| Backend not installed | `ImportError` on plugin load | `422` with install command | `{"detail": "Backend 'parakeet' requires nemo_toolkit. Install: pip install nemo_toolkit[asr]"}` |
| trnorm not installed | `ImportError` at WER compute | Fallback to basic lowercase | Log WARN: "trnorm not installed — Turkish normalization degraded" |

---

## Plugin Registration

Third-party backends register via `pyproject.toml` entry points:

```toml
[project.entry-points."asrbench.backends"]
my_backend = "my_package.backend:MyBackend"
```

Loaded at startup via `importlib.metadata.entry_points(group="asrbench.backends")`.

---

## Audio Preprocessing Cache

Audio is resampled to 16kHz float32 mono on first load and cached:

```
~/.asrbench/cache/<dataset_id>/<segment_checksum>.npy
```

Cache hit condition: `SHA256(original_audio_bytes) == stored_checksum`.
Cache miss: resample + save + update checksum in `datasets` table.
Re-runs on the same dataset skip preprocessing entirely.

---

## Testing Requirements

**Unit tests** (no I/O):
- `test_wer_engine.py` — EN/TR normalization; symmetric application; hallucination (empty ref); edge cases: empty string, punctuation-only, Unicode
- `test_matrix_builder.py` — baseline-first ordering; cartesian product correctness; empty matrix rejection
- `test_vram_monitor.py` — fallback snapshot when pynvml unavailable

**Integration tests** (real DuckDB, localhost FastAPI):
- `test_runs_api.py` — full run lifecycle: start → live WS → complete; cancel mid-run; retry failed run; compare 2 runs
- `test_datasets_api.py` — fetch Common Voice TR test split; checksum verify; delete
- `test_ws_streams.py` — `/ws/logs` receives activity events; `/ws/vram` delivers 500ms-interval snapshots

**Test data:** Use real audio samples (≥10 seconds) and real transcripts — never silent audio or placeholder text. WER tests must use sentences long enough to produce meaningful metric variation.
