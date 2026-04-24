# Changelog

All notable changes to ASRbench are documented here. The project
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0] - 2026-04-24

First public release on PyPI.

### Added

- **Public API auth** — `AuthMiddleware` with loopback bypass and
  mandatory `X-API-Key` for non-loopback clients. `asrbench serve`
  refuses to bind to non-loopback hosts without `--allow-network` +
  `ASRBENCH_API_KEY`.
- **Filesystem allow-list** — `LocalPath` validator confines
  dataset/model paths to `~/.asrbench` plus
  `ASRBENCH_ALLOWED_PATHS`. Traversal attacks (`../`) fail resolution.
- **Rate-limit refinement** — POST /runs/start, /optimize/start,
  /datasets/fetch, /models/register now subject to the limiter. Only
  GET polling on /runs/ and /optimize/ stays exempt.
- **Traceback sanitisation** — `engine/errors.py::sanitize_error`
  strips tracebacks before they land in DB rows or API responses.
- **FFmpeg param allow-list** — string-valued preprocessing knobs go
  through a whitelist regex (defence-in-depth; today only the `format`
  key reaches the subprocess).
- **Startup recovery** — orphaned `pending`/`running` rows flip to
  `failed` on boot so a hard kill can't leave forever-spinning jobs.
- **Graceful shutdown** — 30 s drain window sets
  `cancel_requested=true` on live runs, then flips stragglers to
  `cancelled` with a diagnostic error message.
- **DB-backed cancel flag** — `runs.cancel_requested` +
  `optimization_studies.cancel_requested` replace the in-memory set;
  benchmark engine polls between segments and raises `RunCancelled`.
- **UI wheel shipping** — Vite bundle committed to `asrbench/static`
  and included in the built wheel. `pip install asrbench` now serves
  the Svelte dashboard on first launch.
- **`--dev` serve mode** — pairs with `cd ui && npm run dev`; redirects
  `--open` to the Vite dev server.
- **Dataset fetch timeout** — `limits.dataset_fetch_timeout_s`
  (default 600 s) wraps the HF stream in `asyncio.wait_for`.
- **Per-segment backend timeout** — `limits.segment_timeout_s`
  (default 120 s) killed stuck CUDA kernels before they starve the
  event loop.
- **VRAM capacity guard** — `VRAMMonitor.require_capacity()` refuses
  backend loads when free VRAM + 10 % margin is insufficient;
  `ResourceExhausted` replaces raw mid-load CUDA OOMs.
- **NVIDIA Parakeet backend** (`asrbench[parakeet]`, Apache-2.0 +
  CC-BY-4.0 weights) — TDT/CTC/RNN-T variants via NeMo.
- **Qwen-Audio backend** (`asrbench[qwen]`, **Qwen Community License
  — commercial use restricted**) — Qwen2-Audio via Transformers with
  a fixed transcription-only chat template.
- **Prometheus metrics** (`asrbench[observability]`) — `/metrics`
  endpoint plus custom gauges: `asrbench_run_duration_seconds`,
  `asrbench_run_total{status}`, `asrbench_vram_used_mb`,
  `asrbench_backend_errors_total{backend}`.
- **Log rotation** — `--log-file`, `--log-max-mb`,
  `--log-backup-count` flags on `asrbench serve`.
- **Docker image** — multi-stage (Node 20 → Python 3.11 slim), runs
  as non-root `asrbench` user with `HEALTHCHECK`. `docker-compose.yml`
  included.
- **CI/CD** — `.github/workflows/ci.yml` runs lint, test (Python
  3.11/3.12/3.13 × Linux/macOS/Windows), UI bundle diff, wheel build.
  `.github/workflows/release.yml` publishes to PyPI via OIDC trusted
  publisher on `v*` tags.
- **Lockfile** — `uv.lock` pins the 111-package resolution so
  `uv sync --frozen` reproduces CI exactly.
- **Pre-commit** — `ruff`, `ruff-format`, standard hygiene hooks,
  plus a local reminder when `ui/src` is newer than the committed
  bundle.
- **Coverage gating** — `pytest --cov-fail-under=70` (current run:
  78.09 %).
- **Doctor extensions** — DuckDB version, `~/.asrbench` writability,
  disk space ≥ 10 GB, HF_TOKEN, CUDA_VISIBLE_DEVICES, Parakeet/Qwen
  extras.
- **Auto-generated docs** — `scripts/gen_api_docs.py` + CI
  `docs-check` job keep `docs/API.md` in sync with the live OpenAPI
  schema.
- **README rewrite + `THIRD-PARTY-LICENSES.md`** — commercial-use
  matrix, gated-dataset notes, per-extra license table.

### Changed

- **`api/optimization.py` refactored** from 1459 lines into 5 focused
  modules: `optimization.py` (router, 348 LOC), `_optimization_models`
  (Pydantic, 192), `_optimization_persistence` (DuckDB helpers, 486),
  `_optimization_runners` (single-study task, 221),
  `_optimization_multistage` (two-stage + global-config, 346).
- **Ruff ruleset expanded** to `E, F, I, UP, B, RUF, SIM, C4` with
  narrow per-file ignores for CLI / tests.
- **Mypy** now enables warn_return_any, strict_equality,
  no_implicit_optional, warn_unused_ignores / redundant_casts.
- **Bumped** `@sveltejs/vite-plugin-svelte` 4.x → 5.x to clear a
  Vite 6 peer-dep conflict.

### Security

- Full security-hardening pass (auth, rate-limit, path whitelist,
  traceback sanitisation, ffmpeg param allow-list).

## [0.2.0] - 2026-04-23

Structural refactor focused on correctness, DX, and real-time surface area.

### Added

- **Version SSOT** — `asrbench/_version.py` is the single source; both runtime
  (`asrbench.__version__`, `/system/health`, the FastAPI app title) and
  `pyproject.toml` (`[tool.hatch.version]`) read from it.
- **CORS middleware** with a strict localhost-only origin regex
  (localhost, 127.0.0.1, \[::1\], any port; HTTP or HTTPS), plus a bind
  guard that warns when `server.host` is set outside the loopback hosts.
- **Engine/vram.py — VRAMMonitor** singleton that keeps a single NVML
  handle for the lifetime of the process. `peak_mb` / `reset_peak()`
  bracket a run so `aggregates.vram_peak_mb` is finally populated.
- **Engine/matrix.py — MatrixBuilder** cartesian sweep with
  baseline-first ordering; 8 unit tests cover validation and combinator
  edge cases.
- **Engine/compare.py — CompareEngine** emits `params_diff` /
  `params_same`, per-run `delta_*`, `is_baseline`, and a paired Wilcoxon
  signed-rank p-value when exactly two runs are compared.
- **Engine/events.py — asyncio EventBus** with bounded per-subscriber
  queues and oldest-on-full backpressure. Topic convention:
  `runs:<id>`, `optimize:<id>`, `datasets:<id>`, `vram`, `activity`.
- **ActivityLogger** writes structured JSON records to stderr and fans
  them out on the `activity` topic.
- **New WS endpoints** — `/ws/datasets/{id}`, `/ws/vram`, `/ws/activity`,
  and a `/ws/logs` alias. A 500 ms VRAM sampler is ref-counted across
  dashboard subscribers.
- **API endpoints filled in** — `DELETE /runs/{id}`,
  `POST /runs/{id}/cancel`, `POST /runs/{id}/retry`,
  `GET /runs/{id}/export?fmt={json,csv}`, `DELETE /datasets/{id}`,
  compare response enriched with diff/same/deltas.
- **CLI** — `asrbench config init|show|set|path`, `asrbench doctor`
  (coloured OK/WARN/FAIL table; `--json` for CI), `asrbench compare`,
  `asrbench run cancel|retry|delete|export|segments`,
  `asrbench serve --open`, and a top-level `--version` flag.
- **Pydantic `LocalPath`** validator normalises user-supplied
  `local_path` fields (expanduser + resolve) and rejects control
  characters / NUL bytes.

### Changed

- **Config SSOT** — `_DEFAULTS` dict is the only place that defines
  default values; dataclass fields, the on-disk TOML template, and the
  fallback values used by `_build_config` all read from it.
- **Package dependency pins widened** — `fastapi<1.0`, `duckdb<2.0`,
  `jiwer<5.0`, `httpx<1.0`; `trnorm` now installs from git+https.
- **WEREngine.compute** no longer emits `wilcoxon_p`. The single-run
  implementation compared per-segment WER against a list of zeros, which
  is not a meaningful statistical test. Pairwise significance lives in
  `CompareEngine` now and surfaces as `wilcoxon_p` in the compare
  response.
- **WS endpoints** migrated from 2 s DB polling to event-bus
  subscription; run completion events close the stream cleanly instead
  of letting the poller continue until a terminal status arrives.
- **Rate-limit middleware** now exempts the `/runs/` and `/optimize/`
  path prefixes so UI polling on long-running benchmarks does not trip
  the per-IP 120 req/min bucket.
- **Log levels** — model paths under the optimizer are logged at DEBUG
  instead of INFO to avoid dumping local filesystem layout into the
  default serve log.
- **Aggregates schema** gains a `wil_mean` column (migrated
  idempotently) so `BenchmarkTrialExecutor` can stop aliasing `mer_mean`
  into the `wil` slot.

### Fixed

- **AudioCache round-trips `speaker_id`** so the blockwise-bootstrap
  CI keeps its speaker context across cache hits.
- **SubprocessBackend.family** now derives from the resolved backend
  class's entry-point metadata instead of being silently empty.

### Removed

- **`bench.bat`, `setup.bat`, `run_optimize.bat`, `matrix_bench.py`,
  `optimize_matrix.py`** — replaced by cross-platform `asrbench` CLI
  subcommands. `auto/` moved to `scripts/benchmarks/` with the bench.sh
  path updated.
- **Single-run `wilcoxon_p`** from the WER engine response (see Changed).

## [0.1.0] - 2026-04-10

Initial release.

### Added

- **Benchmark engine** with per-segment timing, corpus-level WER/CER/MER/WIL, RTFx, and VRAM tracking
- **4 built-in backends:** faster-whisper, whisper.cpp, Parakeet (NeMo), Qwen-ASR (Transformers)
- **Plugin system** for third-party backends via entry points
- **Parameter matrix** expansion for beam_size, compute_type, and arbitrary backend params
- **IAMS parameter optimizer** — 7-layer automated search (screening, coordinate descent, pairwise interaction detection, multi-start, ablation, refinement, validation)
- **Language-aware WER normalization** for EN, TR, AR, ZH, JA, KO with bootstrap 95% CI
- **Warmup run** before first segment to eliminate cold-start RTFx bias
- **REST API** — runs, models, datasets, optimization, system health, WebSocket live progress
- **CLI** — `asrbench serve`, `asrbench optimize`
- **DuckDB** embedded storage with full schema (runs, segments, aggregates, optimization studies/trials)
- **Dataset management** — Common Voice, FLEURS, YODAS, TED-LIUM download and preparation
- **Export** — JSON, CSV (PDF planned)
- **Data leakage detection** for Whisper models on LibriSpeech/FLEURS
- **Wilcoxon signed-rank test** for statistical significance on 100+ segments
- **Transcript caching** to skip re-transcription on parameter-only changes
