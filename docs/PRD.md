# asrbench — Product Requirements Document

**Version:** 1.0.0
**Date:** 2026-04-06
**Status:** Approved

---

## Executive Summary

ASR (Automatic Speech Recognition) developers today write project-specific benchmark scripts that are non-portable, tied to a single backend, and produce non-reproducible results. No pip-installable, backend-agnostic ASR evaluation tool combining multi-backend support, a parameter matrix builder, unlimited benchmark history, and a local web dashboard exists as of 2026.

**asrbench** is a fully local ASR benchmarking platform that lets any developer `pip install asrbench`, run `asrbench serve`, and immediately access a web dashboard to run reproducible, multi-backend, multi-language benchmarks — with complete benchmark history, side-by-side comparison, real-time VRAM monitoring, and zero cloud dependency.

---

## Target Users

**Primary:** ASR developers who need to evaluate and compare speech recognition models — researchers, application developers choosing an ASR backend, practitioners validating model upgrades.

**Distribution:** Open source, PyPI. GitHub first, PyPI on first stable release.

### User Stories

1. As an ASR developer, I want to run a model across a parameter matrix (e.g., beam_size 1–5) so I can identify the optimal speed/accuracy trade-off with data.
2. As an ASR developer, I want to compare multiple backends (faster-whisper, whisper.cpp, etc.) on the same dataset so I can make objective backend selection decisions.
3. As an ASR developer, I want to view past benchmark runs side-by-side so I can clearly see which experiments used which parameters and how results differed.
4. As an ASR developer, I want to measure the WER impact of an audio preprocessing pipeline (e.g., filters, normalization) by comparing raw vs. processed audio.
5. As an ASR developer, I want the system to warn me when I'm approaching VRAM limits and suggest which parameter to reduce.
6. As a CLI user, I want to run `asrbench run --config bench.yaml` without opening a browser so I can integrate benchmarks into CI/CD pipelines.
7. As an ASR developer, I want to browse available datasets (duration, size, language, split) and download them directly from the UI without touching the terminal.

---

## Feature Requirements

All features are targeted for v1.

### Must-Have

| # | Feature | Description |
|---|---------|-------------|
| F1 | Backend adapter + plugin system | faster-whisper, whisper.cpp, parakeet, qwen-asr built-in; TRT-LLM via plugin interface. Each backend exposes model family defaults automatically. |
| F2 | Parameter matrix builder | Select which parameters × which values form a test matrix. Two modes: **model comparison** (same params, different models) and **parameter comparison** (same model, different params). Baseline = default params × 1 run; all subsequent runs report Δ against it. |
| F3 | Web dashboard + CLI parity | Every granular operation (model load/unload, run start/cancel/retry, dataset management, compare, export) available from both UI and CLI. FastAPI core — both interfaces call the same endpoints. |
| F4 | Benchmark history + comparison | Unlimited benchmark history in DuckDB. Compare N runs side-by-side; same/different parameters clearly highlighted. Chart and table visualizations. |
| F5 | VRAM monitor | Real-time VRAM usage via pynvml (cross-platform: Windows + Linux). OOM warning before it occurs; suggests which parameter to reduce. |
| F6 | Dataset browser + downloader | Supports Common Voice 17+, FLEURS, YODAS, TED-LIUM. Per-dataset metadata visible in UI: duration, size, language, split. One-click download with checksum verification. |
| F7 | Activity log | Dynamic log level (DEBUG/INFO/WARN/ERROR). Visual terminal-log replacement in UI. On error: retry/resume/skip from UI — zero console required. |
| F8 | WER engine | jiwer v4.0.0 (WER + CER + MER + WIL). EN pipeline: `whisper_normalizer → lowercase → remove punctuation`. TR pipeline: `whisper_normalizer → trnorm → Turkish-safe lowercase`. CER primary metric for Turkish. RTFx reported alongside WER. Wilcoxon p-value for statistical significance. Data leakage warning shown when evaluating Whisper models on LibriSpeech or FLEURS. |
| F9 | Preprocessing pipeline comparison | Compare raw vs. preprocessed audio WER to measure pipeline impact. |
| F10 | Export | Export any run or comparison as JSON, CSV, or PDF. |
| F11 | Custom dataset import | Import local audio + transcript pairs as a named dataset. |
| F12 | Statistical significance chart | Visual Wilcoxon p-value display on comparison views. |
| F13 | Hallucination detection | Leverage jiwer v4 empty-reference support to flag and score hallucinated transcriptions. |

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Functional accuracy | WER/CER/RTFx within ±0.2 pp of HF Open ASR Leaderboard public scores for same models | Manual verification against published scores |
| Benchmark overhead | ≤5% vs. direct model call | Profiling: asrbench run vs. raw faster-whisper call |
| Dashboard response | <200ms for all UI interactions | Browser DevTools / Lighthouse |
| DuckDB query time | <100ms on 10,000+ run history | Query profiling |
| Onboarding | `pip install` + `asrbench serve` + first completed benchmark in ≤10 steps, zero console required | Manual walkthrough |
| Backend coverage | faster-whisper + whisper.cpp + parakeet + qwen-asr all functional; EN + TR + ≥2 additional languages downloadable | Integration test suite |

---

## Competition

| Tool | Gap |
|------|-----|
| HF Open ASR Leaderboard | Requires A100 GPU; not pip-runnable as a personal tool |
| ESPnet | Full research framework — complex setup, not a lightweight runner |
| jiwer | Metric library only — no runner, no dashboard |
| Speechmatics sm-metrics | WER/CER normalization only — no backend runner |

**Validated gap (HIGH confidence):** No tool combines pip-installable setup, multi-backend support, YAML config, web dashboard, unlimited history, and local-first operation.

---

## Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Scope creep (20 features in v1) | High | High | MS1–3 must complete before MS4–6 begin; each milestone has explicit go/no-go criteria |
| Backend API instability (parakeet, qwen-asr) | Medium | High | BaseBackend interface is isolated — one adapter breaking does not affect others; faster-whisper adapter built first |
| VRAM monitoring platform differences | Low | Medium | pynvml provides same API on Windows and Linux; graceful fallback if unavailable |
| Web UI complexity | Medium | High | Svelte chosen for minimal footprint; UI (T16) not started until BenchmarkEngine (T7) and Compare engine (T13) are validated via CLI |

---

## Timeline & Milestones

| Milestone | Tasks | Go/No-Go Criteria |
|-----------|-------|-------------------|
| MS1 — Foundation | T1–T4 | `asrbench serve` starts; `/health` responds |
| MS2 — WER + First Backend | T5–T7 | Single-language faster-whisper run completes; WER/CER/RTFx correct |
| MS3 — Datasets + Backends | T8–T11 | All 4 backends run; datasets download and validate |
| MS4 — Matrix + Compare | T12–T13 | Matrix run and N-run comparison work via CLI |
| MS5 — Web UI + Live Streams | T14–T16 | Full dashboard functional; live log/VRAM/progress streams work |
| MS6 — Complete | T17–T20 | Export, preprocessing compare, statistics, performance optimized |
