# Changelog

All notable changes to ASRbench are documented here.

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
