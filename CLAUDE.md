# ASRbench

Fully local, multi-backend ASR benchmarking platform with IAMS parameter optimization.

## Quick Start

```bash
pip install -e ".[dev,faster-whisper,tr]"
python -m pytest tests/
asrbench serve
```

## Blueprint Profile

**Project:** asrbench | **Type:** devtool/backend | **Stack:** Python 3.11+, FastAPI, Typer, DuckDB | **Target:** Production

### Config
- **Priorities:** Security, Code Quality
- **Constraints:** Keep framework (FastAPI/Typer), Keep language (Python)
- **Data:** Audio files, transcripts (no PII) | **Regulations:** N/A
- **Audience:** Developers / Researchers | **Deploy:** Local (pip install)

### Project Map
```
Entry: asrbench/cli/app.py → Typer CLI
       asrbench/main.py    → FastAPI server (uvicorn)

Modules:
  asrbench/api/           → REST + WebSocket endpoints (7 files)
    runs.py               → benchmark run CRUD + start/cancel/compare
    optimization.py       → IAMS optimizer REST facade
    ws.py                 → WebSocket live progress streams
  asrbench/engine/        → Core computation layer (8 files)
    benchmark.py          → BenchmarkEngine (run orchestration)
    optimizer.py          → IAMSOptimizer (7-layer parameter search)
    wer.py                → WER/CER metrics + language-aware normalization
  asrbench/engine/search/ → IAMS search layers (12 files)
    screening.py          → Layer 1: OFAT-3 sensitivity analysis
    local_1d.py           → Layer 2+6: golden section / pattern / exhaustive
    pairwise_grid.py      → Layer 3: interaction detection
    multistart.py         → Layer 4: multi-start coordinate descent
    ablation.py           → Layer 5: leave-k-out toxic param detection
    validation.py         → Layer 7: confidence certification
  asrbench/backends/      → ASR backend adapters (5 files)
    faster_whisper.py     → faster-whisper via CTranslate2
    whisper_cpp.py        → whisper.cpp via pywhispercpp
    parakeet.py           → NVIDIA Parakeet via NeMo
    qwen_asr.py           → Qwen-Audio via HuggingFace Transformers
  asrbench/data/          → Dataset download + audio cache (2 files)
  asrbench/cli/           → CLI subcommands (4 files)

Data Flow:
  CLI/API → validate inputs → load backend + dataset
    → BenchmarkEngine.run() (per-segment transcribe → WER compute → DB insert)
    → aggregate metrics → WebSocket publish
  CLI/API → IAMSOptimizer.run() (screening → local 1D → pairwise grid
    → multi-start → ablation → refinement → validation)
    → best config + confidence → DB persist

External: DuckDB (embedded), HuggingFace Datasets (download), pynvml (VRAM)
Toolchain: ruff + mypy | pytest | hatchling build
```

### Ideal Metrics
| Metric | Target |
|--------|--------|
| Coupling | < 5 imports/module avg |
| Cohesion | Single responsibility per module |
| Complexity | CC ≤ 15 per function |
| Coverage | ≥ 70% |

### Current Scores
| Dimension | Score | Prev | Delta | Status |
|-----------|-------|------|-------|--------|
| Security & Privacy | 92 | 92 | 0 | OK |
| Code Quality | 89 | 85 | +4 | OK |
| Architecture | 92 | 92 | 0 | OK |
| Performance | 94 | 94 | 0 | OK |
| Resilience | 93 | 93 | 0 | OK |
| Testing | 85 | 85 | 0 | OK |
| Stack Health | 88 | 88 | 0 | OK |
| DX | 90 | 90 | 0 | OK |
| Documentation | 92 | 92 | 0 | OK |
| Overall | 91 | 90 | +1 | OK |

### Last Run
- 2026-04-25: ds-blueprint auto | Findings: 19 (0 HIGH, 4 MEDIUM, 15 LOW) | Fixed: 0 | Skipped: 0 | Failed: 0 | Overall 90→91

## End Blueprint Profile
