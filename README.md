# ASRbench

[![PyPI](https://img.shields.io/pypi/v/asrbench.svg)](https://pypi.org/project/asrbench/)
[![Python](https://img.shields.io/pypi/pyversions/asrbench.svg)](https://pypi.org/project/asrbench/)
[![CI](https://github.com/sungurerdim/asrbench/actions/workflows/ci.yml/badge.svg)](https://github.com/sungurerdim/asrbench/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Fully local, multi-backend ASR benchmarking platform with automated
parameter optimization.

```bash
pip install "asrbench[faster-whisper]"
asrbench serve --open
```

A browser tab opens on `http://127.0.0.1:8765/` with the full Svelte
dashboard: run comparisons, IAMS optimization, dataset + model
registries, real-time VRAM, structured activity logs. All data stays on
the local machine — DuckDB is the only store, no cloud call is made
without the user asking for it.

## Features

* **Four-backend surface** — faster-whisper, whisper.cpp, NVIDIA
  Parakeet (via NeMo), Qwen2-Audio (via Transformers). Parakeet and
  Qwen ship as opt-in extras so the base install stays light.
* **IAMS optimizer** — a 7-layer automated parameter search
  (screening, 1D coordinate descent, pairwise interaction detection,
  multi-start, ablation, refinement, confidence validation). One
  API call, one reproducible best config.
* **Two-stage + global-config runs** — coarse→fine schedules on a
  single dataset, or a single shared preset optimised across N
  datasets with weighted aggregation.
* **Audio preprocessing pipeline** — VAD, loudness normalisation,
  noise reduction, filters, codec simulation. Every knob is an IAMS
  search dimension.
* **WER / CER / MER / WIL** — language-aware normalization (EN, TR,
  AR, ZH, JA, KO) with bootstrap 95 % CI (blockwise per-speaker when
  labels are present).
* **Live dashboard** — Svelte 5 SPA served from the same process,
  WebSocket progress streams, per-GPU VRAM widget.
* **CLI + REST parity** — every operation reachable both ways. The
  CLI itself is an httpx client of the server.
* **Compare + export** — N-run comparison with paired Wilcoxon
  significance, JSON / CSV out of the box, PDF via the `[pdf]` extra.

## Quick start

```bash
# 1. Install with the faster-whisper backend (MIT; small; reference).
pip install "asrbench[faster-whisper]"

# 2. Sanity-check the environment.
asrbench doctor

# 3. Launch the server and open the UI.
asrbench serve --open

# 4. In another terminal — register a model, fetch a dataset, benchmark.
asrbench models register --path /path/to/faster-whisper-large-v3 \
    --backend faster-whisper --name fw-large-v3
asrbench datasets fetch --source fleurs --lang en --split test
asrbench run start --model <model-id> --dataset <dataset-id> --lang en
```

## Backend matrix

| Backend | Extra | License | Commercial OK | Notes |
|---------|-------|---------|---------------|-------|
| faster-whisper | `[faster-whisper]` | MIT (code), MIT (weights) | ✅ | Reference backend; CTranslate2. |
| whisper.cpp | `[whisper-cpp]` | MIT | ✅ | CPU-friendly; ggml weights. |
| Parakeet | `[parakeet]` | Apache-2.0 (runtime), CC-BY-4.0 (weights) | ✅ with attribution | NVIDIA NeMo; CUDA-coupled install. |
| Qwen-Audio | `[qwen]` | Apache-2.0 (runtime), **Qwen Community License (weights)** | ⚠ **restricted** | Read the license before production use. |

See `docs/plugins/parakeet.md` and `docs/plugins/qwen_asr.md` for the
full per-backend parameter list.

## Dataset matrix

| Source | License | HF gated? | Built-in support |
|--------|---------|-----------|------------------|
| LibriSpeech | CC-BY-4.0 | No | ✅ |
| FLEURS | CC-BY-4.0 | No | ✅ |
| Common Voice | CC0-1.0 | Yes — accept terms on HF | ✅ |
| TED-LIUM | CC-BY-NC-ND-3.0 | No | ✅ (non-commercial) |
| Earnings22 | CC-BY-SA-4.0 | No | ✅ |
| MediaSpeech | CC-BY-4.0 | No | ✅ |
| YODAS | CC-BY-4.0 | No | ✅ |
| Custom (local path) | — | — | ✅ via `--local-path` |

Gated datasets need `HF_TOKEN` set in the environment. `asrbench
doctor` flags this.

## Configuration

ASRbench reads `~/.asrbench/config.toml`:

```toml
[server]
host = "127.0.0.1"
port = 8765
log_level = "info"

[storage]
# db_path and cache_dir default to ~/.asrbench/ when omitted

[limits]
max_concurrent_runs = 1
vram_warn_pct = 85.0
dataset_fetch_timeout_s = 600.0
segment_timeout_s = 120.0
```

Inspect or mutate:

```bash
asrbench config init
asrbench config show
asrbench config set server.port 9000
asrbench config path
```

### Exposing the server on the network

Loopback is the default and authentication-free. To expose the API on
a LAN or behind a reverse proxy:

```bash
export ASRBENCH_API_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
asrbench serve --host 0.0.0.0 --allow-network
```

Every remote request must then carry an `X-API-Key` header matching
`ASRBENCH_API_KEY`. Loopback requests stay unauthenticated so the CLI
keeps working.

## CLI reference

```text
asrbench serve          [--host] [--port] [--allow-network] [--dev] [--open/--no-open]
asrbench doctor         [--json]
asrbench config         init | show | set | path
asrbench run            start | status | list | cancel | retry | delete | export | segments
asrbench compare        <run-id> <run-id> ...
asrbench models         list | register | load | unload
asrbench datasets       list | fetch
asrbench optimize       start | two-stage | global-config | status
asrbench power          suggest
```

`asrbench --version` prints the package version.

## REST API

See [`docs/API.md`](docs/API.md) for the auto-generated endpoint list
(regenerated on every release from the live OpenAPI schema).

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `asrbench doctor` flags `ffmpeg` as WARN | Install via `winget install ffmpeg`, `brew install ffmpeg`, or `apt install ffmpeg`. |
| HuggingFace gated-dataset fetch returns 403 | `export HF_TOKEN=hf_…` and accept the dataset terms on the HF website. |
| Run stays `pending` forever | A previous process crashed while launching; restart the server — startup recovery flips orphaned rows to `failed`. |
| 429 on `/runs/start` | Rate limiter fired. `POST /runs/start` allows ~120 req/min per IP; reduce churn or raise `limits.vram_warn_pct`'s sibling settings. |
| `ResourceExhausted: Not enough VRAM` | The estimate is a hard guard — free VRAM or pick a smaller `compute_type`. |

## Third-party licenses

See [`THIRD-PARTY-LICENSES.md`](THIRD-PARTY-LICENSES.md) for the full
list of dependencies, their licenses, and per-backend commercial-use
notes.

## Development

```bash
git clone https://github.com/sungurerdim/asrbench
cd asrbench
uv sync --frozen --extra dev --extra faster-whisper --extra tr
uv run pre-commit install
make test       # or: uv run pytest
make lint       # or: uv run ruff check asrbench tests
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full setup flow (pip
fallback, UI dev, CI layout).

## License

MIT — see `LICENSE`. ASRbench is the work of [Sungur Zahid Erdim](https://github.com/sungurerdim).
