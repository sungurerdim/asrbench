# ASRbench

Fully local, multi-backend ASR benchmarking platform with IAMS parameter
optimization.

```bash
pip install "asrbench[faster-whisper]"
asrbench serve --open
```

Opens a web dashboard at `http://localhost:8765`. No cloud. No
authentication. No GPU required (CPU-only via whisper.cpp is a
supported extra).

## Features

- **Multi-backend** — faster-whisper + whisper.cpp built-in;
  Parakeet / Qwen-ASR shipped as third-party plugin templates under
  `docs/plugins/`.
- **Parameter matrix** — sweep beam_size, compute_type, or any backend
  param; baseline-first ordering so deltas are obvious.
- **IAMS parameter optimization** — 7-layer automated search (screening,
  coordinate descent, pairwise interaction detection, multi-start,
  ablation, refinement, validation) with a confidence-certification
  layer.
- **Unlimited history** — DuckDB embedded storage; compare N runs
  side-by-side with param diffs and paired Wilcoxon significance.
- **Real-time VRAM monitor** — pynvml-backed singleton; 500 ms sampler
  pushes to any connected `/ws/vram` subscriber.
- **Dataset browser** — Common Voice, FLEURS, YODAS, TED-LIUM,
  LibriSpeech, Earnings-22, MediaSpeech; one-call download + checksum.
- **CLI + Web UI parity** — every operation reachable from both
  surfaces; the CLI talks to the same FastAPI server.
- **Language-aware WER normalization** — EN, TR, AR, ZH, JA, KO with
  bootstrap 95% CI (blockwise per-speaker where labels are present).
- **Export** — JSON, CSV out of the box; PDF via the `[pdf]` extra.

## Quick Start

```bash
# 1. Install the package and the built-in faster-whisper backend.
pip install "asrbench[faster-whisper]"

# 2. Sanity-check the environment (Python, ffmpeg, backends, GPU).
asrbench doctor

# 3. Launch the server and open the dashboard.
asrbench serve --open

# From another terminal — register a model and dataset, then run.
asrbench models register --path /path/to/faster-whisper-large-v3 --backend faster-whisper
asrbench datasets fetch --source fleurs --lang en --split test
asrbench run start --model <model-id> --dataset <dataset-id> --lang en
```

## CLI Reference

```bash
asrbench serve [--host] [--port] [--open/--no-open] [--reload]
asrbench doctor [--json]
asrbench config init|show|set|path
asrbench run start|status|list|cancel|retry|delete|export|segments
asrbench compare <run-id> <run-id> ...
asrbench models list|register|load|unload
asrbench datasets list|fetch
asrbench optimize start|two-stage|global-config|status
asrbench power suggest
```

`--version` at the top level prints the package version.

## Backends

| Backend | Install | Ships with core? |
|---------|---------|------------------|
| faster-whisper | `pip install "asrbench[faster-whisper]"` | Yes (reference backend) |
| whisper.cpp | `pip install "asrbench[whisper-cpp]"` | Yes |
| Parakeet (NeMo) | see `docs/plugins/parakeet.md` | No — third-party plugin |
| Qwen-ASR | see `docs/plugins/qwen_asr.md` | No — third-party plugin |
| Custom | entry point `asrbench.backends` | No — user plugin |

NeMo and Transformers are heavy (>2 GB of install space) and carry
CUDA-version coupling that varies per host, so they live under a plugin
template instead of in the core package.

## Configuration

ASRbench reads `~/.asrbench/config.toml`. Inspect or edit it with:

```bash
asrbench config init           # create with defaults
asrbench config show           # print current contents
asrbench config set server.port 9000
asrbench config path           # print the absolute path
```

The server binds to `127.0.0.1` by default. Changing `server.host` to a
non-loopback address triggers a warning at startup — ASRbench has no
authentication and is meant to run locally.

## Development

```bash
git clone https://github.com/sungurerdim/asrbench
cd asrbench
pip install -e ".[dev,faster-whisper,tr]"
pytest
```

## License

MIT
