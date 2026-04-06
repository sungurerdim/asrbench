# ASRbench

Fully local, multi-backend ASR benchmarking platform.

```bash
pip install asrbench
asrbench serve
```

Opens a web dashboard at `http://localhost:8765`. No cloud. No authentication. No GPU required.

## Features

- **Multi-backend** — faster-whisper, whisper.cpp, Parakeet, Qwen-ASR, and third-party plugins
- **Parameter matrix** — sweep beam_size, compute_type, or any backend param; baseline-first ordering
- **Unlimited history** — DuckDB storage; compare N runs side-by-side with delta metrics
- **WER / CER / MER / WIL** — jiwer 4.0; EN and Turkish normalization pipelines built in
- **Real-time VRAM monitor** — pynvml; OOM warning before it occurs
- **Dataset browser** — Common Voice, FLEURS, YODAS, TED-LIUM; one-command download
- **CLI + Web UI parity** — every operation available from both interfaces
- **Export** — JSON, CSV, PDF

## Quick Start

```bash
# Install with faster-whisper backend
pip install "asrbench[faster-whisper]"

# Register a model
asrbench models register --path /path/to/faster-whisper-large-v3 --backend faster-whisper

# Download a dataset
asrbench datasets fetch --source common_voice --lang en --split test

# Run a benchmark
asrbench run --model-id <id> --dataset-id <id> --lang en

# Or: parameter matrix via config
asrbench run --config bench.yaml
```

`bench.yaml` example:

```yaml
mode: param_compare
model_id: <uuid>
dataset_id: <uuid>
lang: en
matrix:
  beam_size: [1, 2, 4, 8]
  compute_type: [float16, int8]
```

## Backends

| Backend | Install | Status |
|---------|---------|--------|
| faster-whisper | `pip install "asrbench[faster-whisper]"` | Built-in |
| whisper.cpp | `pip install "asrbench[whisper-cpp]"` | Built-in |
| Parakeet | `pip install "asrbench[parakeet]"` | Built-in |
| Qwen-ASR | `pip install "asrbench[qwen-asr]"` | Built-in |
| Custom | entry point `asrbench.backends` | Plugin |

## Development

```bash
git clone https://github.com/sungurerdim/asrbench
cd asrbench
pip install -e ".[dev,faster-whisper]"
pytest
```

## License

MIT
