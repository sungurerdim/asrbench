# Third-Party Licenses

ASRbench itself is released under the **MIT** license (see `LICENSE`).
This file lists the third-party components that ship with, or ship
alongside, the package and the licenses they carry. Every entry is
kept current by hand — when you add or bump a dependency, update the
corresponding row in the same PR.

For a machine-readable dump of every resolved package license in the
current install, run:

```bash
pip install pip-licenses
pip-licenses --format=markdown > docs/LICENSES-deps.md
```

## Runtime dependencies (always installed)

| Package | License | Notes |
|---------|---------|-------|
| [FastAPI](https://github.com/tiangolo/fastapi) | MIT | REST + WebSocket server |
| [uvicorn](https://www.uvicorn.org/) | BSD-3-Clause | ASGI runner |
| [Typer](https://typer.tiangolo.com/) | MIT | CLI framework |
| [DuckDB](https://duckdb.org/) | MIT | Embedded analytics DB |
| [jiwer](https://github.com/jitsi/jiwer) | Apache-2.0 | WER/CER/MER/WIL metrics |
| [pynvml](https://github.com/gpuopenanalytics/pynvml) | BSD-3-Clause | NVIDIA VRAM readings |
| [pydantic](https://github.com/pydantic/pydantic) | MIT | Request/response models |
| [soundfile](https://github.com/bastibe/python-soundfile) | BSD-3-Clause | Audio I/O |
| [soxr](https://github.com/dofuuz/python-soxr) | LGPL-2.1 | Resampler wrapper |
| [datasets](https://github.com/huggingface/datasets) | Apache-2.0 | HuggingFace dataset loader |
| [httpx](https://www.python-httpx.org/) | BSD-3-Clause | HTTP client (used by CLI → API) |
| [whisper-normalizer](https://github.com/jitsi/whisper_normalizer) | MIT | OpenAI Whisper text normaliser |

## Optional extras

Install with `pip install 'asrbench[<extra>]'`. Each extra brings its
own transitive deps — use `pip-licenses` after installing to inspect
the full tree.

### `[faster-whisper]`
| Component | License | Notes |
|-----------|---------|-------|
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT | CTranslate2 inference |
| OpenAI Whisper weights (HF) | MIT | Per-model card; large multilingual checkpoint |

### `[whisper-cpp]`
| Component | License | Notes |
|-----------|---------|-------|
| [pywhispercpp](https://github.com/absadiki/pywhispercpp) | MIT | whisper.cpp bindings |
| ggml whisper weights | MIT | Ported from OpenAI Whisper; GGUF container |

### `[parakeet]` — NVIDIA Parakeet via NeMo
| Component | License | Notes |
|-----------|---------|-------|
| [NeMo](https://github.com/NVIDIA/NeMo) | Apache-2.0 | Runtime toolkit |
| Parakeet checkpoints | CC-BY-4.0 | Per NVIDIA NGC model cards; attribution required |
| [PyTorch](https://pytorch.org/) | BSD-3-Clause | Tensor runtime |
| [OmegaConf](https://omegaconf.readthedocs.io/) | BSD-3-Clause | Config DSL |

### `[qwen]` — Qwen2-Audio via Transformers
| Component | License | Notes |
|-----------|---------|-------|
| Qwen2-Audio weights | **Qwen Community License** | **Commercial use restricted** — see <https://github.com/QwenLM/Qwen2-Audio/blob/main/LICENSE> |
| [transformers](https://github.com/huggingface/transformers) | Apache-2.0 | Model runtime |
| [PyTorch](https://pytorch.org/) | BSD-3-Clause | Tensor runtime |
| [accelerate](https://github.com/huggingface/accelerate) | Apache-2.0 | Device mapping |
| [librosa](https://librosa.org/) | ISC | Audio I/O helpers |

### `[tr]` — Turkish normalizer
| Component | License | Notes |
|-----------|---------|-------|
| [trnorm](https://github.com/ysdede/trnorm) | MIT | Turkish ASR text normaliser |
| [SciPy](https://scipy.org/) | BSD-3-Clause | Signal-processing utilities |

### `[preprocessing]` — Extended audio preprocessing
| Component | License | Notes |
|-----------|---------|-------|
| [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) | MIT | EBU R128 loudness |
| [noisereduce](https://github.com/timsainb/noisereduce) | MIT | Spectral noise reduction |
| [SciPy](https://scipy.org/) | BSD-3-Clause | Filter design |

### `[observability]`
| Component | License | Notes |
|-----------|---------|-------|
| [prometheus-fastapi-instrumentator](https://github.com/trallnag/prometheus-fastapi-instrumentator) | ISC | `/metrics` endpoint |

### `[pdf]`
| Component | License | Notes |
|-----------|---------|-------|
| [WeasyPrint](https://github.com/Kozea/WeasyPrint) | BSD-3-Clause | HTML → PDF report rendering |

### `[dev]`
Development tooling only — never shipped to end users. Covers pytest,
ruff, mypy, pre-commit, hypothesis, pytest-cov, pytest-mock,
pytest-xdist. All carry permissive licenses (MIT / BSD / MPL-2.0).

## Datasets

ASRbench reads datasets via the HuggingFace `datasets` loader. Every
dataset carries its own license — a few highlights:

| Dataset | License | Notes |
|---------|---------|-------|
| LibriSpeech | CC-BY-4.0 | Public |
| FLEURS | CC-BY-4.0 | Public |
| Common Voice | CC0-1.0 | Public; HuggingFace access is **gated**, accept terms before download |
| TED-LIUM | CC-BY-NC-ND-3.0 | Non-commercial |
| Earnings22 | CC-BY-SA-4.0 | Research use |
| MediaSpeech | CC-BY-4.0 | Public |
| YODAS | CC-BY-4.0 | Public |

Always re-read the dataset card before publishing tuned outputs —
license interpretation is the user's responsibility, not ASRbench's.

## Binary tools (optional, detected via PATH)

| Tool | License | Notes |
|------|---------|-------|
| [FFmpeg](https://ffmpeg.org/) | LGPL-2.1 (default build) / GPL (with some enable flags) | Used by the `ffmpeg` preprocessing backend and codec simulation |

## Reporting license issues

If you believe a component is misattributed here, open an issue or
email sungurerdim@gmail.com. Fixes will be landed in the next release.
