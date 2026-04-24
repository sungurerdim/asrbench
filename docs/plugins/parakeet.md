# Parakeet backend

NVIDIA Parakeet is a production ASR family (TDT / CTC / RNN-T variants)
served through the NeMo toolkit. ASRbench ships Parakeet as the
`parakeet` backend, registered via an entry point under
`asrbench.backends` but requiring the optional `parakeet` extra at
install time.

## Install

```bash
pip install 'asrbench[parakeet]'
```

The extra pulls:

| Package | Version | License |
|---------|---------|---------|
| `nemo_toolkit[asr]` | `>=1.23,<2.0` | Apache-2.0 |
| `torch` | `>=2.1,<3.0` | BSD-3-Clause |
| `omegaconf` | `>=2.3,<3.0` | BSD-3-Clause |

The NeMo install is CUDA-coupled; follow the upstream
[NeMo install guide](https://github.com/NVIDIA/NeMo) for platform-specific
hints if the default wheel doesn't match your CUDA version.

## Licensing

* NeMo runtime: **Apache-2.0** — commercial use permitted.
* Parakeet checkpoints on NGC / HuggingFace: **CC-BY-4.0** — commercial
  use permitted with attribution. Double-check the model card before
  shipping the tuned output in a paid product.

## Example

Register a Parakeet model (HuggingFace IDs work directly; NeMo pulls
the weights on first load):

```bash
asrbench models register \
  --family parakeet \
  --name parakeet-tdt-0.6b \
  --backend parakeet \
  --local-path nvidia/parakeet-tdt-0.6b
```

Run a benchmark:

```bash
asrbench run \
  --backend parakeet \
  --model parakeet-tdt-0.6b \
  --dataset librispeech-dev-clean-100 \
  --segments 20
```

## Supported parameters

The IAMS optimizer can search over:

| Param | Type | Default | Notes |
|-------|------|---------|-------|
| `beam_size` | int | 1 | Beam width for TDT / RNN-T decoders. |
| `decoder_type` | str | `greedy` | `greedy`, `beam`, or `beam_maes`. |
| `batch_size` | int | 1 | Parallel transcribe batch. |
| `compute_type` | str | `float16` | `float32`, `float16`, `bfloat16`. |

## VRAM footprint

Rough per-model estimates that the `VRAMMonitor.require_capacity` guard
uses to refuse obviously-too-big loads:

| Model | Approx fp16 VRAM |
|-------|------------------|
| `parakeet-ctc-0.6b` | ~1.1 GB |
| `parakeet-tdt-0.6b` | ~1.5 GB |
| `parakeet-rnnt-0.6b` | ~1.5 GB |
| `parakeet-ctc-1.1b` | ~2.2 GB |
| `parakeet-tdt-1.1b` | ~2.8 GB |
| `parakeet-rnnt-1.1b` | ~2.8 GB |

Double the number for `compute_type=float32`.
