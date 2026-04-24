# Qwen-Audio backend

Qwen2-Audio is Alibaba's multimodal speech+text LLM. ASRbench uses it
for transcription via a fixed chat template that constrains the model
to emit nothing but the verbatim transcript. The backend ships as the
`qwen_asr` entry point but requires the `qwen` extra to install its
heavyweight deps.

## ⚠ License warning

Qwen2-Audio is released under the **Qwen Community License** (not an
OSI-approved open license). Key restrictions:

* Commercial use is permitted only under monthly-active-user caps
  spelled out in the license text.
* You must include the full license and a notice of modifications
  when redistributing weights or derived products.
* The license terms evolve with the model family — re-read the
  upstream file for your specific checkpoint before shipping.

Read the full text at
<https://github.com/QwenLM/Qwen2-Audio/blob/main/LICENSE>.

ASRbench makes this explicit:
* The `qwen` extra is **opt-in** — a default install carries no Qwen
  code.
* `asrbench doctor` flags the `transformers` check with the same
  warning when the extra is installed.
* This page and `THIRD-PARTY-LICENSES.md` surface the restriction at
  every entry point into the backend.

## Install

```bash
pip install 'asrbench[qwen]'
```

| Package | Version | License |
|---------|---------|---------|
| `transformers` | `>=4.45,<5.0` | Apache-2.0 |
| `torch` | `>=2.1,<3.0` | BSD-3-Clause |
| `accelerate` | `>=0.30,<2.0` | Apache-2.0 |
| `librosa` | `>=0.10,<1.0` | ISC |

Model weights themselves are distributed under the Qwen Community License
— not through the Python extra.

## Example

```bash
asrbench models register \
  --family qwen \
  --name qwen2-audio-7b-instruct \
  --backend qwen_asr \
  --local-path Qwen/Qwen2-Audio-7B-Instruct

asrbench run \
  --backend qwen_asr \
  --model qwen2-audio-7b-instruct \
  --dataset librispeech-dev-clean-100 \
  --segments 10
```

## Prompting

The backend hardcodes a system prompt that disallows commentary, then
assembles a single user turn containing an audio block plus the
`Transcribe this audio.` instruction. This keeps the model's sampled
output comparable to a plain ASR system.

## Supported parameters

| Param | Type | Default | Notes |
|-------|------|---------|-------|
| `temperature` | float | `0.0` | Only flows through when `do_sample=True`. |
| `top_p` | float | `1.0` | Only flows through when `do_sample=True`. |
| `max_new_tokens` | int | `512` | Ceiling for the generated transcript. |
| `do_sample` | bool | `False` | Set True for stochastic decoding experiments. |
| `compute_type` | str | `bfloat16` | `float32`, `float16`, `bfloat16`. |

## VRAM footprint

`Qwen2-Audio-7B` / `Qwen2-Audio-7B-Instruct`: ~16 GB at fp16/bf16,
~32 GB at fp32. The `VRAMMonitor.require_capacity` guard refuses to
load when the GPU cannot fit the estimate + 10 % margin.
