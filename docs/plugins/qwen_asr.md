# Qwen-ASR plugin

Qwen2-Audio and the downstream Qwen-ASR variants run on HuggingFace
Transformers + PyTorch. Both dependencies are large (PyTorch alone is
~2 GB with CUDA) and their version coupling varies by GPU, so
ASRbench ships Qwen-ASR as a third-party plugin rather than a core
backend. The plugin wiring is identical to
[Parakeet](./parakeet.md) — this file only shows the class body, since
the rest of the scaffolding (`pyproject.toml`, entry point registration)
is mechanically the same.

## Backend implementation

```python
# src/asrbench_qwen/backend.py
from __future__ import annotations

import numpy as np

from asrbench.backends.base import BaseBackend, Segment


class QwenASRBackend(BaseBackend):
    family = "qwen"
    name = "qwen-asr"

    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self._device: str = "cuda"

    def default_params(self) -> dict:
        return {"max_new_tokens": 128, "do_sample": False}

    def load(self, model_path: str, params: dict) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
        ).to(self._device)

    def unload(self) -> None:
        self._processor = None
        self._model = None

    def transcribe(self, audio: np.ndarray, lang: str, params: dict) -> list[Segment]:
        if self._model is None or self._processor is None:
            raise RuntimeError("QwenASRBackend.load() must be called first.")
        import torch

        inputs = self._processor(
            audios=audio, sampling_rate=16000, return_tensors="pt"
        ).to(self._device)
        with torch.inference_mode():
            ids = self._model.generate(**inputs, **params)
        text = self._processor.batch_decode(ids, skip_special_tokens=True)[0]
        return [
            Segment(
                offset_s=0.0,
                duration_s=len(audio) / 16000.0,
                ref_text="",
                hyp_text=text,
            )
        ]
```

## `pyproject.toml` entry point

```toml
[project.entry-points."asrbench.backends"]
qwen-asr = "asrbench_qwen.backend:QwenASRBackend"
```

Install the plugin into the same environment as asrbench
(`pip install ./asrbench-qwen`) and `asrbench doctor` will list it as an
available backend.

## Testing advice

- `importorskip("transformers")` and `importorskip("torch")` so CI
  hosts without these libraries can still collect the tests.
- Pin `transformers` to a version that matches the Qwen model card you
  target — tokenizer and model class layouts change between majors.
- Qwen models usually expect 16 kHz mono float32 input, which is what
  the ASRbench pipeline already produces; no extra resampling needed.
