# Parakeet plugin

NVIDIA's Parakeet family is not shipped with the core `asrbench`
package because NeMo Toolkit pulls in >2 GB of ASR dependencies and
couples tightly to the host CUDA runtime. This guide shows how to
publish Parakeet as a third-party plugin that ASRbench discovers via
`importlib.metadata` entry points.

## 1. Project layout

```
asrbench-parakeet/
  pyproject.toml
  src/asrbench_parakeet/__init__.py
  src/asrbench_parakeet/backend.py
```

## 2. `pyproject.toml`

```toml
[project]
name = "asrbench-parakeet"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "asrbench>=0.2",
    "nemo_toolkit[asr]>=2.2,<3.0",
]

[project.entry-points."asrbench.backends"]
parakeet = "asrbench_parakeet.backend:ParakeetBackend"
```

## 3. Backend implementation

```python
# src/asrbench_parakeet/backend.py
from __future__ import annotations

from typing import Any

import numpy as np

from asrbench.backends.base import BaseBackend, Segment


class ParakeetBackend(BaseBackend):
    family = "parakeet"
    name = "parakeet"

    def __init__(self) -> None:
        self._model: Any = None

    def default_params(self) -> dict:
        return {"decode_mode": "rnnt"}

    def load(self, model_path: str, params: dict) -> None:
        import nemo.collections.asr as nemo_asr

        self._model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_path)

    def unload(self) -> None:
        self._model = None

    def transcribe(
        self, audio: np.ndarray, lang: str, params: dict
    ) -> list[Segment]:
        if self._model is None:
            raise RuntimeError("ParakeetBackend.load() must be called first.")
        # NeMo expects a list of audio signals or file paths. Keep the
        # implementation minimal — real drivers add VAD, chunking, etc.
        texts = self._model.transcribe([audio])
        return [Segment(offset_s=0.0, duration_s=len(audio) / 16000.0,
                        ref_text="", hyp_text=texts[0])]
```

## 4. Install + register

```bash
pip install ./asrbench-parakeet
asrbench doctor        # should now list "parakeet" as an installed backend
```

Once installed, ASRbench picks the plugin up automatically — no core
code change required. Any REST call, CLI subcommand, or optimizer study
that accepts a `backend` name can now use `parakeet`.

## 5. Testing advice

- Mirror `tests/unit/test_faster_whisper_backend.py` for the backend
  contract tests.
- Use `importorskip("nemo.collections.asr")` so CI can collect without
  NeMo installed.
- Pin `nemo_toolkit` to the version you actually validated against —
  the NeMo API moves between minors.
