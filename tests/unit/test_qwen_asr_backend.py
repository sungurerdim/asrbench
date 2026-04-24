"""Unit tests for the Qwen-Audio adapter (Faz 6.2).

transformers + torch are heavy optional deps; these tests mock them via
``sys.modules`` so the adapter can be exercised on the base dev install.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture()
def mock_transformers(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    transformers = types.ModuleType("transformers")

    processor_cls = MagicMock(name="AutoProcessor")
    model_cls = MagicMock(name="AutoModelForCausalLM")

    processor_instance = MagicMock(name="processor_instance")
    processor_instance.apply_chat_template.return_value = "<|chat|>...prompt..."
    processor_instance.return_value = {
        "input_ids": _FakeTensor([[1, 2, 3]]),
        "attention_mask": _FakeTensor([[1, 1, 1]]),
    }
    processor_instance.batch_decode.return_value = ["the transcript"]

    model_instance = MagicMock(name="model_instance")
    model_instance.generate.return_value = _FakeTensor([[1, 2, 3, 10, 11, 12]])

    processor_cls.from_pretrained.return_value = processor_instance
    model_cls.from_pretrained.return_value = model_instance

    transformers.AutoProcessor = processor_cls
    transformers.AutoModelForCausalLM = model_cls
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return processor_cls, model_cls


@pytest.fixture()
def patch_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the VRAM monitor to report plenty of headroom for happy paths."""
    import asrbench.engine.vram as vram_mod
    from asrbench.engine.vram import VRAMMonitor, VRAMSnapshot

    fake = VRAMMonitor()
    fake.snapshot = lambda: VRAMSnapshot(  # type: ignore[method-assign]
        available=True, used_mb=0, total_mb=80_000
    )
    monkeypatch.setattr(vram_mod, "get_vram_monitor", lambda: fake)


@pytest.fixture()
def mock_torch(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.float16 = "float16"
    torch.bfloat16 = "bfloat16"

    cuda = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
    )
    torch.cuda = cuda

    class _InferenceMode:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    torch.inference_mode = lambda: _InferenceMode()
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch  # type: ignore[return-value]


class _FakeTensor:
    """Minimal stand-in for torch.Tensor supporting only the ops we use."""

    def __init__(self, data: list) -> None:
        self._data = data
        self.shape = (len(data), len(data[0]) if data else 0)

    def to(self, *_args, **_kwargs) -> _FakeTensor:
        return self

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return _FakeTensor([row[index[1]] for row in self._data])
        return self._data[index]


class TestQwenMissingDep:
    def test_load_without_transformers_raises_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for key in list(sys.modules):
            if key.startswith("transformers"):
                monkeypatch.delitem(sys.modules, key, raising=False)

        original_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def _blocked_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name.startswith("transformers"):
                raise ImportError("transformers blocked for test")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _blocked_import)

        from asrbench.backends.qwen_asr import QwenASRBackend

        backend = QwenASRBackend()
        with pytest.raises(RuntimeError, match="asrbench\\[qwen\\]"):
            backend.load("Qwen/Qwen2-Audio-7B-Instruct", {})


class TestQwenHappyPath:
    def test_load_transcribe_unload(
        self,
        mock_transformers: tuple[MagicMock, MagicMock],
        mock_torch: MagicMock,
        patch_vram: None,
    ) -> None:
        del mock_torch, patch_vram
        processor_cls, model_cls = mock_transformers
        from asrbench.backends.qwen_asr import QwenASRBackend

        backend = QwenASRBackend()
        backend.load("Qwen/Qwen2-Audio-7B-Instruct", {"compute_type": "bfloat16"})

        processor_cls.from_pretrained.assert_called_once()
        model_cls.from_pretrained.assert_called_once()

        audio = np.zeros(16_000, dtype=np.float32)
        segments = backend.transcribe(audio, "en", {"max_new_tokens": 128})
        assert len(segments) == 1
        assert segments[0].hyp_text == "the transcript"
        assert segments[0].duration_s == pytest.approx(1.0)

        backend.unload()

    def test_transcribe_without_load_raises(self) -> None:
        from asrbench.backends.qwen_asr import QwenASRBackend

        backend = QwenASRBackend()
        audio = np.zeros(16_000, dtype=np.float32)
        with pytest.raises(RuntimeError, match="before load"):
            backend.transcribe(audio, "en", {})

    def test_default_params_exposes_searchable_knobs(self) -> None:
        from asrbench.backends.qwen_asr import SEARCHABLE_PARAMS, QwenASRBackend

        backend = QwenASRBackend()
        defaults = backend.default_params()
        for key in ("temperature", "top_p", "max_new_tokens", "do_sample", "compute_type"):
            assert key in defaults
        assert backend.supported_params() == SEARCHABLE_PARAMS

    def test_generation_kwargs_drops_sampling_noise_when_deterministic(self) -> None:
        from asrbench.backends.qwen_asr import QwenASRBackend

        backend = QwenASRBackend()
        deterministic = backend._generation_kwargs(
            {"do_sample": False, "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 64}
        )
        assert deterministic["do_sample"] is False
        assert "temperature" not in deterministic  # pruned
        assert "top_p" not in deterministic

        stochastic = backend._generation_kwargs(
            {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 64}
        )
        assert stochastic["temperature"] == pytest.approx(0.7)
        assert stochastic["top_p"] == pytest.approx(0.9)


class TestQwenVRAMGuard:
    def test_oversized_model_refused(
        self,
        mock_transformers: tuple[MagicMock, MagicMock],
        mock_torch: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        del mock_transformers, mock_torch
        import asrbench.engine.vram as vram_mod
        from asrbench.backends.qwen_asr import QwenASRBackend
        from asrbench.engine.vram import ResourceExhausted, VRAMMonitor, VRAMSnapshot

        tight = VRAMMonitor()
        tight.snapshot = lambda: VRAMSnapshot(  # type: ignore[method-assign]
            available=True, used_mb=7_000, total_mb=8_000
        )
        monkeypatch.setattr(vram_mod, "get_vram_monitor", lambda: tight)

        backend = QwenASRBackend()
        with pytest.raises(ResourceExhausted):
            backend.load("Qwen/Qwen2-Audio-7B-Instruct", {"compute_type": "float16"})
