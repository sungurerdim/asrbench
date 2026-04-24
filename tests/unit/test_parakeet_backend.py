"""Unit tests for the NeMo-backed Parakeet adapter (Faz 6.1).

NeMo is a heavy optional dep — these tests mock it entirely via
``sys.modules`` so they run in the base dev install.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture()
def mock_nemo(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Inject a fake ``nemo.collections.asr.models.ASRModel`` into sys.modules."""
    # Build the dotted module structure nemo.collections.asr.models
    nemo = types.ModuleType("nemo")
    collections = types.ModuleType("nemo.collections")
    asr = types.ModuleType("nemo.collections.asr")
    models = types.ModuleType("nemo.collections.asr.models")

    asr_model = MagicMock(name="ASRModel")
    fake_instance = MagicMock(name="asr_model_instance")
    fake_instance.transcribe.return_value = ["hello world"]
    # _move_to_device_and_cast calls .to(...) twice; make the chain
    # return the same mock so `.transcribe()` stays attached.
    fake_instance.to.return_value = fake_instance
    asr_model.from_pretrained.return_value = fake_instance
    models.ASRModel = asr_model

    monkeypatch.setitem(sys.modules, "nemo", nemo)
    monkeypatch.setitem(sys.modules, "nemo.collections", collections)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", asr)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", models)
    return asr_model


@pytest.fixture()
def mock_omegaconf(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    omegaconf = types.ModuleType("omegaconf")
    omegaconf.OmegaConf = MagicMock(name="OmegaConf")
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf)
    return omegaconf.OmegaConf


@pytest.fixture()
def patch_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the VRAM monitor to report plenty of headroom for happy paths."""
    import asrbench.engine.vram as vram_mod
    from asrbench.engine.vram import VRAMMonitor, VRAMSnapshot

    fake = VRAMMonitor()
    fake.snapshot = lambda: VRAMSnapshot(  # type: ignore[method-assign]
        available=True, used_mb=0, total_mb=32_000
    )
    monkeypatch.setattr(vram_mod, "get_vram_monitor", lambda: fake)


class TestParakeetMissingDep:
    def test_load_without_nemo_raises_install_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A crisp RuntimeError when NeMo isn't installed."""
        # Make sure `import nemo...` fails even if the dev machine has it.
        for key in list(sys.modules):
            if key.startswith("nemo"):
                monkeypatch.delitem(sys.modules, key, raising=False)

        original_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def _blocked_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name.startswith("nemo"):
                raise ImportError("nemo blocked for test")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _blocked_import)

        from asrbench.backends.parakeet import ParakeetBackend

        backend = ParakeetBackend()
        with pytest.raises(RuntimeError, match="asrbench\\[parakeet\\]"):
            backend.load("nvidia/parakeet-tdt-0.6b", {})


class TestParakeetHappyPath:
    def test_load_then_transcribe_then_unload(
        self, mock_nemo: MagicMock, mock_omegaconf: MagicMock, patch_vram: None
    ) -> None:
        del mock_omegaconf, patch_vram
        from asrbench.backends.parakeet import ParakeetBackend

        backend = ParakeetBackend()
        backend.load("nvidia/parakeet-tdt-0.6b", {"compute_type": "float16"})

        # NeMo's ASRModel.from_pretrained got the right model name.
        mock_nemo.from_pretrained.assert_called_once_with(model_name="nvidia/parakeet-tdt-0.6b")

        audio = np.zeros(16_000, dtype=np.float32)
        segments = backend.transcribe(audio, "en", {"beam_size": 4})
        assert len(segments) == 1
        assert segments[0].hyp_text == "hello world"
        assert segments[0].duration_s == pytest.approx(1.0)

        backend.unload()

    def test_transcribe_without_load_raises(self) -> None:
        from asrbench.backends.parakeet import ParakeetBackend

        backend = ParakeetBackend()
        audio = np.zeros(16_000, dtype=np.float32)
        with pytest.raises(RuntimeError, match="before load"):
            backend.transcribe(audio, "en", {})

    def test_default_params_exposes_searchable_knobs(self) -> None:
        from asrbench.backends.parakeet import SEARCHABLE_PARAMS, ParakeetBackend

        backend = ParakeetBackend()
        defaults = backend.default_params()
        for key in ("beam_size", "decoder_type", "batch_size", "compute_type"):
            assert key in defaults
        assert backend.supported_params() == SEARCHABLE_PARAMS

    def test_extract_hypothesis_handles_list_form(self) -> None:
        from asrbench.backends.parakeet import _extract_hypothesis

        assert _extract_hypothesis(["plain string"]) == "plain string"
        assert _extract_hypothesis([["nested"]]) == "nested"
        obj = MagicMock()
        obj.text = "object form"
        assert _extract_hypothesis([obj]) == "object form"
        assert _extract_hypothesis([]) == ""


class TestParakeetVRAMGuard:
    def test_load_respects_vram_estimate(
        self,
        mock_nemo: MagicMock,
        mock_omegaconf: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the VRAM monitor refuses, load must raise ResourceExhausted."""
        del mock_nemo, mock_omegaconf
        import asrbench.engine.vram as vram_mod
        from asrbench.backends.parakeet import ParakeetBackend
        from asrbench.engine.vram import ResourceExhausted, VRAMMonitor, VRAMSnapshot

        tight = VRAMMonitor()
        tight.snapshot = lambda: VRAMSnapshot(  # type: ignore[method-assign]
            available=True, used_mb=15_500, total_mb=16_000
        )
        monkeypatch.setattr(vram_mod, "get_vram_monitor", lambda: tight)

        backend = ParakeetBackend()
        with pytest.raises(ResourceExhausted):
            backend.load("nvidia/parakeet-tdt-1.1b", {"compute_type": "float32"})
