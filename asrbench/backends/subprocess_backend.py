"""CUDA process isolation — run any BaseBackend in a separate OS process.

Each load/transcribe/unload cycle runs in a dedicated child process so that
CUDA contexts are fully cleaned up between different model configurations.
This prevents GPU memory leaks that occur when multiple models are loaded
sequentially in the same process.

IPC: multiprocessing.Pipe (pickle-based). Start method: "spawn" (required
for CUDA fork-safety on Linux, default on Windows/macOS).
"""

from __future__ import annotations

import gc
import importlib.metadata
import logging
import multiprocessing
import traceback
from multiprocessing.connection import Connection
from typing import Any

import numpy as np

from asrbench.backends.base import BaseBackend, Segment

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 600


def _resolve_backend_cls(backend_name: str) -> type[BaseBackend]:
    """
    Resolve a backend class from entry-points registered under ``asrbench.backends``.

    Raises:
        RuntimeError: if the backend name is not found in entry-points.
    """
    eps = importlib.metadata.entry_points(group="asrbench.backends")
    for ep in eps:
        if ep.name == backend_name:
            return ep.load()  # type: ignore[return-value]
    available = ", ".join(ep.name for ep in eps)
    raise RuntimeError(
        f"Backend '{backend_name}' not found in entry-points. "
        f"Available: {available or '(none — install a backend package)'}. "
        f"Install it: pip install 'asrbench[{backend_name}]'"
    )


def _worker(backend_name: str, child_conn: Connection) -> None:
    """
    Child process event loop.

    Protocol:
        ("load", model_path, params)              → ("ok",)
        ("transcribe", audio_bytes, lang, params)  → ("result", segments_data)
        ("unload",)                                → ("ok",) then exit
        ("default_params",)                        → ("result", params_dict)

    On error: ("error", traceback_string)
    """
    backend: BaseBackend | None = None

    try:
        backend_cls = _resolve_backend_cls(backend_name)
        backend = backend_cls()

        while True:
            msg = child_conn.recv()
            cmd = msg[0]

            try:
                if cmd == "load":
                    _, model_path, params = msg
                    backend.load(model_path, params)
                    child_conn.send(("ok",))

                elif cmd == "transcribe":
                    _, audio_bytes, lang, params = msg
                    audio = np.frombuffer(audio_bytes, dtype=np.float32).copy()
                    segments = backend.transcribe(audio, lang, params)
                    seg_data = [
                        {
                            "offset_s": s.offset_s,
                            "duration_s": s.duration_s,
                            "ref_text": s.ref_text,
                            "hyp_text": s.hyp_text,
                        }
                        for s in segments
                    ]
                    child_conn.send(("result", seg_data))

                elif cmd == "unload":
                    backend.unload()
                    child_conn.send(("ok",))
                    break

                elif cmd == "default_params":
                    child_conn.send(("result", backend.default_params()))

                else:
                    child_conn.send(("error", f"Unknown command: {cmd}"))

            except Exception:
                child_conn.send(("error", traceback.format_exc()))

    except Exception:
        try:
            child_conn.send(("error", traceback.format_exc()))
        except Exception:
            pass
    finally:
        # Aggressive cleanup
        if backend is not None:
            try:
                backend.unload()
            except Exception:
                pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        child_conn.close()


class SubprocessBackend(BaseBackend):
    """
    BaseBackend wrapper: load/transcribe/unload all run in a child process.

    Each unload() terminates the child → CUDA context fully released.
    A new load() spawns a fresh child process.

    Usage::

        backend = SubprocessBackend("faster-whisper")
        backend.load(model_path, params)
        segments = backend.transcribe(audio, "en", params)
        backend.unload()
    """

    def __init__(self, backend_name: str, *, timeout_s: float = _DEFAULT_TIMEOUT_S) -> None:
        self.family = ""
        self.name = f"subprocess:{backend_name}"
        self._backend_name = backend_name
        self._timeout_s = timeout_s
        self._proc: Any = None
        self._parent_conn: Any = None

    def default_params(self) -> dict:
        """Resolve the inner backend class and return its default_params."""
        cls = _resolve_backend_cls(self._backend_name)
        return cls().default_params()

    def load(self, model_path: str, params: dict) -> None:
        """Spawn a child process, send load command, wait for ack."""
        if self._proc is not None:
            self.unload()

        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)

        proc = ctx.Process(
            target=_worker,
            args=(self._backend_name, child_conn),
            daemon=True,
        )
        proc.start()
        child_conn.close()  # parent doesn't use child end

        self._proc = proc
        self._parent_conn = parent_conn

        self._send(("load", model_path, params))
        self._recv("load")

    def transcribe(self, audio: np.ndarray, lang: str, params: dict) -> list[Segment]:
        """Send audio to child, receive list[Segment]."""
        if self._parent_conn is None or self._proc is None:
            raise RuntimeError(
                "SubprocessBackend.transcribe() called before load(). "
                "Call load(model_path, params) first."
            )

        audio_bytes = audio.astype(np.float32).tobytes()
        self._send(("transcribe", audio_bytes, lang, params))
        resp = self._recv("transcribe")

        return [
            Segment(
                offset_s=s["offset_s"],
                duration_s=s["duration_s"],
                ref_text=s["ref_text"],
                hyp_text=s["hyp_text"],
            )
            for s in resp
        ]

    def unload(self) -> None:
        """Send unload command, wait for child to exit, cleanup."""
        if self._parent_conn is None or self._proc is None:
            return

        try:
            self._send(("unload",))
            self._recv("unload")
        except (OSError, EOFError, RuntimeError):
            pass

        self._cleanup()

    def _send(self, msg: tuple[Any, ...]) -> None:
        assert self._parent_conn is not None
        self._parent_conn.send(msg)

    def _recv(self, context: str) -> Any:
        assert self._parent_conn is not None
        if not self._parent_conn.poll(self._timeout_s):
            self._cleanup()
            raise TimeoutError(
                f"SubprocessBackend: '{context}' timed out after {self._timeout_s}s. "
                "The child process was killed. Increase timeout_s if this is expected."
            )

        resp = self._parent_conn.recv()
        if resp[0] == "error":
            raise RuntimeError(
                f"SubprocessBackend: child process error during '{context}':\n{resp[1]}"
            )
        if resp[0] == "result":
            return resp[1]
        return None

    def _cleanup(self) -> None:
        """Terminate child process and close connection."""
        if self._proc is not None:
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=5)
                if self._proc.is_alive():
                    self._proc.kill()
                    self._proc.join(timeout=2)
            self._proc = None

        if self._parent_conn is not None:
            try:
                self._parent_conn.close()
            except OSError:
                pass
            self._parent_conn = None
