"""``asrbench doctor`` — quick environment sanity check."""

from __future__ import annotations

import importlib
import importlib.metadata as im
import shutil
import socket
import sys
from dataclasses import dataclass
from pathlib import Path

import typer

app = typer.Typer(help="Report whether the environment is healthy.")


@dataclass(frozen=True)
class Check:
    status: str  # OK | WARN | FAIL
    label: str
    detail: str = ""


_STATUS_STYLES = {
    "OK": ("green", "OK"),
    "WARN": ("yellow", "WARN"),
    "FAIL": ("red", "FAIL"),
}


def _color(text: str, color: str) -> str:
    # typer.echo supports colour via typer.style; wrap for readability.
    return typer.style(text, fg=color, bold=True)


def _check_python() -> Check:
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 11):
        return Check("OK", "Python", f"{major}.{minor}.{sys.version_info[2]}")
    return Check(
        "FAIL",
        "Python",
        f"{major}.{minor} detected; ASRbench requires 3.11+.",
    )


def _check_backend(name: str, install_hint: str) -> Check:
    try:
        importlib.import_module(name)
    except ImportError:
        return Check("WARN", f"backend: {name}", f"not installed — {install_hint}")
    try:
        version = im.version(name.replace("_", "-"))
    except im.PackageNotFoundError:
        version = "unknown"
    return Check("OK", f"backend: {name}", f"v{version}")


def _check_binary(name: str, install_hint: str) -> Check:
    if shutil.which(name):
        return Check("OK", name, "on PATH")
    return Check("WARN", name, f"not found — {install_hint}")


def _check_port(port: int) -> Check:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        s.close()
        return Check("OK", f"port {port}", "free")
    except OSError:
        return Check("WARN", f"port {port}", "in use — pass --port to `asrbench serve`")


def _check_hf_cache() -> Check:
    import os

    path = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    if path.exists():
        # du-style estimate without walking — report existence only.
        return Check("OK", "HuggingFace cache", str(path))
    return Check("OK", "HuggingFace cache", f"{path} (will be created on first fetch)")


def _check_hf_token() -> Check:
    import os

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return Check("OK", "HF_TOKEN", f"set (len={len(token)})")
    return Check(
        "WARN",
        "HF_TOKEN",
        "unset — gated datasets (Common Voice) will fail. "
        "Create at https://huggingface.co/settings/tokens.",
    )


def _check_cuda_env() -> Check:
    import os

    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return Check("OK", "CUDA_VISIBLE_DEVICES", "unset (default = every GPU visible)")
    return Check("OK", "CUDA_VISIBLE_DEVICES", value)


def _check_asrbench_home() -> Check:
    home = Path.home() / ".asrbench"
    if not home.exists():
        return Check("OK", "~/.asrbench", "will be created on first run")
    try:
        probe = home / ".doctor-write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return Check("FAIL", "~/.asrbench", f"not writable: {exc}")
    return Check("OK", "~/.asrbench", f"writable ({home})")


def _check_disk_space(path: Path, minimum_gb: float = 10.0) -> Check:
    try:
        usage = shutil.disk_usage(str(path if path.exists() else path.parent))
    except OSError as exc:
        return Check("WARN", "disk space", f"{path}: {exc}")
    free_gb = usage.free / (1024**3)
    if free_gb < minimum_gb:
        return Check(
            "WARN",
            "disk space",
            f"{free_gb:.1f} GB free at {path} (benchmarks need ≥ {minimum_gb:.0f} GB)",
        )
    return Check("OK", "disk space", f"{free_gb:.1f} GB free at {path}")


def _check_duckdb_version() -> Check:
    try:
        version = im.version("duckdb")
    except im.PackageNotFoundError:
        return Check("FAIL", "DuckDB", "not installed")
    return Check("OK", "DuckDB", f"v{version}")


def _check_vram() -> Check:
    try:
        from asrbench.engine.vram import get_vram_monitor

        snap = get_vram_monitor().snapshot()
    except Exception as exc:
        return Check("WARN", "GPU / VRAM", f"query failed: {exc}")
    if not snap.available:
        return Check(
            "WARN",
            "GPU / VRAM",
            "no NVIDIA GPU visible — CPU-only backends (whisper.cpp) still work.",
        )
    return Check("OK", "GPU / VRAM", f"{snap.used_mb:.0f} / {snap.total_mb:.0f} MB used")


def _run_all_checks() -> list[Check]:
    return [
        _check_python(),
        _check_duckdb_version(),
        _check_asrbench_home(),
        _check_disk_space(Path.home() / ".asrbench"),
        _check_binary(
            "ffmpeg",
            "install via `winget install ffmpeg` / `apt install ffmpeg` / `brew install ffmpeg`",
        ),
        _check_backend("faster_whisper", "pip install 'asrbench[faster-whisper]'"),
        _check_backend("pywhispercpp", "pip install 'asrbench[whisper-cpp]'"),
        _check_backend(
            "nemo",
            "pip install 'asrbench[parakeet]' (NVIDIA Parakeet via NeMo)",
        ),
        _check_backend(
            "transformers",
            "pip install 'asrbench[qwen]' (Qwen2-Audio; review Qwen Community License first)",
        ),
        _check_backend(
            "trnorm",
            "pip install 'asrbench[tr]' (Turkish normalizer)",
        ),
        _check_hf_cache(),
        _check_hf_token(),
        _check_cuda_env(),
        _check_vram(),
        _check_port(8765),
    ]


@app.callback(invoke_without_command=True)
def doctor(
    ctx: typer.Context,
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON instead of the human table."
    ),
) -> None:
    """Print a coloured status table; exits non-zero when any FAIL is observed."""
    if ctx.invoked_subcommand:
        return

    checks = _run_all_checks()

    if json_output:
        import json

        typer.echo(
            json.dumps(
                [{"status": c.status, "label": c.label, "detail": c.detail} for c in checks],
                indent=2,
            )
        )
    else:
        width = max(len(c.label) for c in checks)
        for c in checks:
            color, tag = _STATUS_STYLES.get(c.status, ("white", c.status))
            typer.echo(f"[{_color(tag, color)}] {c.label.ljust(width)}  {c.detail}")

    if any(c.status == "FAIL" for c in checks):
        raise typer.Exit(code=1)
