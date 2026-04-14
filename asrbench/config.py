"""Application configuration with TOML file support and sensible defaults."""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CONFIG_PATH: Path = Path.home() / ".asrbench" / "config.toml"

_DEFAULTS_TOML = """\
[server]
host = "127.0.0.1"
port = 8765
log_level = "info"

[storage]
# db_path and cache_dir default to ~/.asrbench/ when omitted

[limits]
max_concurrent_runs = 1
vram_warn_pct = 85.0

[bench]
# Default interactive benchmark preferences (used by bench.bat).
# Omit a key to get the interactive prompt for that setting.
# lang = "tr"
# dataset = "common_voice"
# model = "large-v3"
# condition = "clean"
# max_duration_s = 3600
"""


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    log_level: str = "info"


@dataclass
class StorageConfig:
    db_path: Path = field(default_factory=lambda: Path.home() / ".asrbench" / "benchmark.db")
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".asrbench" / "cache")

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path).resolve()
        self.cache_dir = Path(self.cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class LimitsConfig:
    max_concurrent_runs: int = 1
    vram_warn_pct: float = 85.0


@dataclass
class BenchConfig:
    """Default preferences for the interactive bench.bat runner."""

    lang: str = ""
    dataset: str = ""
    model: str = ""
    condition: str = ""
    max_duration_s: float = 0


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)


def _escape_windows_paths(text: str) -> str:
    """Double backslashes inside double-quoted TOML values.

    TOML basic strings treat ``\\`` as escape sequences. Windows paths like
    ``C:\\Users\\...`` contain accidental escapes (``\\U``, ``\\b``, etc.).
    This replaces single backslashes with double backslashes only inside
    double-quoted string values, so ``tomllib`` sees literal paths.
    """
    import re

    def _double_backslashes(m: re.Match[str]) -> str:
        content = m.group(1)
        # Skip if already escaped or is a known TOML escape
        return '"' + content.replace("\\", "\\\\") + '"'

    return re.sub(r'"([^"]*)"', _double_backslashes, text)


def _parse_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file, returning an empty dict on failure."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    try:
        raw = path.read_text(encoding="utf-8")
        # On Windows, TOML values may contain backslash paths (e.g. C:\Users\...).
        # TOML treats \ as escape in basic strings. Pre-process: double backslashes
        # inside quoted values so tomllib parses them as literal paths.
        raw = _escape_windows_paths(raw)
        return tomllib.loads(raw)
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logger.warning("Failed to parse %s: %s — using defaults", path, exc)
        return {}


def _build_config(data: dict[str, Any]) -> Config:
    """Build a Config from a parsed TOML dict, falling back to defaults for missing keys."""
    srv = data.get("server", {})
    sto = data.get("storage", {})
    lim = data.get("limits", {})
    bnc = data.get("bench", {})

    server = ServerConfig(
        host=srv.get("host", "127.0.0.1"),
        port=int(srv.get("port", 8765)),
        log_level=srv.get("log_level", "info"),
    )

    storage_kwargs: dict[str, Any] = {}
    if "db_path" in sto:
        storage_kwargs["db_path"] = Path(sto["db_path"])
    if "cache_dir" in sto:
        storage_kwargs["cache_dir"] = Path(sto["cache_dir"])
    storage = StorageConfig(**storage_kwargs)

    limits = LimitsConfig(
        max_concurrent_runs=int(lim.get("max_concurrent_runs", 1)),
        vram_warn_pct=float(lim.get("vram_warn_pct", 85.0)),
    )

    bench = BenchConfig(
        lang=str(bnc.get("lang", "")),
        dataset=str(bnc.get("dataset", "")),
        model=str(bnc.get("model", "")),
        condition=str(bnc.get("condition", "")),
        max_duration_s=float(bnc.get("max_duration_s", 0)),
    )

    return Config(server=server, storage=storage, limits=limits, bench=bench)


@functools.lru_cache(maxsize=1)
def get_config() -> Config:
    """
    Read ~/.asrbench/config.toml, create with defaults if absent.

    The result is cached — call ``get_config.cache_clear()`` to force a re-read
    (useful in tests that redirect the config path via monkeypatch).
    """
    path = _CONFIG_PATH

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_DEFAULTS_TOML, encoding="utf-8")
        logger.info("Created default config at %s", path)

    data = _parse_toml(path)
    return _build_config(data)
