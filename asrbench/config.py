"""Application configuration with TOML file support and sensible defaults.

Defaults live in a single ``_DEFAULTS`` mapping — dataclass field defaults, the
fallback values used when a TOML key is missing, and the initial config file
written to disk all read from it. Adding a new setting only requires updating
one place.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)


def _default_asrbench_home() -> Path:
    """Resolve the asrbench home directory at call time.

    The path is re-derived from ``Path.home()`` on every call so tests can
    redirect the home directory via ``monkeypatch.setenv("HOME", ...)`` after
    module import.
    """
    return Path.home() / ".asrbench"


def _default_config_path() -> Path:
    return _default_asrbench_home() / "config.toml"


def _default_db_path() -> Path:
    return _default_asrbench_home() / "benchmark.db"


def _default_cache_dir() -> Path:
    return _default_asrbench_home() / "cache"


_DEFAULTS: Final[dict[str, dict[str, Any]]] = {
    "server": {
        "host": "127.0.0.1",
        "port": 8765,
        "log_level": "info",
    },
    "limits": {
        "max_concurrent_runs": 1,
        "vram_warn_pct": 85.0,
    },
    "bench": {
        "lang": "",
        "dataset": "",
        "model": "",
        "condition": "",
        "max_duration_s": 0.0,
    },
}


def _render_defaults_toml() -> str:
    """Render ``_DEFAULTS`` as TOML so the on-disk config matches runtime defaults."""
    lines: list[str] = []
    lines.append("[server]")
    for key, val in _DEFAULTS["server"].items():
        lines.append(f"{key} = {_toml_value(val)}")
    lines.append("")
    lines.append("[storage]")
    lines.append("# db_path and cache_dir default to ~/.asrbench/ when omitted")
    lines.append("")
    lines.append("[limits]")
    for key, val in _DEFAULTS["limits"].items():
        lines.append(f"{key} = {_toml_value(val)}")
    lines.append("")
    lines.append("[bench]")
    lines.append("# Default preferences for `asrbench bench`.")
    lines.append("# Omit a key to get the interactive prompt for that setting.")
    for key, val in _DEFAULTS["bench"].items():
        lines.append(f"# {key} = {_toml_value(val)}")
    lines.append("")
    return "\n".join(lines)


def _toml_value(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, str):
        return f'"{val}"'
    return str(val)


_DEFAULTS_TOML: Final[str] = _render_defaults_toml()


@dataclass
class ServerConfig:
    host: str = _DEFAULTS["server"]["host"]
    port: int = _DEFAULTS["server"]["port"]
    log_level: str = _DEFAULTS["server"]["log_level"]


@dataclass
class StorageConfig:
    db_path: Path = field(default_factory=_default_db_path)
    cache_dir: Path = field(default_factory=_default_cache_dir)

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path).resolve()
        self.cache_dir = Path(self.cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class LimitsConfig:
    max_concurrent_runs: int = _DEFAULTS["limits"]["max_concurrent_runs"]
    vram_warn_pct: float = _DEFAULTS["limits"]["vram_warn_pct"]


@dataclass
class BenchConfig:
    """Default preferences for `asrbench bench` interactive runs."""

    lang: str = _DEFAULTS["bench"]["lang"]
    dataset: str = _DEFAULTS["bench"]["dataset"]
    model: str = _DEFAULTS["bench"]["model"]
    condition: str = _DEFAULTS["bench"]["condition"]
    max_duration_s: float = _DEFAULTS["bench"]["max_duration_s"]


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
        raw = _escape_windows_paths(raw)
        return tomllib.loads(raw)
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logger.warning("Failed to parse %s: %s — using defaults", path, exc)
        return {}


def _build_config(data: dict[str, Any]) -> Config:
    """Build a Config from a parsed TOML dict, falling back to ``_DEFAULTS`` for missing keys."""
    srv = data.get("server", {})
    sto = data.get("storage", {})
    lim = data.get("limits", {})
    bnc = data.get("bench", {})

    server = ServerConfig(
        host=str(srv.get("host", _DEFAULTS["server"]["host"])),
        port=int(srv.get("port", _DEFAULTS["server"]["port"])),
        log_level=str(srv.get("log_level", _DEFAULTS["server"]["log_level"])),
    )

    storage_kwargs: dict[str, Any] = {}
    if "db_path" in sto:
        storage_kwargs["db_path"] = Path(sto["db_path"])
    if "cache_dir" in sto:
        storage_kwargs["cache_dir"] = Path(sto["cache_dir"])
    storage = StorageConfig(**storage_kwargs)

    limits = LimitsConfig(
        max_concurrent_runs=int(
            lim.get("max_concurrent_runs", _DEFAULTS["limits"]["max_concurrent_runs"])
        ),
        vram_warn_pct=float(lim.get("vram_warn_pct", _DEFAULTS["limits"]["vram_warn_pct"])),
    )

    bench = BenchConfig(
        lang=str(bnc.get("lang", _DEFAULTS["bench"]["lang"])),
        dataset=str(bnc.get("dataset", _DEFAULTS["bench"]["dataset"])),
        model=str(bnc.get("model", _DEFAULTS["bench"]["model"])),
        condition=str(bnc.get("condition", _DEFAULTS["bench"]["condition"])),
        max_duration_s=float(bnc.get("max_duration_s", _DEFAULTS["bench"]["max_duration_s"])),
    )

    return Config(server=server, storage=storage, limits=limits, bench=bench)


@functools.lru_cache(maxsize=1)
def get_config() -> Config:
    """Read ``~/.asrbench/config.toml``, creating it from defaults if absent.

    The result is cached — call ``get_config.cache_clear()`` to force a re-read
    (useful in tests that redirect the config path via monkeypatch).
    """
    path = _default_config_path()

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_DEFAULTS_TOML, encoding="utf-8")
        logger.info("Created default config at %s", path)

    data = _parse_toml(path)
    return _build_config(data)
