"""Unit tests for config loader."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_default_config_creates_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """get_config() creates ~/.asrbench/config.toml with defaults when absent."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    import asrbench.config as cfg_module
    from asrbench.config import get_config

    # Reset cache
    get_config.cache_clear()

    # Point config path to temp dir
    config_path = tmp_path / ".asrbench" / "config.toml"
    monkeypatch.setattr(cfg_module, "_default_config_path", lambda: config_path)

    config = get_config()
    get_config.cache_clear()

    assert config_path.exists(), "Config file should be created automatically"
    assert config.server.port == 8765
    assert config.server.host == "127.0.0.1"
    assert config.limits.max_concurrent_runs == 1
    assert config.limits.vram_warn_pct == 85.0


def test_custom_config_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """get_config() reads custom values from TOML file."""
    import asrbench.config as cfg_module
    from asrbench.config import get_config

    get_config.cache_clear()

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[server]\nhost = "0.0.0.0"\nport = 9999\nlog_level = "debug"\n'
        "[storage]\n"
        f'db_path = "{tmp_path / "bench.db"}"\n'
        f'cache_dir = "{tmp_path / "cache"}"\n'
        "[limits]\nmax_concurrent_runs = 4\nvram_warn_pct = 90\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(cfg_module, "_default_config_path", lambda: config_path)
    config = get_config()
    get_config.cache_clear()

    assert config.server.host == "0.0.0.0"
    assert config.server.port == 9999
    assert config.limits.max_concurrent_runs == 4
    assert config.limits.vram_warn_pct == 90.0


def test_storage_paths_expanded(tmp_path: Path) -> None:
    """StorageConfig expands ~ in paths and resolves them."""
    from asrbench.config import StorageConfig, get_config

    get_config.cache_clear()

    sc = StorageConfig(
        db_path=Path(f"{tmp_path}/benchmark.db"),
        cache_dir=Path(f"{tmp_path}/cache"),
    )
    assert sc.db_path.is_absolute()
    assert sc.cache_dir.is_absolute()
    assert sc.cache_dir.exists(), "cache_dir should be created automatically"
