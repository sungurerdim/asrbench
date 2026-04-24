"""Unit tests for the LocalPath whitelist validator (Faz 1.3)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from asrbench.api.validators import LocalPath, allowed_path_roots, path_is_allowed


class _Model(BaseModel):
    path: LocalPath


def test_path_under_asrbench_home_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("ASRBENCH_ALLOWED_PATHS", raising=False)

    target = tmp_path / ".asrbench" / "cache" / "models" / "tiny"
    m = _Model(path=str(target))
    assert m.path == str(target.resolve())


def test_path_outside_home_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("ASRBENCH_ALLOWED_PATHS", raising=False)

    with pytest.raises(ValidationError) as excinfo:
        _Model(path="/etc/passwd")
    assert "outside the allowed roots" in str(excinfo.value)


def test_allowed_paths_env_extends_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extra = tmp_path / "custom-data"
    extra.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("ASRBENCH_ALLOWED_PATHS", str(extra))

    m = _Model(path=str(extra / "a.wav"))
    assert m.path == str((extra / "a.wav").resolve())


def test_multiple_allowed_paths_comma_separated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_a = tmp_path / "share-a"
    root_b = tmp_path / "share-b"
    root_a.mkdir()
    root_b.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("ASRBENCH_ALLOWED_PATHS", f"{root_a},{root_b}")

    _Model(path=str(root_a / "x"))
    _Model(path=str(root_b / "y"))

    with pytest.raises(ValidationError):
        _Model(path=str(tmp_path / "share-c" / "z"))


def test_traversal_attack_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``~/.asrbench/../../etc/passwd`` must not slip through."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("ASRBENCH_ALLOWED_PATHS", raising=False)

    attack = str(tmp_path / ".asrbench" / ".." / ".." / "etc" / "passwd")
    with pytest.raises(ValidationError):
        _Model(path=attack)


def test_control_characters_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    with pytest.raises(ValidationError) as excinfo:
        _Model(path=str(tmp_path / ".asrbench") + "\x00/evil")
    assert "control characters" in str(excinfo.value)


def test_none_passes_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional LocalPath fields (None) must not trigger validation."""
    from pydantic import BaseModel

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    class Maybe(BaseModel):
        path: LocalPath | None = None

    m = Maybe()
    assert m.path is None


def test_allowed_path_roots_reflects_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("ASRBENCH_ALLOWED_PATHS", str(tmp_path / "extra"))
    roots = allowed_path_roots()
    assert (tmp_path / ".asrbench").resolve() in roots
    assert (tmp_path / "extra").resolve() in roots


def test_path_is_allowed_boolean_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("ASRBENCH_ALLOWED_PATHS", raising=False)

    assert path_is_allowed(tmp_path / ".asrbench" / "x") is True
    assert path_is_allowed(Path("/etc/passwd")) is False
