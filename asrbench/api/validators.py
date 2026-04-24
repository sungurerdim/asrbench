"""Reusable Pydantic validators for REST request payloads."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator

__all__ = ["LocalPath", "allowed_path_roots", "path_is_allowed"]


_DEFAULT_ALLOWED_ENV: str = "ASRBENCH_ALLOWED_PATHS"


def _default_allowed_roots() -> list[Path]:
    """Resolve the built-in allow-list for dataset/model ``local_path`` values.

    The cache dir under ``~/.asrbench`` is always allowed because ASRbench
    itself writes files there. Additional roots can be added via the
    ``ASRBENCH_ALLOWED_PATHS`` env var (OS path separator or ``,``) for
    operators who keep datasets on a mounted share.
    """
    roots: list[Path] = [Path.home() / ".asrbench"]

    env_value = os.environ.get(_DEFAULT_ALLOWED_ENV, "").strip()
    if env_value:
        # Accept both ``,`` and the OS-native path separator so
        # ``C:\data;D:\audio`` on Windows and ``/data:/audio`` on POSIX
        # both work without users learning a new syntax.
        for token in env_value.replace(os.pathsep, ",").split(","):
            token = token.strip()
            if not token:
                continue
            try:
                roots.append(Path(token).expanduser().resolve(strict=False))
            except (OSError, RuntimeError):
                continue

    return [r.resolve(strict=False) for r in roots]


def allowed_path_roots() -> list[Path]:
    """Public helper — returns the current allow-list (tests patch env)."""
    return _default_allowed_roots()


def path_is_allowed(candidate: Path, roots: list[Path] | None = None) -> bool:
    """Return True when ``candidate`` resolves under one of ``roots``.

    Uses ``Path.is_relative_to`` instead of string-prefix matching so
    ``/data2/audio`` is not accidentally treated as a child of
    ``/data``. ``resolve(strict=False)`` collapses ``..`` components
    and symlinks before the comparison — this is the step that
    defeats traversal attacks like ``~/.asrbench/../../etc/passwd``.
    """
    if roots is None:
        roots = allowed_path_roots()
    target = candidate.resolve(strict=False)
    for root in roots:
        try:
            if target.is_relative_to(root):
                return True
        except ValueError:
            continue
    return False


def _normalize_local_path(raw: str | None) -> str | None:
    """Expand, resolve, and whitelist-check a ``local_path`` request field.

    ASRbench registers model and dataset records before the file is actually
    downloaded, so validating existence would reject legitimate "register now,
    fetch later" flows. We normalise the path (``expanduser`` +
    ``resolve(strict=False)``) and reject:

    * strings with NUL or control characters (almost always malformed or
      malicious);
    * paths that resolve outside the allow-list. Without this check a
      remote client with a valid API key could register ``/etc/passwd``
      as a dataset and then issue ``DELETE /datasets/<id>?delete_files=true``
      to unlink it.

    The allow-list is ``~/.asrbench/`` plus whatever the operator added
    via ``ASRBENCH_ALLOWED_PATHS``.
    """
    if raw is None:
        return None

    if any(ord(ch) < 32 and ch not in ("\t",) for ch in raw) or "\x00" in raw:
        raise ValueError("local_path contains control characters.")

    try:
        path = Path(raw).expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"local_path is not a valid filesystem path: {exc}") from exc

    if not path_is_allowed(path):
        roots = ", ".join(str(r) for r in allowed_path_roots())
        raise ValueError(
            f"local_path {path!s} is outside the allowed roots ({roots}). "
            f"Set ASRBENCH_ALLOWED_PATHS to add additional roots."
        )

    return str(path)


LocalPath = Annotated[str, AfterValidator(_normalize_local_path)]
"""Pydantic annotated type for ``local_path`` request fields.

Usage::

    class MyRequest(BaseModel):
        local_path: LocalPath | None = None

Normalises user input (``~`` expansion, relative-path resolution) and
enforces that the resolved path sits under one of the allow-list roots
(``~/.asrbench`` plus anything in ``ASRBENCH_ALLOWED_PATHS``). Control
characters and NUL bytes are rejected outright.
"""
