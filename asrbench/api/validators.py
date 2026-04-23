"""Reusable Pydantic validators for REST request payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator

__all__ = ["LocalPath"]


def _normalize_local_path(raw: str | None) -> str | None:
    """Expand ``~`` and resolve relative parts without requiring the path to exist.

    ASRbench registers model and dataset records before the file is actually
    downloaded, so validating existence would reject legitimate "register now,
    fetch later" flows. We instead normalise the path (``expanduser`` +
    ``resolve(strict=False)``) and reject strings that contain NUL or control
    characters, which are almost always either accidental or malicious.
    """
    if raw is None:
        return None

    if any(ord(ch) < 32 and ch not in ("\t",) for ch in raw) or "\x00" in raw:
        raise ValueError("local_path contains control characters.")

    try:
        path = Path(raw).expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"local_path is not a valid filesystem path: {exc}") from exc

    return str(path)


LocalPath = Annotated[str, AfterValidator(_normalize_local_path)]
"""Pydantic annotated type for ``local_path`` request fields.

Usage::

    class MyRequest(BaseModel):
        local_path: LocalPath | None = None

Normalises user input (``~`` expansion, relative-path resolution) without
requiring the target to exist. Control characters and NUL bytes are
rejected because they indicate malformed input and cannot reach a real
path on either Windows or POSIX.
"""
