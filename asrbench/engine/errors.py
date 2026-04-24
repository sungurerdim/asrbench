"""Error helpers — single source of truth for exception text persistence.

DB rows and HTTP responses must never carry Python tracebacks. A stack
trace leaks file paths, line numbers, and local-variable context that an
attacker can use to fingerprint the installation and plan further
probes; worse, the fields are user-visible in the optimiser / run UI, so
any secret that happens to flow through an exception message (HF token
in a URL, DB path on a shared filesystem) ends up on the dashboard.

Use :func:`sanitize_error` everywhere an exception is about to cross a
persistence or network boundary, and let the logging side pull the full
traceback via ``logger.exception(...)`` or ``exc_info=True`` so
operators still have everything they need in stderr.
"""

from __future__ import annotations

from typing import Final

__all__ = ["MAX_ERROR_MESSAGE_LENGTH", "sanitize_error"]


MAX_ERROR_MESSAGE_LENGTH: Final[int] = 500
"""Truncation budget for sanitised error strings.

The DB schema caps ``error_message`` at 4000 chars, but 500 is plenty
for ``Type: short human message`` and keeps the UI tidy. Longer
exception messages get an ellipsis suffix.
"""


def sanitize_error(exc: BaseException) -> str:
    """Return a short, traceback-free description of *exc*.

    Format: ``TypeName: message``. Any embedded newlines are collapsed
    to spaces so the string round-trips safely through DuckDB VARCHAR
    columns and JSON responses. Messages longer than
    :data:`MAX_ERROR_MESSAGE_LENGTH` are truncated with an ellipsis
    marker.
    """
    name = type(exc).__name__
    message = str(exc).replace("\n", " ").replace("\r", " ").strip()
    text = f"{name}: {message}" if message else name
    if len(text) > MAX_ERROR_MESSAGE_LENGTH:
        text = text[: MAX_ERROR_MESSAGE_LENGTH - 1].rstrip() + "…"
    return text
