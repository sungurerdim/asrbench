"""Unit tests for the error-sanitisation helper (Faz 1.4)."""

from __future__ import annotations

from asrbench.engine.errors import MAX_ERROR_MESSAGE_LENGTH, sanitize_error


def test_plain_exception_formatted_as_type_and_message() -> None:
    err = ValueError("bad thing happened")
    assert sanitize_error(err) == "ValueError: bad thing happened"


def test_empty_message_returns_bare_type_name() -> None:
    err = RuntimeError()
    assert sanitize_error(err) == "RuntimeError"


def test_embedded_newlines_collapsed() -> None:
    err = RuntimeError("line 1\nline 2\r\nline 3")
    result = sanitize_error(err)
    assert "\n" not in result
    assert "\r" not in result
    assert "line 1" in result and "line 3" in result


def test_traceback_text_is_not_leaked() -> None:
    """A traceback-shaped message must NOT pass through verbatim."""
    text = (
        "Traceback (most recent call last):\n"
        '  File "/home/user/secret.py", line 42, in x\n'
        '    raise RuntimeError("boom")\n'
        "RuntimeError: boom"
    )
    err = RuntimeError(text)
    result = sanitize_error(err)
    assert "\n" not in result
    # Length cap guarantees no massive dump
    assert len(result) <= MAX_ERROR_MESSAGE_LENGTH


def test_long_message_truncated_with_ellipsis() -> None:
    err = RuntimeError("x" * (MAX_ERROR_MESSAGE_LENGTH + 200))
    result = sanitize_error(err)
    assert len(result) == MAX_ERROR_MESSAGE_LENGTH
    assert result.endswith("…")


def test_custom_exception_subclass_name_preserved() -> None:
    class TransientModelError(RuntimeError):
        pass

    err = TransientModelError("model load timed out")
    assert sanitize_error(err) == "TransientModelError: model load timed out"
