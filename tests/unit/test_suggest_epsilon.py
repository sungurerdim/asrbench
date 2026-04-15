"""Tests for suggest_epsilon — WER noise-floor calibrated epsilon."""

from __future__ import annotations

import math

import pytest

from asrbench.engine.search.significance import suggest_epsilon


def _se_formula(duration_s: float, *, wpm: float = 150.0, base_wer: float = 0.10) -> float:
    """Reference standard error for WER at ``duration_s`` — same math the suggester uses."""
    n_words = max(1.0, duration_s * wpm / 60.0)
    return math.sqrt(base_wer * (1.0 - base_wer) / n_words)


def test_suggest_epsilon_15_minutes() -> None:
    # 900 s × 150 wpm / 60 = 2250 words
    # SE ≈ sqrt(0.09 / 2250) ≈ 0.00632
    # 2× safety ≈ 0.0126 → round(3) = 0.013
    eps = suggest_epsilon(900)
    assert eps == pytest.approx(0.013, abs=0.001)


def test_suggest_epsilon_40_minutes() -> None:
    # 2400 s → 6000 words → SE ≈ 0.00387 → 2x ≈ 0.00775 → 0.008
    eps = suggest_epsilon(2400)
    assert eps == pytest.approx(0.008, abs=0.001)


def test_suggest_epsilon_safety_multiplier_scales_linearly() -> None:
    # safety=3 should be roughly 1.5× safety=2 (before rounding drift)
    eps_2 = suggest_epsilon(1800, safety=2.0)
    eps_3 = suggest_epsilon(1800, safety=3.0)
    assert eps_3 > eps_2
    # Ratio should be within rounding tolerance of 1.5
    assert 1.4 <= (eps_3 / eps_2) <= 1.6


def test_suggest_epsilon_noise_floor_matches_formula() -> None:
    # End-to-end sanity: suggest_epsilon(duration) should equal
    # round(2 * SE(duration), 3) for the default parameters.
    for duration in (300, 600, 900, 1800, 2400, 3600):
        expected = round(2.0 * _se_formula(duration), 3)
        assert suggest_epsilon(duration) == expected


def test_suggest_epsilon_zero_duration_clamped() -> None:
    # Guardrail: durations of 0 or less must not raise a div-by-zero
    eps = suggest_epsilon(0)
    assert eps > 0
