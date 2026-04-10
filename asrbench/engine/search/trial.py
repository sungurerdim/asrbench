"""
Trial abstractions for the IAMS optimizer.

A Trial is one evaluation of a candidate parameter configuration. The optimizer
sees the world through `TrialExecutor`, a narrow interface returning metrics
and CI for a given config. This decouples the search algorithm from:

    1. Whether the execution is a real BenchmarkEngine run (expensive, slow)
    2. Whether it's a synthetic math function (cheap, deterministic, for tests)
    3. Whether results come from a cache or a fresh run

The production executor (BenchmarkTrialExecutor, defined in a separate module)
wraps BenchmarkEngine. This file provides:

    - `TrialResult`: typed container for a single trial's outcome
    - `TrialExecutor`: Protocol describing the executor interface
    - `SyntheticTrialExecutor`: deterministic math-function executor for unit
      and integration tests, plus reproducibility-critical debugging

The synthetic executor takes a user-supplied callable `f(config) -> metrics_dict`
and an optional noise seed. It is the workhorse of the entire test suite for
the search layers — every algorithm can be validated against known optima.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class TrialResult:
    """
    One trial's complete outcome — used by every search layer.

    Fields:
        config        : The parameter values that were evaluated
        metrics       : Raw metrics dict (wer, cer, rtfx_mean, vram_peak_mb, ...)
                        plus any CI fields (wer_ci_lower, wer_ci_upper, ...)
        score         : Objective.score(metrics) — lower is better
        score_ci      : (lower, upper) — Objective.score_ci(metrics)
        phase         : IAMS layer that produced this trial, for logging/study.json
        reasoning     : Human-readable explanation of why this config was chosen
        trial_id      : Optional stable identifier (may be None for synthetic trials)
    """

    config: Mapping[str, Any]
    metrics: Mapping[str, float | None]
    score: float
    score_ci: tuple[float, float]
    phase: str = "unknown"
    reasoning: str = ""
    trial_id: str | None = None

    def config_key(self) -> str:
        """
        Deterministic key for config identity — used for result caching.

        Two TrialResults with the same config produce the same key, regardless
        of field order in the dict.
        """
        return str(hash(tuple(sorted(self.config.items()))))

    def with_phase(self, phase: str, reasoning: str = "") -> TrialResult:
        """Return a copy tagged with a different phase/reasoning (immutable update)."""
        return TrialResult(
            config=self.config,
            metrics=self.metrics,
            score=self.score,
            score_ci=self.score_ci,
            phase=phase,
            reasoning=reasoning or self.reasoning,
            trial_id=self.trial_id,
        )


@runtime_checkable
class TrialExecutor(Protocol):
    """
    Narrow contract: given a config, return a TrialResult.

    The executor is responsible for:
        - Running the underlying process (benchmark or synthetic function)
        - Calling Objective.score() + Objective.score_ci() on the raw metrics
        - Populating phase and reasoning (optional, caller may override)

    Implementations MUST be:
        - Deterministic when the underlying process is deterministic
        - Idempotent w.r.t. cache: repeated calls with the same config may
          return cached results or re-run, at the executor's discretion

    Exceptions:
        - Runtime failures should raise — the search layer decides whether to
          skip, retry, or abort. Do NOT return NaN or None as a "failure marker".
    """

    def evaluate(
        self,
        config: Mapping[str, Any],
        *,
        phase: str = "unknown",
        reasoning: str = "",
    ) -> TrialResult: ...

    @property
    def runs_used(self) -> int:
        """Total evaluate() calls (including cache hits, if exposed as a run)."""
        ...


@dataclass
class SyntheticTrialExecutor:
    """
    Deterministic, cheap trial executor for tests.

    Construction takes a user callable `metric_fn(config) -> metrics_dict` that
    must return at least the keys the Objective cares about. Optional Gaussian
    noise can be injected (seeded) to simulate measurement variability — the
    noise seed is derived per-config so the same config always gets the same
    noisy result (idempotent under seed).

    Usage:
        def landscape(cfg):
            # A simple quadratic in beam_size with a minimum at 7
            w = 0.1 + 0.005 * (cfg["beam_size"] - 7) ** 2
            return {"wer": w, "cer": w * 0.5, "rtfx_mean": 20.0,
                    "vram_peak_mb": 4000.0, "wer_ci_lower": w - 0.002,
                    "wer_ci_upper": w + 0.002}
        exec = SyntheticTrialExecutor(metric_fn=landscape, objective=obj)
        tr = exec.evaluate({"beam_size": 5})

    The executor also maintains an internal counter of evaluations (`runs_used`)
    so budget controllers can drive against it directly in tests.
    """

    metric_fn: Callable[[Mapping[str, Any]], Mapping[str, float]]
    objective: Any  # Objective, but annotating avoids circular import
    noise_std: float = 0.0
    seed: int = 42
    _runs_used: int = field(default=0, init=False)
    _cache: dict[str, TrialResult] = field(default_factory=dict, init=False)
    _cache_enabled: bool = True

    @property
    def runs_used(self) -> int:
        return self._runs_used

    def set_cache_enabled(self, enabled: bool) -> None:
        """
        Toggle config-level caching.

        When enabled (default), repeated evaluate() calls with the same config
        return the same cached TrialResult without re-invoking metric_fn or
        incrementing runs_used. Disable it in tests that want to count every
        call (e.g., budget controller tests).
        """
        self._cache_enabled = enabled

    def evaluate(
        self,
        config: Mapping[str, Any],
        *,
        phase: str = "unknown",
        reasoning: str = "",
    ) -> TrialResult:
        key = self._config_key(config)
        if self._cache_enabled and key in self._cache:
            cached = self._cache[key]
            return cached.with_phase(phase, reasoning)

        raw = dict(self.metric_fn(dict(config)))

        if self.noise_std > 0:
            # Per-config seeded noise: same config → same noisy result always
            config_hash = int(hashlib.sha1(key.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(self.seed ^ config_hash)
            noise = float(rng.normal(0.0, self.noise_std))
            if "wer" in raw:
                raw["wer"] = max(0.0, float(raw["wer"]) + noise)
                # Scale CI proportionally so downstream significance gate stays consistent
                if "wer_ci_lower" in raw and "wer_ci_upper" in raw:
                    shift = noise
                    raw["wer_ci_lower"] = max(0.0, float(raw["wer_ci_lower"]) + shift)
                    raw["wer_ci_upper"] = max(0.0, float(raw["wer_ci_upper"]) + shift)

        score = self.objective.score(raw)
        ci = self.objective.score_ci(raw)

        result = TrialResult(
            config=dict(config),
            metrics=raw,
            score=score,
            score_ci=ci,
            phase=phase,
            reasoning=reasoning,
            trial_id=f"synthetic-{key}",
        )
        self._runs_used += 1
        if self._cache_enabled:
            self._cache[key] = result
        return result

    @staticmethod
    def _config_key(config: Mapping[str, Any]) -> str:
        return str(hash(tuple(sorted(config.items()))))
