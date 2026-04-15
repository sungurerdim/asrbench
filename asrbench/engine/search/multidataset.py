"""
MultiDatasetTrialExecutor — "find one config that works across N datasets".

Use case:
    IAMS's default is "per-study" — one study per (model, dataset, language,
    condition) — because the optimal preprocessing + decoding config can
    differ between, say, clean TR and noisy EN. In a mobile product that
    ships a single preset to every user, this granularity is wasted: you
    need ONE config that holds up across the whole matrix.

Design:
    Wrap N ``TrialExecutor`` instances, each bound to a different
    ``PreparedDataset``. Each ``evaluate(config)`` call runs the config on
    every inner executor, then combines their ``TrialResult`` scores into
    a single weighted aggregate score. The CI interval is propagated via
    variance-weighted combination so downstream significance gates still
    fire on true differences rather than cross-dataset noise.

    IAMS sees this as a single executor and runs its 7-layer algorithm on
    it, producing one best config. Per-dataset trial rows are still stored
    inside each inner executor (e.g. the ``optimization_trials`` table),
    so the per-dataset breakdown stays auditable — only the IAMS-level
    score is aggregated.

Weighting:
    Weights default to the dataset durations in seconds. Longer corpora
    get more say in the aggregate, which mirrors how a real deployment
    user's WER would be measured. Callers can pass explicit weights
    (e.g. uniform, or "TR prioritized") for scenario-specific optimization.

Significance:
    With weighted WER aggregation the CI width is NOT simply the
    max/mean of component CIs. We combine variances under the assumption
    that each dataset's error is independent — a reasonable approximation
    for this use case. The result is a slightly tighter CI than any one
    dataset alone, which is correct: with N times more observations we
    should be more confident.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from asrbench.engine.search.trial import TrialExecutor, TrialResult


@dataclass
class MultiDatasetTrialExecutor:
    """
    Aggregate N TrialExecutors into one "global config" trial executor.

    Parameters
    ----------
    executors:
        Ordered list of inner executors, one per dataset.
    weights:
        Optional per-executor weights. When omitted, uniform 1/N weights
        are used. Weights are normalized to sum to 1.
    labels:
        Optional per-executor labels; surfaced in the aggregate
        ``reasoning`` string so the audit trail names which datasets
        contributed to each trial.
    """

    executors: Sequence[TrialExecutor]
    weights: Sequence[float] | None = None
    labels: Sequence[str] | None = None
    _runs_used: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.executors:
            raise ValueError("MultiDatasetTrialExecutor requires at least one inner executor.")
        n = len(self.executors)
        if self.weights is None:
            self.weights = [1.0 / n] * n
        else:
            if len(self.weights) != n:
                raise ValueError(
                    f"weights length {len(self.weights)} does not match executor count {n}."
                )
            total = float(sum(self.weights))
            if total <= 0:
                raise ValueError("Sum of weights must be positive.")
            self.weights = [float(w) / total for w in self.weights]
        if self.labels is None:
            self.labels = [f"ds{i}" for i in range(n)]
        elif len(self.labels) != n:
            raise ValueError(f"labels length {len(self.labels)} does not match executor count {n}.")

    @property
    def runs_used(self) -> int:
        return self._runs_used

    def evaluate(
        self,
        config: Mapping[str, Any],
        *,
        phase: str = "unknown",
        reasoning: str = "",
    ) -> TrialResult:
        """
        Evaluate *config* on every inner executor and aggregate the results.

        Returns a single synthetic TrialResult whose ``score`` is the
        weighted mean of component scores, ``score_ci`` is the
        variance-weighted CI, and ``metrics`` is the union of component
        metrics (prefixed with the dataset label).
        """
        components: list[tuple[str, TrialResult]] = []
        for label, executor in zip(self.labels or [], self.executors):  # type: ignore[arg-type]
            component_reasoning = reasoning or phase
            trial = executor.evaluate(
                config,
                phase=phase,
                reasoning=f"[{label}] {component_reasoning}",
            )
            components.append((label, trial))

        weights = list(self.weights or [])
        aggregate_score = sum(w * t.score for w, (_, t) in zip(weights, components))

        # Variance-weighted CI: treat each dataset's CI half-width as a
        # standard-error proxy and combine under independence.
        lo_var = 0.0
        hi_var = 0.0
        for w, (_, t) in zip(weights, components):
            half = max(1e-12, (t.score_ci[1] - t.score_ci[0]) / 2.0)
            # Halve the half-width twice (once to SE, once to combined),
            # then rebuild the 95% CI under Gaussian assumption.
            # Variance of weighted sum: sum(w_i^2 * var_i).
            se = half / 1.96
            lo_var += (w * se) ** 2
            hi_var += (w * se) ** 2
        combined_se = (lo_var + hi_var) ** 0.5 / (2**0.5) if lo_var + hi_var > 0 else 0.0
        half_combined = 1.96 * combined_se
        aggregate_ci = (aggregate_score - half_combined, aggregate_score + half_combined)

        # Merge metrics with per-dataset prefix so downstream reporting
        # keeps the breakdown. Also surface a pooled "wer" and "wer_ci_*"
        # at the top level so IAMS objectives can still score the trial
        # without being dataset-aware.
        merged_metrics: dict[str, float | None] = {}
        for label, t in components:
            for key, val in t.metrics.items():
                merged_metrics[f"{label}.{key}"] = val
        merged_metrics["wer"] = aggregate_score
        merged_metrics["wer_ci_lower"] = aggregate_ci[0]
        merged_metrics["wer_ci_upper"] = aggregate_ci[1]

        breakdown = ", ".join(f"{lbl}={t.score:.4f}" for lbl, t in components)
        aggregated_reasoning = (
            f"{reasoning} [aggregate={aggregate_score:.4f}; {breakdown}]"
            if reasoning
            else f"[aggregate={aggregate_score:.4f}; {breakdown}]"
        )

        self._runs_used += 1

        return TrialResult(
            config=dict(config),
            metrics=merged_metrics,
            score=aggregate_score,
            score_ci=aggregate_ci,
            phase=phase,
            reasoning=aggregated_reasoning,
            trial_id=None,
            pruned=any(getattr(t, "pruned", False) for _, t in components),
        )

    def evaluate_at_fraction(
        self,
        config: Mapping[str, Any],
        *,
        phase: str = "unknown",
        reasoning: str = "",
        fraction: float = 1.0,
    ) -> TrialResult:
        """
        Multi-fidelity hook: forward ``fraction`` to every inner executor
        that supports it, fall back to ``evaluate`` for the rest. Enables
        chaining with ``MultiFidelityTrialExecutor`` at the outer level.
        """
        components: list[tuple[str, TrialResult]] = []
        for label, executor in zip(self.labels or [], self.executors):  # type: ignore[arg-type]
            component_reasoning = reasoning or phase
            if hasattr(executor, "evaluate_at_fraction"):
                trial = executor.evaluate_at_fraction(  # type: ignore[attr-defined]
                    config,
                    phase=phase,
                    reasoning=f"[{label}] {component_reasoning}",
                    fraction=fraction,
                )
            else:
                trial = executor.evaluate(
                    config,
                    phase=phase,
                    reasoning=f"[{label}] {component_reasoning}",
                )
            components.append((label, trial))
        return self._aggregate(components, config, phase, reasoning)

    def _aggregate(
        self,
        components: list[tuple[str, TrialResult]],
        config: Mapping[str, Any],
        phase: str,
        reasoning: str,
    ) -> TrialResult:
        """Shared aggregation helper used by both evaluate paths."""
        weights = list(self.weights or [])
        aggregate_score = sum(w * t.score for w, (_, t) in zip(weights, components))
        lo_var = 0.0
        hi_var = 0.0
        for w, (_, t) in zip(weights, components):
            half = max(1e-12, (t.score_ci[1] - t.score_ci[0]) / 2.0)
            se = half / 1.96
            lo_var += (w * se) ** 2
            hi_var += (w * se) ** 2
        combined_se = (lo_var + hi_var) ** 0.5 / (2**0.5) if lo_var + hi_var > 0 else 0.0
        half_combined = 1.96 * combined_se
        aggregate_ci = (aggregate_score - half_combined, aggregate_score + half_combined)

        merged_metrics: dict[str, float | None] = {}
        for label, t in components:
            for key, val in t.metrics.items():
                merged_metrics[f"{label}.{key}"] = val
        merged_metrics["wer"] = aggregate_score
        merged_metrics["wer_ci_lower"] = aggregate_ci[0]
        merged_metrics["wer_ci_upper"] = aggregate_ci[1]

        breakdown = ", ".join(f"{lbl}={t.score:.4f}" for lbl, t in components)
        aggregated_reasoning = (
            f"{reasoning} [aggregate={aggregate_score:.4f}; {breakdown}]"
            if reasoning
            else f"[aggregate={aggregate_score:.4f}; {breakdown}]"
        )
        self._runs_used += 1
        return TrialResult(
            config=dict(config),
            metrics=merged_metrics,
            score=aggregate_score,
            score_ci=aggregate_ci,
            phase=phase,
            reasoning=aggregated_reasoning,
            trial_id=None,
            pruned=any(getattr(t, "pruned", False) for _, t in components),
        )
