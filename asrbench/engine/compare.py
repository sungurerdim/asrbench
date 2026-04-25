"""CompareEngine — side-by-side comparison of two or more completed runs.

Given a set of runs (each with its aggregate row and parameter dict), emit:

- ``runs`` — the original run records plus ``is_baseline`` and
  ``delta_wer`` / ``delta_cer`` / ``delta_rtfx_mean`` relative to the baseline.
- ``params_diff`` — the subset of parameter keys whose value differs across
  at least two of the compared runs; the UI highlights these.
- ``params_same`` — the keys that are identical everywhere.
- ``wilcoxon_p`` — pairwise Wilcoxon signed-rank p-value on per-segment WER
  when exactly two runs are compared and both have segment rows; ``None``
  otherwise. A p-value below 0.05 means the two runs' per-segment WER
  distributions differ at the 5% significance level.

The engine is pure: it accepts the data it needs and does not read the DB
itself. Callers (the ``/runs/compare`` endpoint, the ``asrbench compare``
CLI subcommand) are responsible for fetching rows and wiring them in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["CompareEngine", "CompareInput", "CompareResult"]


@dataclass
class CompareInput:
    """One row of comparison input — everything the engine needs for a single run."""

    run_id: str
    params: dict[str, Any] = field(default_factory=dict)
    aggregate: dict[str, Any] = field(default_factory=dict)
    segment_wers: list[float] | None = None


@dataclass
class CompareResult:
    """Structured output of ``CompareEngine.compare``."""

    runs: list[dict[str, Any]]
    params_diff: list[str]
    params_same: list[str]
    wilcoxon_p: float | None = None


_DELTA_METRICS: tuple[str, ...] = ("wer_mean", "cer_mean", "mer_mean", "rtfx_mean", "rtfx_p95")


class CompareEngine:
    """Compute parameter diffs, per-run deltas, and pairwise significance."""

    def compare(
        self,
        runs: list[CompareInput],
        *,
        baseline_run_id: str | None = None,
    ) -> CompareResult:
        """Compare the provided runs. Returns a ``CompareResult``.

        Args:
            runs: at least two input rows.
            baseline_run_id: run_id to treat as baseline. If not given, the
                first row is used.

        Raises:
            ValueError: fewer than two runs provided, or ``baseline_run_id``
                does not match any input.
        """
        if len(runs) < 2:
            raise ValueError(f"CompareEngine.compare requires at least 2 runs; got {len(runs)}.")

        baseline_index = self._resolve_baseline_index(runs, baseline_run_id)
        baseline = runs[baseline_index]

        params_diff, params_same = self._diff_params([r.params for r in runs])

        enriched: list[dict[str, Any]] = []
        for idx, run in enumerate(runs):
            row = {
                "run_id": run.run_id,
                "params": dict(run.params),
                "aggregate": dict(run.aggregate),
                "is_baseline": idx == baseline_index,
            }
            for metric in _DELTA_METRICS:
                row[f"delta_{metric}"] = self._delta(
                    run.aggregate.get(metric), baseline.aggregate.get(metric)
                )
            enriched.append(row)

        wilcoxon_p: float | None = None
        if len(runs) == 2:
            wilcoxon_p = self._wilcoxon(runs[0].segment_wers, runs[1].segment_wers)

        return CompareResult(
            runs=enriched,
            params_diff=params_diff,
            params_same=params_same,
            wilcoxon_p=wilcoxon_p,
        )

    @staticmethod
    def _resolve_baseline_index(runs: list[CompareInput], baseline_run_id: str | None) -> int:
        if baseline_run_id is None:
            return 0
        for i, run in enumerate(runs):
            if run.run_id == baseline_run_id:
                return i
        raise ValueError(
            f"baseline_run_id '{baseline_run_id}' does not match any of the "
            f"provided runs ({', '.join(r.run_id for r in runs)})."
        )

    @staticmethod
    def _diff_params(
        param_sets: list[dict[str, Any]],
    ) -> tuple[list[str], list[str]]:
        """Split the union of keys into ones that differ vs ones that are identical."""
        all_keys: set[str] = set()
        for p in param_sets:
            all_keys.update(p.keys())

        diff: list[str] = []
        same: list[str] = []
        for key in sorted(all_keys):
            first = param_sets[0].get(key)
            if all(p.get(key) == first for p in param_sets[1:]):
                same.append(key)
            else:
                diff.append(key)
        return diff, same

    @staticmethod
    def _delta(value: Any, baseline_value: Any) -> float | None:
        if value is None or baseline_value is None:
            return None
        try:
            return float(value) - float(baseline_value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _wilcoxon(a: list[float] | None, b: list[float] | None) -> float | None:
        """Paired Wilcoxon signed-rank test on two per-segment WER series.

        Returns None when scipy is unavailable, either series is missing, the
        series have different lengths, or the test itself fails (e.g. all
        zero differences).
        """
        if not a or not b or len(a) != len(b):
            return None
        try:
            from scipy.stats import wilcoxon
        except ImportError:
            logger.debug("scipy not installed — Wilcoxon comparison skipped")
            return None
        try:
            result = wilcoxon(a, b, zero_method="wilcox")
            # scipy.stats.wilcoxon returns a namedtuple; second field is the p-value.
            return float(result[1])
        except Exception as exc:
            logger.debug("Wilcoxon test failed: %s", exc)
            return None
