"""
IAMS Layer 3 — pairwise 2D grid scan for interaction detection.

Coordinate descent (Layer 2) is blind to two-parameter interactions:
a config like `(vad=true, chunk_length=30)` that dominates both univariate
minima but requires BOTH settings together is invisible to sequential
one-at-a-time optimization. Layer 3 catches these by running full 3×3
grids on the top-K most sensitive parameter pairs.

For each pair (i, j) of top-K sensitive parameters:

    - Enumerate 3 probe values for each (min, default, max).
    - Run the 9 configs with every OTHER parameter held at the current baseline.
    - Check the 9 scores for off-diagonal minima (configs that are better
      than both univariate minima predict).
    - Emit "promising points" — off-diagonal cells significantly below
      the best univariate prediction — to Layer 4 (multi-start).

Interaction score per pair:
    I_{ij} = max(abs_deviation_from_additive)
    where abs_deviation is the difference between the observed score and the
    additive-model prediction score_base + Δ_i_alone + Δ_j_alone.

A pair with large |I_{ij}| is a sign that the two parameters interact. The
pair with the biggest interaction is the most important one to restart from.

Probe overlap elimination:
    The baseline trial and univariate boundary trials from Layer 1 are already
    in the executor's cache. A 3×3 grid that shares those points (the two-on-
    one-axis probes, e.g. (beam=min, temp=default)) reuses them automatically
    via the TrialExecutor's config-level cache — there are only 4 genuinely
    new evaluations per pair, not 9.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from asrbench.engine.search.budget import BudgetController
from asrbench.engine.search.space import ParameterSpace, ParamSpec
from asrbench.engine.search.trial import TrialExecutor, TrialResult


@dataclass
class PairwiseGrid:
    """Per-pair output of Layer 3."""

    param_a: str
    param_b: str
    trials: list[TrialResult] = field(default_factory=list)
    interaction_score: float = 0.0
    promising_points: list[TrialResult] = field(default_factory=list)

    def best(self) -> TrialResult:
        return min(self.trials, key=lambda t: t.score)


@dataclass
class PairwiseGridResult:
    """Aggregated result over all scanned pairs."""

    grids: list[PairwiseGrid] = field(default_factory=list)
    all_trials: list[TrialResult] = field(default_factory=list)

    def best_overall(self) -> TrialResult | None:
        if not self.all_trials:
            return None
        return min(self.all_trials, key=lambda t: t.score)

    def promising_points(self) -> list[TrialResult]:
        """Flat list of all promising off-diagonal candidates across grids."""
        out: list[TrialResult] = []
        for g in self.grids:
            out.extend(g.promising_points)
        return out

    def pair_by_interaction(self) -> list[PairwiseGrid]:
        """Grids sorted by descending interaction strength."""
        return sorted(self.grids, key=lambda g: -g.interaction_score)


class PairwiseGridScan:
    """
    Layer 3 orchestrator.

    Args:
        executor: TrialExecutor (synthetic or real)
        space: ParameterSpace — used to resolve parameter specs
        budget: BudgetController — must have capacity for the grid
        top_k: maximum number of most-sensitive parameters to include in pairs.
               The total pair count is top_k * (top_k - 1) / 2.
        eps_min: significance threshold used when classifying promising points.
        baseline_score: current best score so far (from Layer 2) — used as the
                        reference for "promising" classification.

    Call:
        scan = PairwiseGridScan(executor, space, budget, top_k=4, eps_min=0.005,
                                baseline_config={...}, baseline_score=0.08)
        result = scan.run(sensitive_params=["beam", "temp", "vad", "chunk"])
    """

    def __init__(
        self,
        executor: TrialExecutor,
        space: ParameterSpace,
        budget: BudgetController,
        *,
        top_k: int = 4,
        eps_min: float = 0.005,
        baseline_config: dict[str, Any] | None = None,
        baseline_score: float | None = None,
    ) -> None:
        if top_k < 2:
            raise ValueError(f"PairwiseGridScan: top_k must be >= 2 to form pairs, got {top_k}")
        self.executor = executor
        self.space = space
        self.budget = budget
        self.top_k = top_k
        self.eps_min = eps_min
        self.baseline_config = baseline_config or space.defaults()
        self.baseline_score = baseline_score

    def run(self, sensitive_params: list[str]) -> PairwiseGridResult:
        """
        Run 3×3 grids on all pairs from the top-K most sensitive parameters.

        sensitive_params must already be sorted by descending sensitivity
        (as produced by ScreeningResult.sensitive_order).
        """
        selected = sensitive_params[: self.top_k]
        result = PairwiseGridResult()

        for name_a, name_b in combinations(selected, 2):
            if not self.budget.can_run():
                break
            spec_a = self.space.get(name_a)
            spec_b = self.space.get(name_b)
            grid = self._scan_pair(spec_a, spec_b)
            result.grids.append(grid)

        # Merge all trial lists once at the end instead of per-pair extend
        for g in result.grids:
            result.all_trials.extend(g.trials)

        return result

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _scan_pair(self, spec_a: ParamSpec, spec_b: ParamSpec) -> PairwiseGrid:
        probes_a = self._three_probes(spec_a)
        probes_b = self._three_probes(spec_b)

        grid = PairwiseGrid(param_a=spec_a.name, param_b=spec_b.name)
        score_grid: dict[tuple[Any, Any], float] = {}

        for va in probes_a:
            for vb in probes_b:
                if not self.budget.can_run():
                    break
                config = dict(self.baseline_config)
                config[spec_a.name] = spec_a.clamp(va)
                config[spec_b.name] = spec_b.clamp(vb)
                reasoning = f"pairwise grid ({spec_a.name}={va}, {spec_b.name}={vb})"
                trial = self.executor.evaluate(config, phase="pairwise_grid", reasoning=reasoning)
                self.budget.record(trial.score)
                grid.trials.append(trial)
                score_grid[(va, vb)] = trial.score

        grid.interaction_score = self._interaction_score(
            score_grid, spec_a, spec_b, probes_a, probes_b
        )
        grid.promising_points = self._find_promising(
            grid.trials, score_grid, spec_a, spec_b, probes_a, probes_b
        )
        return grid

    @staticmethod
    def _three_probes(spec: ParamSpec) -> list[Any]:
        """
        Pick the three probe points for one axis of the 2D grid.

        - float/int: (min, default, max)
        - bool: (False, True, default) → deduplicated to 2 values
        - enum: first, middle, last → deduplicated
        """
        if spec.type in ("float", "int"):
            return [spec.min, spec.default, spec.max]
        if spec.type == "bool":
            return [False, True]  # only two values, caller handles 2×2 grid
        if spec.type == "enum":
            assert spec.values is not None
            values = list(spec.values)
            if len(values) == 1:
                return values
            if len(values) == 2:
                return values
            # First, middle, last
            return [values[0], values[len(values) // 2], values[-1]]
        raise ValueError(f"Cannot probe unknown param type {spec.type!r}")

    def _interaction_score(
        self,
        score_grid: dict[tuple[Any, Any], float],
        spec_a: ParamSpec,
        spec_b: ParamSpec,
        probes_a: list[Any],
        probes_b: list[Any],
    ) -> float:
        """
        Compute a scalar interaction score for this pair.

        Method: additive-model deviation. For each cell (a, b), predict the
        score under an additive assumption:

            predicted(a, b) = score(base_a, base_b)
                              + [score(a, base_b) - score(base_a, base_b)]
                              + [score(base_a, b) - score(base_a, base_b)]

        The deviation is |observed - predicted|. Interaction score is the
        maximum deviation across the grid — a monotone proxy for "how much
        does the real landscape depart from additivity".
        """
        base_a = spec_a.default
        base_b = spec_b.default
        base_key = (base_a, base_b)
        if base_key not in score_grid:
            return 0.0
        base = score_grid[base_key]

        max_dev = 0.0
        for a in probes_a:
            for b in probes_b:
                if (a, b) not in score_grid:
                    continue
                if (a, base_b) not in score_grid or (base_a, b) not in score_grid:
                    continue
                delta_a = score_grid[(a, base_b)] - base
                delta_b = score_grid[(base_a, b)] - base
                predicted = base + delta_a + delta_b
                observed = score_grid[(a, b)]
                dev = abs(observed - predicted)
                if dev > max_dev:
                    max_dev = dev
        return max_dev

    def _find_promising(
        self,
        trials: list[TrialResult],
        score_grid: dict[tuple[Any, Any], float],
        spec_a: ParamSpec,
        spec_b: ParamSpec,
        probes_a: list[Any],
        probes_b: list[Any],
    ) -> list[TrialResult]:
        """
        A cell is "promising" when its observed score is significantly better
        than BOTH corresponding univariate predictions.

        Formally: for cell (a, b), the univariate best for param A (holding B
        at base) is `min(score(va, base_b) for va in probes_a)`. Similarly for B.
        The additive-model expectation of the best off-diagonal is the minimum
        of those two values (roughly — we bound by each axis's best probe).
        A cell is promising iff its score is at least `eps_min` lower than
        BOTH of those axis best scores, AND the cell is off-diagonal
        (i.e. neither value is at the baseline default).
        """
        base_a = spec_a.default
        base_b = spec_b.default

        col_axis_best = min(
            (score_grid.get((va, base_b), float("inf")) for va in probes_a),
            default=float("inf"),
        )
        row_axis_best = min(
            (score_grid.get((base_a, vb), float("inf")) for vb in probes_b),
            default=float("inf"),
        )
        univariate_bound = min(col_axis_best, row_axis_best)

        promising: list[TrialResult] = []
        for t in trials:
            va = t.config.get(spec_a.name)
            vb = t.config.get(spec_b.name)
            # Skip cells on the cross (where one value equals its base)
            if va == base_a or vb == base_b:
                continue
            # Require strict improvement over the additive bound by at least eps
            if t.score < univariate_bound - self.eps_min:
                promising.append(t)
        return promising
