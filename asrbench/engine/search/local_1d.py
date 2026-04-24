"""
IAMS Layer 2 + Layer 6 — 1D coordinate search routines.

Given a sensitive parameter and a "current best" baseline, find the value
for that parameter that minimizes the objective, with all other parameters
held fixed. The returned TrialResult becomes the new best-so-far for that
coordinate — and in the IAMS hybrid-sequential flow, is folded back into
the baseline for the next parameter's search.

Three strategies are dispatched based on parameter type:

    GoldenSectionSearch     → continuous float (param.is_continuous())
    PatternSearch           → int with large range (|Ω| > 6)
    ExhaustiveSearch        → bool, enum, or small int (|Ω| ≤ 6)

Layer 6 (high-resolution refinement) reuses the same routines but with
tighter tolerances (configurable via the `tolerance` parameter on GSS and
the `max_iterations` on pattern search).

Every search:

    1. Accepts a starting point (usually the best probe from screening plus
       any seed trials the pairwise grid may have produced)
    2. Calls TrialExecutor.evaluate() for new candidates, recording each in
       the budget controller
    3. Returns the best TrialResult found under the (CI ∧ eps) significance
       gate
    4. Early-terminates when the budget is exhausted OR no significant
       improvement is observed across `patience` consecutive candidates

Seed trials:
    If the screening phase has already evaluated boundary points (min/max)
    and the baseline is the all-defaults trial, the search can skip those
    three points and start from the local neighborhood. Pass the precomputed
    trials in `seed_trials` — the routine will use them without re-running.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from asrbench.engine.search.budget import BudgetController
from asrbench.engine.search.significance import is_improvement
from asrbench.engine.search.space import ParamSpec
from asrbench.engine.search.trial import TrialExecutor, TrialResult

# The golden ratio complement, used by golden section search.
_PHI = (math.sqrt(5) - 1) / 2  # ≈ 0.618


class BudgetExhausted(Exception):
    """
    Raised by _evaluate_override when the budget hard cap would be exceeded
    by a fresh evaluation. Search routines catch this and return the best
    trial seen so far. Cache hits NEVER raise this exception — they bypass
    the cap because they cost nothing.
    """


@dataclass
class LocalSearchResult:
    """Outcome of a 1D search over one parameter."""

    param_name: str
    best: TrialResult
    trials: list[TrialResult] = field(default_factory=list)
    iterations: int = 0
    early_stopped: bool = False
    early_stop_reason: str = ""


def _evaluate_override(
    executor: TrialExecutor,
    baseline_config: dict[str, Any],
    param: ParamSpec,
    value: Any,
    phase: str,
    reasoning: str,
    budget: BudgetController,
) -> TrialResult:
    """
    Build a full config by taking the baseline and overriding one parameter,
    evaluate it, and record the score against the budget.

    The override value is clamped through the ParamSpec so search routines
    that propose values slightly outside the legal range (e.g. GSS after
    floating-point drift) don't crash.
    """
    override = dict(baseline_config)
    override[param.name] = param.clamp(value)
    trial = executor.evaluate(override, phase=phase, reasoning=reasoning)
    budget.record(trial.score)
    return trial


def _quadratic_refine(
    trials: list[TrialResult],
    baseline_config: dict[str, Any],
    param: ParamSpec,
    executor: TrialExecutor,
    budget: BudgetController,
    *,
    phase: str,
) -> TrialResult | None:
    """
    Fit a parabola to the three best points collected so far on this
    parameter and, if the fit is convex, probe its analytic minimum.

    Rationale:
        Golden section + pattern search both converge linearly — each
        iteration shrinks the bracket by a constant factor but never uses
        the landscape's curvature. When the objective is well-approximated
        by a parabola in the local neighborhood (which it usually is near
        a minimum), one extra probe at the parabola's vertex gets us
        substantially closer to the true optimum for the cost of a single
        trial.

    Eligibility gates:
        - Parameter must be continuous float or int (clampable to number)
        - At least 3 distinct probes with finite scores
        - ``np.polyfit(deg=2)`` returns a convex quadratic (a > 0)
        - The analytic vertex must not already be within 1% of an existing
          probe, so we don't waste a trial on a point we've all-but-measured
        - Budget must have at least one run left

    Returns:
        The new TrialResult if the refinement probe ran, else None.
    """
    if param.type not in ("float", "int"):
        return None
    if budget.should_stop():
        return None

    # Keep only probes that have a numeric value for this parameter.
    numeric: list[tuple[float, TrialResult]] = []
    for t in trials:
        v = t.config.get(param.name)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if math.isfinite(t.score):
                numeric.append((float(v), t))
    if len(numeric) < 3:
        return None

    # Pick the three lowest-scoring probes, preferring distinct x values.
    numeric.sort(key=lambda p: p[1].score)
    picked: list[tuple[float, TrialResult]] = []
    seen_x: set[float] = set()
    for x, t in numeric:
        if x in seen_x:
            continue
        picked.append((x, t))
        seen_x.add(x)
        if len(picked) == 3:
            break
    if len(picked) < 3:
        return None

    xs = np.array([p[0] for p in picked], dtype=float)
    ys = np.array([p[1].score for p in picked], dtype=float)
    try:
        a, b, _c = np.polyfit(xs, ys, 2)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not math.isfinite(float(a)) or a <= 0:
        return None  # concave or ill-conditioned

    x_star = -float(b) / (2.0 * float(a))
    x_star = param.clamp(x_star)

    # Too close to an existing probe? Skip — no new information to gain.
    lo = float(param.min) if param.min is not None else min(xs)
    hi = float(param.max) if param.max is not None else max(xs)
    span = max(1e-12, hi - lo)
    for probed_x, _ in picked:
        if abs(float(x_star) - float(probed_x)) < span * 0.01:
            return None

    return _evaluate_override(
        executor,
        baseline_config,
        param,
        x_star,
        phase + "-quad",
        f"quadratic-refine vertex={x_star}",
        budget,
    )


def golden_section_search(
    executor: TrialExecutor,
    baseline_config: dict[str, Any],
    param: ParamSpec,
    budget: BudgetController,
    *,
    eps_min: float = 0.005,
    max_iterations: int = 8,
    tolerance: float | None = None,
    seed_trials: list[TrialResult] | None = None,
    phase: str = "local_1d",
) -> LocalSearchResult:
    """
    Golden section search on a continuous float parameter.

    Invariant: at each iteration, we keep four points a < c < d < b where
    f(c) and f(d) are the two interior probes. The golden ratio guarantees
    that we only need ONE new evaluation per iteration to maintain the
    invariant as we shrink the bracket.

    Stopping conditions:
        1. max_iterations exhausted
        2. |b - a| < tolerance (default: 1% of the initial range)
        3. budget.should_stop() → True
        4. No significant improvement in the last 3 iterations

    Returns the lowest-scoring trial seen.
    """
    if param.type != "float":
        raise ValueError(f"golden_section_search requires a float parameter, got {param.type!r}")
    if param.min is None or param.max is None:
        raise ValueError(f"Parameter '{param.name}' has no min/max — cannot run golden section")

    a = float(param.min)
    b = float(param.max)
    if tolerance is None:
        tolerance = (b - a) * 0.01

    trials: list[TrialResult] = []
    seed_by_value: dict[float, TrialResult] = {}
    if seed_trials:
        for t in seed_trials:
            v = t.config.get(param.name)
            if isinstance(v, (int, float)):
                seed_by_value[float(v)] = t
                trials.append(t)

    def probe(x: float, reasoning: str) -> TrialResult:
        key = float(param.clamp(x))
        if key in seed_by_value:
            return seed_by_value[key]
        t = _evaluate_override(executor, baseline_config, param, key, phase, reasoning, budget)
        seed_by_value[key] = t
        trials.append(t)
        return t

    # Initialize the two interior probes
    c = b - _PHI * (b - a)
    d = a + _PHI * (b - a)
    t_c = probe(c, f"golden-section init c={c:.4f}")
    t_d = probe(d, f"golden-section init d={d:.4f}")

    best = min(trials, key=lambda t: t.score)
    no_improve_streak = 0
    iterations = 0
    early_stopped = False
    early_stop_reason = ""

    # Adaptive patience: when the budget is nearly used up, stop sooner.
    # Scale linearly between 1 and 3 based on remaining fraction.
    remaining_ratio = budget.remaining / budget.hard_cap if budget.hard_cap else 1.0
    adaptive_patience = max(1, round(3 * remaining_ratio))

    for i in range(max_iterations):
        iterations = i + 1
        if budget.should_stop():
            early_stopped = True
            early_stop_reason = "budget exhausted or converged"
            break
        if abs(b - a) < tolerance:
            early_stopped = True
            early_stop_reason = f"bracket width {abs(b - a):.5f} < tolerance {tolerance:.5f}"
            break

        if t_c.score < t_d.score:
            # Minimum is in [a, d] — shrink to the left
            b, d, t_d = d, c, t_c
            c = b - _PHI * (b - a)
            t_c = probe(c, f"golden-section iter {i + 1} c={c:.4f}")
            candidate = t_c
        else:
            # Minimum is in [c, b] — shrink to the right
            a, c, t_c = c, d, t_d
            d = a + _PHI * (b - a)
            t_d = probe(d, f"golden-section iter {i + 1} d={d:.4f}")
            candidate = t_d

        if is_improvement(candidate, best, eps_min=eps_min):
            best = candidate
            no_improve_streak = 0
        else:
            no_improve_streak += 1
            if no_improve_streak >= adaptive_patience:
                early_stopped = True
                early_stop_reason = f"no improvement in {adaptive_patience} consecutive iterations"
                break

    # Quadratic vertex refinement — cheap curvature-aware extra probe.
    refined = _quadratic_refine(trials, baseline_config, param, executor, budget, phase=phase)
    if refined is not None:
        trials.append(refined)
        if refined.score < best.score:
            best = refined

    return LocalSearchResult(
        param_name=param.name,
        best=best,
        trials=trials,
        iterations=iterations,
        early_stopped=early_stopped,
        early_stop_reason=early_stop_reason,
    )


def exhaustive_search(
    executor: TrialExecutor,
    baseline_config: dict[str, Any],
    param: ParamSpec,
    budget: BudgetController,
    *,
    eps_min: float = 0.005,
    seed_trials: list[TrialResult] | None = None,
    phase: str = "local_1d",
) -> LocalSearchResult:
    """
    Enumerate every value in the parameter's domain.

    Used for bool, enum, and small int / stepped-float parameters where the
    full domain fits in a handful of evaluations. Early termination on budget
    exhaustion, but no patience-based stopping — we always want the global
    minimum over a discrete finite set.
    """
    values = param.enumerate_values(max_points=32)
    seed_by_value: dict[Any, TrialResult] = {}
    trials: list[TrialResult] = []
    if seed_trials:
        for t in seed_trials:
            v = t.config.get(param.name)
            if v is not None:
                seed_by_value[v] = t
                trials.append(t)

    for v in values:
        if budget.should_stop():
            break
        if v in seed_by_value:
            continue
        t = _evaluate_override(
            executor,
            baseline_config,
            param,
            v,
            phase,
            f"exhaustive {param.name}={v!r}",
            budget,
        )
        seed_by_value[v] = t
        trials.append(t)

    if not trials:
        raise RuntimeError(
            f"exhaustive_search over '{param.name}' produced zero trials — "
            "budget was exhausted before any value could be probed"
        )

    # Pick the best trial under the significance gate: lowest score wins,
    # with eps_min used only as a tiebreaker via is_improvement() semantics.
    best = trials[0]
    for t in trials[1:]:
        if is_improvement(t, best, eps_min=eps_min):
            best = t
        elif t.score < best.score:
            # Not significant but still lower — keep the better point anyway
            # so Layer 2's sequential update is monotone.
            best = t

    return LocalSearchResult(
        param_name=param.name,
        best=best,
        trials=trials,
        iterations=len(trials),
        early_stopped=budget.should_stop(),
        early_stop_reason="budget exhausted" if budget.should_stop() else "",
    )


def pattern_search(
    executor: TrialExecutor,
    baseline_config: dict[str, Any],
    param: ParamSpec,
    budget: BudgetController,
    *,
    eps_min: float = 0.005,
    max_iterations: int = 12,
    seed_trials: list[TrialResult] | None = None,
    phase: str = "local_1d",
) -> LocalSearchResult:
    """
    Hooke-Jeeves-style pattern search for integer parameters with large range.

    Starts from the baseline value and probes ±step. If either side improves,
    accept it and repeat from the new center. If neither improves, halve the
    step and retry. Terminates when step == 1 and no direction improves, or
    when max_iterations / budget is reached.

    This gives a deterministic O(log(range)) search over integers without the
    floating-point arithmetic of golden section.
    """
    if param.type != "int":
        raise ValueError(f"pattern_search requires an int parameter, got {param.type!r}")
    if param.min is None or param.max is None:
        raise ValueError(f"Parameter '{param.name}' has no min/max — cannot pattern search")

    lo = int(param.min)
    hi = int(param.max)
    center = int(baseline_config.get(param.name, param.default))
    center = max(lo, min(hi, center))
    step = max(1, (hi - lo) // 4)

    trials: list[TrialResult] = []
    seed_by_value: dict[int, TrialResult] = {}
    if seed_trials:
        for t in seed_trials:
            v = t.config.get(param.name)
            if isinstance(v, int) and not isinstance(v, bool):
                seed_by_value[int(v)] = t
                trials.append(t)

    def probe(value: int, reasoning: str) -> TrialResult:
        value = max(lo, min(hi, value))
        if value in seed_by_value:
            return seed_by_value[value]
        t = _evaluate_override(executor, baseline_config, param, value, phase, reasoning, budget)
        seed_by_value[value] = t
        trials.append(t)
        return t

    t_center = probe(center, f"pattern-search center={center}")
    best = t_center
    iterations = 0
    early_stopped = False
    early_stop_reason = ""

    # Adaptive budget awareness: when the controller is nearly exhausted,
    # cap the outer loop more tightly so we don't burn the remainder on a
    # step-halving cascade that won't escape.
    remaining_ratio = budget.remaining / budget.hard_cap if budget.hard_cap else 1.0
    effective_max = max(1, round(max_iterations * max(remaining_ratio, 0.25)))

    for i in range(effective_max):
        iterations = i + 1
        if budget.should_stop():
            early_stopped = True
            early_stop_reason = "budget exhausted or converged"
            break

        left_val = center - step
        right_val = center + step
        candidates: list[tuple[int, TrialResult]] = []
        # Per-probe budget check: a single loop iteration can burn up to 2
        # trials, so we need to re-verify cap before each individual evaluate.
        if left_val >= lo and budget.can_run():
            t_left = probe(left_val, f"pattern-search iter {i + 1} left={left_val}")
            candidates.append((left_val, t_left))
        if right_val <= hi and budget.can_run():
            t_right = probe(right_val, f"pattern-search iter {i + 1} right={right_val}")
            candidates.append((right_val, t_right))

        if not candidates:
            early_stopped = True
            early_stop_reason = "center at both bounds; nothing to explore"
            break

        # Pick the best candidate, with significance gate tiebreaker
        candidates.sort(key=lambda cv: cv[1].score)
        best_val, best_cand = candidates[0]

        if is_improvement(best_cand, best, eps_min=eps_min):
            best = best_cand
            center = best_val  # Accept the move
        else:
            # No improvement at this step size — halve it
            if step == 1:
                early_stopped = True
                early_stop_reason = "step=1 and no improvement in either direction"
                break
            step = max(1, step // 2)

    refined = _quadratic_refine(trials, baseline_config, param, executor, budget, phase=phase)
    if refined is not None:
        trials.append(refined)
        if refined.score < best.score:
            best = refined

    return LocalSearchResult(
        param_name=param.name,
        best=best,
        trials=trials,
        iterations=iterations,
        early_stopped=early_stopped,
        early_stop_reason=early_stop_reason,
    )


def search_1d(
    executor: TrialExecutor,
    baseline_config: dict[str, Any],
    param: ParamSpec,
    budget: BudgetController,
    *,
    eps_min: float = 0.005,
    seed_trials: list[TrialResult] | None = None,
    phase: str = "local_1d",
) -> LocalSearchResult:
    """
    Dispatch to the right 1D search strategy for this parameter.

    - float (continuous)    → golden section
    - int with >6 distinct values → pattern search
    - everything else       → exhaustive
    """
    if param.type == "float" and param.step is None:
        return golden_section_search(
            executor,
            baseline_config,
            param,
            budget,
            eps_min=eps_min,
            seed_trials=seed_trials,
            phase=phase,
        )
    if param.type == "int":
        # If the user supplied a step, the parameter is effectively discrete
        # over a sparse grid — enumerate it exhaustively so pattern search
        # doesn't probe illegal in-between values. Also routes small domains
        # (≤6 points) to exhaustive unconditionally.
        step = int(param.step) if param.step is not None else 1
        grid_size = (int(param.max) - int(param.min)) // step + 1
        if param.step is None and grid_size > 6:
            return pattern_search(
                executor,
                baseline_config,
                param,
                budget,
                eps_min=eps_min,
                seed_trials=seed_trials,
                phase=phase,
            )
    # enum, bool, small int, stepped float → exhaustive
    return exhaustive_search(
        executor,
        baseline_config,
        param,
        budget,
        eps_min=eps_min,
        seed_trials=seed_trials,
        phase=phase,
    )
