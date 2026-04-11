"""
Statistical significance gate for the IAMS optimizer.

Two trials (or two sets of trials) are judged "significantly different" only
when BOTH of the following hold:

    1. Statistical test: their 95% confidence intervals do not overlap.
    2. Practical test: the absolute score difference is at least `eps_min`.

This AND-semantics is the strictest of the reasonable choices and was picked
deliberately in the design phase: we would rather declare two configs "equivalent"
and skip further refinement than chase phantom improvements that are within
measurement noise.

How it is used:

- Layer 1 (screening): `is_sensitive(baseline, min, max, ...)` decides whether
  a parameter is worth optimizing at all. An insensitive parameter is fixed
  at its default for the rest of the run.

- Layer 2 (local 1D search): every candidate step must pass `is_improvement`
  vs the current best, otherwise the search terminates for that parameter.

- Layer 4 (multi-start): candidates from different starting points are compared
  pairwise; ties (non-significant diffs) are broken by total run count.

- Layer 5 (ablation): a parameter is declared "toxic" only when removing it
  produces a significant improvement.

No search layer compares raw metrics directly. They all route through this
module so the epsilon and CI policy is defined exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass

from asrbench.engine.search.objective import Objective
from asrbench.engine.search.trial import TrialResult


@dataclass(frozen=True)
class SignificanceVerdict:
    """
    Explain-yourself output of a significance comparison.

    Used by the IAMS orchestrator when writing `reasoning` into the trial log,
    so the user can audit every decision later.
    """

    a_better: bool  # True if trial A has strictly better (lower) score
    b_better: bool  # True if trial B has strictly better (lower) score
    significant: bool  # Both gates passed (CI non-overlap AND |Δ| ≥ eps)
    delta: float  # score_a - score_b
    ci_overlap: bool  # True if the two CIs overlap
    eps_min: float  # epsilon threshold used for this comparison
    reason: str  # human-readable summary

    @property
    def equivalent(self) -> bool:
        return not self.significant


def ci_overlap(ci_a: tuple[float, float], ci_b: tuple[float, float]) -> bool:
    """
    Return True iff the closed intervals [ci_a.lo, ci_a.hi] and [ci_b.lo, ci_b.hi] intersect.

    Degenerate intervals (lo == hi, i.e. the executor has no CI for this metric)
    still work: a point overlaps another interval iff it lies inside.
    """
    a_lo, a_hi = ci_a
    b_lo, b_hi = ci_b
    if a_lo > a_hi:
        a_lo, a_hi = a_hi, a_lo  # be forgiving of callers that pass (hi, lo)
    if b_lo > b_hi:
        b_lo, b_hi = b_hi, b_lo
    return not (a_hi < b_lo or b_hi < a_lo)


def compare(
    a: TrialResult,
    b: TrialResult,
    eps_min: float = 0.005,
) -> SignificanceVerdict:
    """
    Compare two trial results under the (CI ∧ eps) significance gate.

    Returns a SignificanceVerdict. Convention:
        - a_better iff score_a < score_b AND gate passed
        - b_better iff score_b < score_a AND gate passed
        - otherwise both False and significant=False

    `a` and `b` must have scores produced by the SAME objective. The executor
    enforces this by routing every trial through the injected objective — but
    callers that assemble TrialResults manually must be careful.
    """
    delta = a.score - b.score
    overlap = ci_overlap(a.score_ci, b.score_ci)
    abs_delta = abs(delta)

    statistical = not overlap
    practical = abs_delta >= eps_min
    significant = statistical and practical

    a_better = significant and delta < 0
    b_better = significant and delta > 0

    if not significant:
        if not statistical and not practical:
            reason = f"equivalent (Δ={delta:.4f} < ε={eps_min:.4f} AND CIs overlap)"
        elif not statistical:
            reason = f"equivalent (CIs overlap; Δ={delta:.4f} would pass ε gate alone)"
        else:
            reason = f"equivalent (Δ={delta:.4f} < ε={eps_min:.4f}; CIs do not overlap)"
    else:
        direction = "a better" if a_better else "b better"
        reason = f"significant ({direction}: Δ={delta:+.4f}, ε={eps_min:.4f}, CI gap confirmed)"

    return SignificanceVerdict(
        a_better=a_better,
        b_better=b_better,
        significant=significant,
        delta=delta,
        ci_overlap=overlap,
        eps_min=eps_min,
        reason=reason,
    )


def is_improvement(
    candidate: TrialResult,
    incumbent: TrialResult,
    eps_min: float = 0.005,
) -> bool:
    """
    Shortcut used by local 1D search: is `candidate` significantly better than `incumbent`?

    Returns True iff the significance gate fires AND candidate has lower score.
    Fast path: skips SignificanceVerdict construction and f-string formatting.
    """
    delta = candidate.score - incumbent.score
    if delta >= 0:
        return False
    if abs(delta) < eps_min:
        return False
    return not ci_overlap(candidate.score_ci, incumbent.score_ci)


def is_sensitive(
    baseline: TrialResult,
    min_trial: TrialResult,
    max_trial: TrialResult,
    eps_min: float = 0.005,
) -> bool:
    """
    Layer 1 gate: is this parameter worth further search?

    A parameter is sensitive when AT LEAST ONE of the three pairs
    (baseline↔min, baseline↔max, min↔max) shows a significant difference.

    Fast path: inline significance check without constructing SignificanceVerdict.
    """
    pairs = (
        (baseline, min_trial),
        (baseline, max_trial),
        (min_trial, max_trial),
    )
    for a, b in pairs:
        if abs(a.score - b.score) >= eps_min and not ci_overlap(a.score_ci, b.score_ci):
            return True
    return False


def sensitivity_score(
    baseline: TrialResult,
    min_trial: TrialResult,
    max_trial: TrialResult,
    objective: Objective,  # noqa: ARG001  (kept for API symmetry — future use)
) -> float:
    """
    Scalar ranking used to sort sensitive parameters by priority.

    Defined as the spread across the 3-point screening probes:
        s = max(score_base, score_min, score_max) - min(...)

    Higher spread = more leverage for optimization. The IAMS Layer-2 orchestrator
    optimizes parameters in descending order of this score so that the earliest
    baseline updates correspond to the biggest WER movements.

    Note: this is a magnitude, not a significance test. Use `is_sensitive()` to
    decide whether to bother at all; use `sensitivity_score()` to decide order.
    """
    scores = [baseline.score, min_trial.score, max_trial.score]
    return max(scores) - min(scores)
