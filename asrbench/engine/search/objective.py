"""
Objective function abstraction for the IAMS optimizer.

The optimizer is algorithmically a minimizer: every search layer calls
`objective.score(metrics)` and tries to lower it. Other directions (maximize
RTFx, for example) are mapped to minimization by sign inversion internally.

Two shipped implementations:

- SingleMetricObjective: target one measurement (wer | cer | rtfx | vram),
  with explicit direction (minimize | maximize). WER defaults to minimize,
  RTFx defaults to maximize, VRAM to minimize.

- WeightedObjective: scalar combination of multiple metrics with user-defined
  weights. A negative weight flips the sense of that metric (so RTFx with
  weight=-0.1 is "a little better when larger"). Metrics are normalized
  against a reference scale so weights stay intuitive.

CI propagation:
- Both objectives must also provide `score_ci(metrics)` which returns
  (lower, upper) — the score's bootstrap confidence interval derived from
  the underlying per-metric CIs. The significance gate in Layer 1 consumes
  this to decide whether two trials truly differ.

Required metric keys (optional ones tolerated):
    wer, cer, mer, wil, rtfx_mean, vram_peak_mb
    wer_ci_lower, wer_ci_upper  (only WER has CI today — future work: add
                                 per-metric CI for rtfx/vram)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

# Directions recognized throughout the optimizer.
Direction = Literal["minimize", "maximize"]

# Metrics the objective layer knows how to score. Adding a new one requires
# updating both single-metric extraction and CI propagation.
SUPPORTED_METRICS = frozenset({"wer", "cer", "mer", "wil", "rtfx", "vram"})

# Canonical per-metric default direction — used when user says --objective wer
# without specifying --maximize / --minimize.
_DEFAULT_DIRECTION: dict[str, Direction] = {
    "wer": "minimize",
    "cer": "minimize",
    "mer": "minimize",
    "wil": "minimize",
    "rtfx": "maximize",
    "vram": "minimize",
}

# Mapping from objective metric name to the actual key in the BenchmarkEngine
# aggregate dict. Isolates the objective layer from schema drift elsewhere.
_METRIC_KEY: dict[str, str] = {
    "wer": "wer",
    "cer": "cer",
    "mer": "mer",
    "wil": "wil",
    "rtfx": "rtfx_mean",
    "vram": "vram_peak_mb",
}


def _extract(metrics: Mapping[str, float | None], name: str) -> float:
    """
    Pull a metric value from the dict, raising a precise error if missing/None.

    Search layers should never see a None score — the underlying trial
    executor is responsible for guaranteeing that every metric named in
    the objective has a real value. If not, we fail loudly here rather
    than let NaN propagate into search decisions.
    """
    if name not in _METRIC_KEY:
        raise ValueError(
            f"Unknown objective metric '{name}'. Supported: {sorted(SUPPORTED_METRICS)}"
        )
    key = _METRIC_KEY[name]
    if key not in metrics:
        raise KeyError(
            f"Metric '{key}' not present in trial metrics "
            f"(required by objective '{name}'). Available: {sorted(metrics.keys())}"
        )
    value = metrics[key]
    if value is None:
        raise ValueError(
            f"Metric '{key}' is None in trial metrics — cannot score this trial. "
            "The trial executor must guarantee real values for the objective metric."
        )
    return float(value)


class Objective(ABC):
    """
    Abstract objective: minimization semantics, with optional CI propagation.

    Search layers only depend on this interface. They never inspect raw metrics.
    """

    @abstractmethod
    def score(self, metrics: Mapping[str, float | None]) -> float:
        """Return the scalar score for these metrics. Lower is always better."""

    @abstractmethod
    def score_ci(self, metrics: Mapping[str, float | None]) -> tuple[float, float]:
        """
        Return (lower, upper) — the 95% confidence interval for score().

        Must satisfy `lower <= score(metrics) <= upper`. When the underlying
        metric has no CI available, return a degenerate (score, score) tuple
        so the significance gate degrades gracefully to the epsilon check.
        """

    @abstractmethod
    def describe(self) -> str:
        """Human-readable one-line description for logs and study.json output."""


@dataclass(frozen=True)
class SingleMetricObjective(Objective):
    """
    Optimize one metric in one direction.

    Examples:
        SingleMetricObjective(metric="wer")                  # minimize WER
        SingleMetricObjective(metric="rtfx")                 # maximize RTFx
        SingleMetricObjective(metric="wer", direction="maximize")  # unusual but allowed

    Maximization is implemented by returning `-value` as the score, so the
    search layers still see a minimization problem.
    """

    metric: str
    direction: Direction | None = None  # None → use _DEFAULT_DIRECTION[metric]

    def __post_init__(self) -> None:
        if self.metric not in SUPPORTED_METRICS:
            raise ValueError(
                f"SingleMetricObjective: unknown metric '{self.metric}'. "
                f"Supported: {sorted(SUPPORTED_METRICS)}"
            )
        # Normalize direction to a concrete value (frozen dataclass workaround).
        object.__setattr__(
            self,
            "direction",
            self.direction or _DEFAULT_DIRECTION[self.metric],
        )
        if self.direction not in ("minimize", "maximize"):
            raise ValueError(
                f"SingleMetricObjective: direction must be 'minimize' or 'maximize', "
                f"got {self.direction!r}"
            )

    def _sign(self) -> float:
        return 1.0 if self.direction == "minimize" else -1.0

    def score(self, metrics: Mapping[str, float | None]) -> float:
        return self._sign() * _extract(metrics, self.metric)

    def score_ci(self, metrics: Mapping[str, float | None]) -> tuple[float, float]:
        # WER is the only metric with per-trial CI so far (bootstrap CI from B2).
        # For other metrics we return a degenerate interval centered on the score,
        # which makes the significance gate fall back to the epsilon check alone.
        if self.metric == "wer":
            lo = metrics.get("wer_ci_lower")
            hi = metrics.get("wer_ci_upper")
            if lo is not None and hi is not None:
                sign = self._sign()
                # Maximization flips the interval: [-hi, -lo]
                if sign < 0:
                    return (-float(hi), -float(lo))
                return (float(lo), float(hi))
        s = self.score(metrics)
        return (s, s)

    def describe(self) -> str:
        return f"{self.direction} {self.metric}"


@dataclass(frozen=True)
class WeightedObjective(Objective):
    """
    Linear combination of multiple metrics.

    Score formula:
        f(x) = Σᵢ wᵢ · metricᵢ(x)

    Weight sign semantics:
        - positive weight: metric is minimized (contributes directly)
        - negative weight: metric is maximized (contributes as -metric)

    Example:
        WeightedObjective(weights={"wer": 1.0, "rtfx": -0.1, "vram": 0.001})
        → minimize (wer - 0.1*rtfx + 0.001*vram)

    Magnitudes matter: users should tune them to express "what matters how much".
    A common starting point for ASR is { wer: 1.0, rtfx: -0.05 } meaning "a 1%
    absolute WER drop is worth a 20x RTFx boost". Try to keep contributions
    roughly balanced — unbalanced weights turn a multi-objective into
    single-objective in practice.
    """

    weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError(
                "WeightedObjective: weights dict cannot be empty. "
                "Provide at least one metric with its weight."
            )
        for name, w in self.weights.items():
            if name not in SUPPORTED_METRICS:
                raise ValueError(
                    f"WeightedObjective: unknown metric '{name}'. "
                    f"Supported: {sorted(SUPPORTED_METRICS)}"
                )
            if not isinstance(w, (int, float)) or isinstance(w, bool):
                raise ValueError(
                    f"WeightedObjective: weight for '{name}' must be numeric, got {w!r}"
                )
            if w == 0:
                raise ValueError(
                    f"WeightedObjective: weight for '{name}' is 0, which makes "
                    "the metric irrelevant. Remove it from the weights dict instead."
                )

    def score(self, metrics: Mapping[str, float | None]) -> float:
        total = 0.0
        for name, w in self.weights.items():
            total += float(w) * _extract(metrics, name)
        return total

    def score_ci(self, metrics: Mapping[str, float | None]) -> tuple[float, float]:
        # Only WER has a real CI today. Propagate it through the linear combo:
        # the WER contribution's interval is [w_wer * wer_lo, w_wer * wer_hi]
        # (or reversed if w_wer < 0), and other terms contribute their point value.
        # Delta over the full score = |w_wer| * (wer_hi - wer_lo) / 2.
        base = self.score(metrics)
        if "wer" not in self.weights:
            return (base, base)
        lo = metrics.get("wer_ci_lower")
        hi = metrics.get("wer_ci_upper")
        if lo is None or hi is None:
            return (base, base)
        w_wer = float(self.weights["wer"])
        half_width = abs(w_wer) * (float(hi) - float(lo)) / 2.0
        return (base - half_width, base + half_width)

    def describe(self) -> str:
        parts = [f"{'-' if w < 0 else '+'}{abs(w):g}·{name}" for name, w in self.weights.items()]
        return "weighted(" + " ".join(parts).lstrip("+") + ")"
