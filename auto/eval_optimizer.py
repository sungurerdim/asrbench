"""AutoTune evaluation script for IAMSOptimizer throughput."""

from __future__ import annotations

import re
import subprocess
import sys
import time
from collections.abc import Mapping
from typing import Any


def _metrics(wer: float) -> dict[str, float]:
    return {
        "wer": wer,
        "cer": wer * 0.5,
        "mer": wer,
        "wil": wer,
        "rtfx_mean": 20.0,
        "vram_peak_mb": 4000.0,
        "wer_ci_lower": max(0.0, wer - 0.0005),
        "wer_ci_upper": wer + 0.0005,
    }


def convex_5param(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    beam = int(cfg.get("beam_size", 5))
    temp = float(cfg.get("temperature", 0.0))
    patience = float(cfg.get("patience", 0.5))
    vad = bool(cfg.get("vad", False))
    best_of = int(cfg.get("best_of", 1))
    wer = (
        0.05
        + 0.003 * (beam - 7) ** 2
        + 0.02 * (temp - 0.3) ** 2
        + 0.01 * (patience - 1.0) ** 2
        + (0.0 if vad else 0.04)
        + 0.002 * (best_of - 5) ** 2
    )
    return _metrics(wer)


def measure_latency() -> float:
    from asrbench.engine.optimizer import IAMSOptimizer
    from asrbench.engine.search.budget import BudgetController
    from asrbench.engine.search.objective import SingleMetricObjective
    from asrbench.engine.search.space import ParameterSpace, ParamSpec
    from asrbench.engine.search.trial import SyntheticTrialExecutor

    space = ParameterSpace(
        parameters=(
            ParamSpec(name="beam_size", type="int", min=1, default=3, max=15),
            ParamSpec(name="temperature", type="float", min=0.0, default=0.1, max=1.0),
            ParamSpec(name="patience", type="float", min=0.0, default=0.5, max=2.0),
            ParamSpec(name="vad", type="bool", default=False),
            ParamSpec(name="best_of", type="int", min=1, default=3, max=10),
        )
    )
    obj = SingleMetricObjective(metric="wer")

    # Warmup
    ex = SyntheticTrialExecutor(metric_fn=convex_5param, objective=obj)
    budget = BudgetController(hard_cap=200, convergence_window=0)
    IAMSOptimizer(
        executor=ex, space=space, objective=obj, budget=budget, eps_min=0.002, mode="maximum"
    ).run()

    # Measure 5 runs, take median
    times: list[float] = []
    for _ in range(5):
        ex = SyntheticTrialExecutor(metric_fn=convex_5param, objective=obj)
        budget = BudgetController(hard_cap=200, convergence_window=0)
        opt = IAMSOptimizer(
            executor=ex, space=space, objective=obj, budget=budget, eps_min=0.002, mode="maximum"
        )
        t0 = time.perf_counter()
        opt.run()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times.sort()
    return times[len(times) // 2]


def measure_pass_rate() -> float:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/test_iams_optimizer.py",
            "tests/unit/test_search_trial.py",
            "-q",
            "--tb=no",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    for line in result.stdout.splitlines():
        if "passed" in line:
            m = re.search(r"(\d+) passed", line)
            if m:
                passed = int(m.group(1))
                f = re.search(r"(\d+) failed", line)
                failed = int(f.group(1)) if f else 0
                total = passed + failed
                return passed / total if total > 0 else 0.0
    return 0.0


if __name__ == "__main__":
    latency = measure_latency()
    pass_rate = measure_pass_rate()
    print(f"latency_ms:\t{latency:.3f}")
    print(f"pass_rate:\t{pass_rate:.4f}")
