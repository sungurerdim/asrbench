"""AutoTune evaluation script for WER engine latency."""

from __future__ import annotations

import random
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# 1. Generate a deterministic 100-segment corpus with realistic WER variance
# ---------------------------------------------------------------------------
random.seed(42)

_VOCAB = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "a", "an", "is", "was", "were", "been", "have", "has", "hello", "world", "good", "morning", "afternoon", "evening", "today", "yesterday", "tomorrow", "please", "thank", "you", "welcome", "sorry", "excuse", "me", "can", "could", "would", "should", "running", "walking", "talking", "eating", "sleeping", "reading", "writing", "working", "beautiful", "wonderful", "terrible", "amazing", "incredible", "fantastic", "great", "computer", "science", "engineering", "mathematics", "physics", "chemistry", "biology", "university", "student", "teacher", "professor", "doctor", "engineer", "scientist", "government", "country", "president", "minister", "parliament", "congress", "senate", "economy", "market", "business", "company", "industry", "technology", "innovation"]


def _make_pair(n_words: int, error_rate: float) -> tuple[str, str]:
    ref_words = [random.choice(_VOCAB) for _ in range(n_words)]
    hyp_words = list(ref_words)
    n_errors = max(0, int(len(hyp_words) * error_rate))
    indices = random.sample(range(len(hyp_words)), min(n_errors, len(hyp_words)))
    for i in indices:
        action = random.random()
        if action < 0.5:
            hyp_words[i] = random.choice(_VOCAB)  # substitution
        elif action < 0.75:
            hyp_words[i] = ""  # deletion
        else:
            hyp_words.insert(i, random.choice(_VOCAB))  # insertion
    return " ".join(ref_words), " ".join(w for w in hyp_words if w)


REFS: list[str] = []
HYPS: list[str] = []
for _ in range(100):
    n_words = random.randint(5, 25)
    err = random.uniform(0.0, 0.4)
    r, h = _make_pair(n_words, err)
    REFS.append(r)
    HYPS.append(h)


# ---------------------------------------------------------------------------
# 2. Measure latency: call compute() N times, report median
# ---------------------------------------------------------------------------
def measure_latency() -> float:
    from asrbench.engine.wer import WEREngine

    engine = WEREngine()

    # Warm up (first call may trigger lazy imports / JIT)
    engine.compute(REFS[:2], HYPS[:2], lang="en")

    n_runs = 5
    times: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        engine.compute(REFS, HYPS, lang="en")
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times.sort()
    return times[n_runs // 2]  # median


# ---------------------------------------------------------------------------
# 3. Run existing tests to ensure correctness preserved
# ---------------------------------------------------------------------------
def measure_pass_rate() -> float:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_wer_engine.py", "-q", "--tb=no"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Parse "27 passed" from output
    for line in result.stdout.splitlines():
        if "passed" in line:
            import re

            m = re.search(r"(\d+) passed", line)
            if m:
                passed = int(m.group(1))
                # Check for failures
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
