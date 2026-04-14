"""
Regression tests for BenchmarkTrialExecutor cache-key isolation.

Motivation: the old `_config_key` was ``str(hash(tuple(sorted(config.items()))))``.
That expression has two severe defects:

1. **Process-dependent**: Python's built-in ``hash`` is randomized via
   PYTHONHASHSEED for string inputs, so the same config produced different
   keys across restarts — preventing any durable cross-run comparison.

2. **Context-blind**: the key was derived only from the config dict, without
   model_id / dataset_id / language. Prior-study trials that were measured on
   a 15-minute Stage-1 dataset could be ``warm_load``-ed into a Stage-2
   executor pointing at the SAME config but a 60-minute Stage-2 dataset.
   The first Stage-2 ``evaluate()`` would then return the cached Stage-1
   score as a "hit", silently leaking 15-minute measurement noise into a
   60-minute optimization decision.

The fix threads (model_id, dataset_id, lang) into a deterministic BLAKE2b
digest and teaches ``warm_load`` to refuse source contexts that do not match.
These tests lock both behaviors in.
"""

from __future__ import annotations

from types import SimpleNamespace

from asrbench.engine.search.benchmark_executor import BenchmarkTrialExecutor
from asrbench.engine.search.trial import (
    TrialResult,
    canonical_config_repr,
)


def _make_executor(model_id: str, dataset_id: str, lang: str) -> BenchmarkTrialExecutor:
    """
    Build a BenchmarkTrialExecutor with only the fields the cache-key logic
    touches. We bypass __init__ via __new__ because the full constructor would
    require a live DuckDB connection and a loaded backend.
    """
    ex = BenchmarkTrialExecutor.__new__(BenchmarkTrialExecutor)
    ex.model_id = model_id
    ex.dataset = SimpleNamespace(dataset_id=dataset_id)  # type: ignore[assignment]
    ex.lang = lang
    ex._cache = {}
    ex._runs_used = 0
    return ex


class TestConfigKeyDeterminism:
    def test_same_context_same_key(self) -> None:
        ex1 = _make_executor("m1", "d1", "en")
        ex2 = _make_executor("m1", "d1", "en")
        k1 = ex1._config_key({"beam_size": 5, "temperature": 0.0})
        k2 = ex2._config_key({"beam_size": 5, "temperature": 0.0})
        assert k1 == k2

    def test_order_independence(self) -> None:
        ex = _make_executor("m1", "d1", "en")
        k1 = ex._config_key({"a": 1, "b": 2, "c": 3})
        k2 = ex._config_key({"c": 3, "a": 1, "b": 2})
        assert k1 == k2

    def test_blake2b_128_width(self) -> None:
        ex = _make_executor("m1", "d1", "en")
        k = ex._config_key({"beam": 5})
        # BLAKE2b with digest_size=16 → 32-char hex string
        assert len(k) == 32
        int(k, 16)  # must be valid hex

    def test_process_independent(self) -> None:
        """
        The old implementation used builtin ``hash`` which is randomized per
        process for strings. Our replacement must be derived purely from the
        input bytes, so two separately built payloads with the same content
        give the same digest without any session state.
        """
        import hashlib

        ex = _make_executor("m1", "d1", "en")
        config = {"beam_size": 5, "temperature": 0.0}

        k = ex._config_key(config)
        expected_payload = f"m1|d1|en||{canonical_config_repr(config)}"
        expected = hashlib.blake2b(expected_payload.encode(), digest_size=16).hexdigest()
        assert k == expected


class TestConfigKeyContextIsolation:
    def test_different_dataset_different_key(self) -> None:
        """A Stage-1 (short dataset) key must not collide with a Stage-2 key."""
        ex_s1 = _make_executor("m1", "dataset-900s", "tr")
        ex_s2 = _make_executor("m1", "dataset-3600s", "tr")
        cfg = {"beam_size": 5, "temperature": 0.0}
        assert ex_s1._config_key(cfg) != ex_s2._config_key(cfg)

    def test_different_model_different_key(self) -> None:
        ex_a = _make_executor("model-a", "d1", "en")
        ex_b = _make_executor("model-b", "d1", "en")
        cfg = {"beam_size": 5}
        assert ex_a._config_key(cfg) != ex_b._config_key(cfg)

    def test_different_lang_different_key(self) -> None:
        ex_en = _make_executor("m1", "d1", "en")
        ex_tr = _make_executor("m1", "d1", "tr")
        cfg = {"beam_size": 5}
        assert ex_en._config_key(cfg) != ex_tr._config_key(cfg)

    def test_value_type_distinct(self) -> None:
        """
        ``repr`` round-trips int/float/bool distinctly so 0 / 0.0 / False
        don't alias into the same cache slot — the backend may treat them
        as semantically different.
        """
        ex = _make_executor("m1", "d1", "en")
        k_int = ex._config_key({"x": 0})
        k_float = ex._config_key({"x": 0.0})
        k_bool = ex._config_key({"x": False})
        assert len({k_int, k_float, k_bool}) == 3


class TestWarmLoadContextGuard:
    def _make_prior_trial(self, score: float = 0.10) -> TrialResult:
        return TrialResult.from_db_row(
            config={"beam_size": 5, "temperature": 0.0},
            score=score,
            score_ci=(score - 0.005, score + 0.005),
        )

    def test_matching_context_loads(self) -> None:
        ex = _make_executor("m1", "d1", "en")
        loaded = ex.warm_load(
            [self._make_prior_trial(0.10)],
            source_model_id="m1",
            source_dataset_id="d1",
            source_lang="en",
        )
        assert loaded == 1
        assert len(ex._cache) == 1  # type: ignore[attr-defined]

    def test_mismatched_dataset_refused(self) -> None:
        """Stage-1 trial on a 15-min dataset must NOT leak into Stage-2."""
        ex = _make_executor("m1", "dataset-3600s", "en")
        loaded = ex.warm_load(
            [self._make_prior_trial(0.10)],
            source_model_id="m1",
            source_dataset_id="dataset-900s",
            source_lang="en",
        )
        assert loaded == 0
        assert ex._cache == {}  # type: ignore[attr-defined]

    def test_mismatched_model_refused(self) -> None:
        ex = _make_executor("large-v3-turbo", "d1", "en")
        loaded = ex.warm_load(
            [self._make_prior_trial(0.10)],
            source_model_id="large-v3",
            source_dataset_id="d1",
            source_lang="en",
        )
        assert loaded == 0

    def test_mismatched_lang_refused(self) -> None:
        ex = _make_executor("m1", "d1", "tr")
        loaded = ex.warm_load(
            [self._make_prior_trial(0.10)],
            source_model_id="m1",
            source_dataset_id="d1",
            source_lang="en",
        )
        assert loaded == 0

    def test_legacy_call_without_context_still_works(self) -> None:
        """
        Older tests and the Synthetic executor path call ``warm_load`` without
        any source_* kwargs. In that backward-compatible mode we do NOT guard
        — otherwise every existing test would break. The guard only activates
        when at least one source_* identifier is supplied.
        """
        ex = _make_executor("m1", "d1", "en")
        loaded = ex.warm_load([self._make_prior_trial(0.10)])
        assert loaded == 1

    def test_refused_load_does_not_block_later_evaluate(self) -> None:
        """
        After a refused warm_load, the cache is empty so ``evaluate`` with
        the same config does NOT hit any stale entry. (We don't invoke the
        full evaluate path — just verify the cache state directly.)
        """
        ex = _make_executor("m1", "d1", "en")
        ex.warm_load(
            [self._make_prior_trial(0.10)],
            source_model_id="m1",
            source_dataset_id="different",
            source_lang="en",
        )
        fresh_key = ex._config_key({"beam_size": 5, "temperature": 0.0})
        assert fresh_key not in ex._cache  # type: ignore[attr-defined]


class TestTrialResultConfigKey:
    def test_deterministic_and_order_independent(self) -> None:
        t = TrialResult.from_db_row(config={"b": 2, "a": 1}, score=0.1, score_ci=(0.09, 0.11))
        # Recompute key from a config with the same content in a different order
        key1 = t.config_key()
        t2 = TrialResult.from_db_row(config={"a": 1, "b": 2}, score=0.1, score_ci=(0.09, 0.11))
        assert key1 == t2.config_key()
        assert len(key1) == 32

    def test_canonical_repr_helper_is_stable(self) -> None:
        assert canonical_config_repr({"a": 1, "b": 2}) == canonical_config_repr({"b": 2, "a": 1})
        # Distinct value types stay distinct
        assert canonical_config_repr({"x": 0}) != canonical_config_repr({"x": 0.0})
        assert canonical_config_repr({"x": 0}) != canonical_config_repr({"x": False})


class TestLegacySyntheticExecutorBackCompat:
    """
    The Synthetic executor used in the test suite never carried (model, dataset,
    lang) context — its cache key is still purely config-based, just with the
    hashing function swapped from builtin hash() to BLAKE2b. Make sure the
    swap didn't break equality semantics the existing tests depend on.
    """

    def test_synthetic_cache_hit_on_equal_config(self) -> None:
        from asrbench.engine.search.objective import SingleMetricObjective
        from asrbench.engine.search.trial import SyntheticTrialExecutor

        def landscape(_cfg: object) -> dict:
            return {
                "wer": 0.1,
                "cer": 0.05,
                "rtfx_mean": 20.0,
                "vram_peak_mb": 4000.0,
                "wer_ci_lower": 0.09,
                "wer_ci_upper": 0.11,
            }

        ex = SyntheticTrialExecutor(
            metric_fn=landscape, objective=SingleMetricObjective(metric="wer")
        )
        r1 = ex.evaluate({"a": 1, "b": 2})
        r2 = ex.evaluate({"b": 2, "a": 1})
        # Cache hit: same score, runs_used only incremented once
        assert r1.score == r2.score
        assert ex.runs_used == 1
