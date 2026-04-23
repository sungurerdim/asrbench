# Contributing to ASRbench

## Setup

```bash
git clone https://github.com/sungurerdim/asrbench
cd asrbench
pip install -e ".[dev,faster-whisper,tr]"
```

Requires Python 3.11+.

## Code Style

- **Formatter/Linter:** ruff (configured in `pyproject.toml`, line-length 100)
- **Type checker:** mypy (strict mode off, missing imports ignored)
- **Imports:** sorted by ruff (isort rules), `from __future__ import annotations` in every file
- **Annotations:** use `X | None` not `Optional[X]`, import from `collections.abc` not `typing`

```bash
ruff check asrbench/ tests/
ruff check asrbench/ tests/ --fix   # auto-fix
mypy asrbench/
```

## Testing

```bash
# All unit tests
python -m pytest tests/unit/

# Specific module
python -m pytest tests/unit/test_wer_engine.py -v

# With output
python -m pytest tests/ -v --no-header
```

Tests use pytest with `asyncio_mode = "auto"`. Test files go in `tests/unit/` for unit tests and `tests/integration/` for integration tests.

### Writing Tests

- Use `SyntheticTrialExecutor` for IAMS search layer tests (deterministic, no real backend needed)
- Use `pytest.approx()` for floating-point comparisons
- Test boundary conditions: empty input, max size, concurrent access
- Every bug fix requires a regression test

## Project Structure

```
asrbench/
  api/            REST + WebSocket endpoints
  backends/       ASR backend adapters (plugin system)
  cli/            Typer CLI subcommands
  data/           Dataset download + audio cache
  engine/         Core computation (benchmark, WER, optimizer)
    search/       IAMS 7-layer parameter search
tests/
  unit/           Fast, isolated unit tests
  integration/    Tests requiring DB or backend
docs/             Planning and reference docs
scripts/          Developer utilities (not installed with the package)
  preflight_matrix.py    Sanity checks for matrix.json definitions
  benchmarks/   AutoTune evaluation harness (optimizer latency research)
    bench.sh             Harness entry point
    eval_optimizer.py    Metric extractor for IAMSOptimizer throughput
    eval_wer.py          Metric extractor for WEREngine latency
    .autotune.json       AutoTune configuration (read-only by convention)
    README.md            Experiment loop + rules
```

### `scripts/`

`scripts/` hosts developer-facing utilities that are **not part of the installed
package**. They are meant to be run from the repository root (`python scripts/...`
or `bash scripts/...`). Runtime outputs such as `run.log` and `results.tsv` are
gitignored; commit the source scripts, never the artifacts.

## Key Patterns

- **Backend plugins:** registered via `pyproject.toml` entry points (`asrbench.backends` group)
- **Database:** DuckDB (embedded, single file). Schema in `asrbench/db.py`. Use `conn.cursor()` for thread safety.
- **Async:** FastAPI endpoints are async; BenchmarkEngine.run() is an async coroutine. CLI is sync.
- **IAMS search layers:** each layer is a pure module in `engine/search/` with its own unit test. Layers communicate via `TrialResult` and `TrialExecutor` protocol.

## Pull Requests

1. Create a feature branch from `master`
2. Make your changes
3. Ensure `ruff check` and `pytest` pass
4. Keep commits atomic with descriptive messages
5. Open a PR with a summary of changes
