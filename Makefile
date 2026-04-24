# Developer shortcuts — see CONTRIBUTING.md for the long form.
#
# Every target delegates to `uv` when present so local dev matches CI
# exactly. If uv is missing we fall back to plain python / pip / npm so
# the Makefile still works on a fresh clone.

.PHONY: help install install-pip lint format typecheck test test-cov \
        ui-build ui-dev coverage-html pre-commit pre-commit-install \
        docker-build clean all

UV := $(shell command -v uv 2>/dev/null)

help:
	@echo "ASRbench developer targets:"
	@echo "  install              — uv sync with dev + common runtime extras"
	@echo "  install-pip          — same, but via pip (for environments without uv)"
	@echo "  lint                 — ruff check + ruff format --check"
	@echo "  format               — ruff check --fix + ruff format"
	@echo "  typecheck            — mypy asrbench"
	@echo "  test                 — pytest"
	@echo "  test-cov             — pytest with branch coverage and the 70% gate"
	@echo "  coverage-html        — write htmlcov/index.html for a richer view"
	@echo "  ui-build             — npm ci && npm run build (writes asrbench/static)"
	@echo "  ui-dev               — npm run dev (Vite dev server at :5173)"
	@echo "  pre-commit-install   — pre-commit install (requires dev extras)"
	@echo "  pre-commit           — run every hook against the full tree"
	@echo "  docker-build         — docker build . -t asrbench:dev"
	@echo "  clean                — wipe caches and wheel build output"

install:
ifeq ($(UV),)
	@echo "uv not found — falling back to pip. Install uv for faster, reproducible syncs."
	python -m pip install -e ".[dev,faster-whisper,tr]"
else
	$(UV) sync --frozen --extra dev --extra faster-whisper --extra tr
endif

install-pip:
	python -m pip install -e ".[dev,faster-whisper,tr]"

lint:
ifeq ($(UV),)
	python -m ruff check asrbench tests
	python -m ruff format --check asrbench tests
else
	$(UV) run ruff check asrbench tests
	$(UV) run ruff format --check asrbench tests
endif

format:
ifeq ($(UV),)
	python -m ruff check --fix asrbench tests
	python -m ruff format asrbench tests
else
	$(UV) run ruff check --fix asrbench tests
	$(UV) run ruff format asrbench tests
endif

typecheck:
ifeq ($(UV),)
	python -m mypy asrbench
else
	$(UV) run mypy asrbench
endif

test:
ifeq ($(UV),)
	python -m pytest
else
	$(UV) run pytest
endif

test-cov:
ifeq ($(UV),)
	python -m pytest --cov=asrbench --cov-report=term-missing --cov-fail-under=70
else
	$(UV) run pytest --cov=asrbench --cov-report=term-missing --cov-fail-under=70
endif

coverage-html:
ifeq ($(UV),)
	python -m pytest --cov=asrbench --cov-report=html
else
	$(UV) run pytest --cov=asrbench --cov-report=html
endif
	@echo "Open htmlcov/index.html in your browser."

ui-build:
	cd ui && npm ci && npm run build

ui-dev:
	@echo "Starting Vite dev server at http://localhost:5173/"
	@echo "Pair with: asrbench serve --dev"
	cd ui && npm run dev

pre-commit-install:
ifeq ($(UV),)
	python -m pre_commit install
else
	$(UV) run pre-commit install
endif

pre-commit:
ifeq ($(UV),)
	python -m pre_commit run --all-files
else
	$(UV) run pre-commit run --all-files
endif

docker-build:
	docker build -t asrbench:dev .

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov
	rm -rf build dist *.egg-info
	find asrbench tests -type d -name __pycache__ -exec rm -rf {} +
	find asrbench tests -type f -name "*.pyc" -delete

all: lint typecheck test
