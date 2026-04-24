# syntax=docker/dockerfile:1.7
#
# ASRbench container image — multi-stage:
#   1. `ui-build`: Node 20, Vite build → asrbench/static/
#   2. `final`:   Python 3.11 slim, pip install the wheel + extras,
#                 run as a non-root user.
#
# Build:
#   docker build -t asrbench:dev .
# Run:
#   docker run --rm -p 8765:8765 -v $HOME/.asrbench:/home/asrbench/.asrbench \
#       -e HF_TOKEN="$HF_TOKEN" asrbench:dev

# ---------------------------------------------------------------------------
# Stage 1 — build the UI bundle from source
# ---------------------------------------------------------------------------
FROM node:20-slim AS ui-build
WORKDIR /src/ui
COPY ui/package*.json ./
RUN npm ci --no-audit --no-fund
COPY ui /src/ui
COPY asrbench/static /src/asrbench/static
RUN npm run build
# The Vite config writes to ../asrbench/static; mirror that into the
# next stage.
RUN ls -la /src/asrbench/static

# ---------------------------------------------------------------------------
# Stage 2 — build the wheel so the runtime stage installs from dist/
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS wheel-build
WORKDIR /src
RUN pip install --no-cache-dir hatchling==1.25.0 build==1.2.2
COPY . /src
# Inject the freshly-built UI bundle so the wheel carries it.
COPY --from=ui-build /src/asrbench/static /src/asrbench/static
RUN python -m build --wheel --outdir /dist

# ---------------------------------------------------------------------------
# Stage 3 — runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS final

# System deps: ffmpeg for codec sim + curl for the HEALTHCHECK probe.
# Keep the layer small — no build-essential once the wheel is prebuilt.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user matches the UID used in the wheel's entry point so
# ~/.asrbench mounts map to a writable directory.
RUN useradd --create-home --shell /bin/bash --uid 1000 asrbench
USER asrbench
WORKDIR /home/asrbench

COPY --from=wheel-build --chown=asrbench:asrbench /dist/*.whl /tmp/
RUN pip install --user --no-cache-dir \
        /tmp/asrbench-*.whl \
        "asrbench[faster-whisper]" \
        "asrbench[whisper-cpp]" \
        "asrbench[observability]" \
    && rm /tmp/asrbench-*.whl

ENV PATH="/home/asrbench/.local/bin:${PATH}"
ENV ASRBENCH_HOME="/home/asrbench/.asrbench"

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8765/system/health || exit 1

# Default: serve on 0.0.0.0 inside the container. The operator must
# still supply ASRBENCH_API_KEY for --allow-network to succeed.
ENTRYPOINT ["asrbench"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8765", "--allow-network"]
