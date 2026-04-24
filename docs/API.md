# ASRbench API reference

Auto-generated from the live FastAPI OpenAPI schema (version `0.1.0`).
Regenerate with `python scripts/gen_api_docs.py`.

## datasets

### `GET /datasets`

**List Datasets**

*Tags:* datasets

List all registered datasets, optionally filtered by language and/or source.

*Responses:* 200, 422

### `POST /datasets/fetch`

**Fetch Dataset**

*Tags:* datasets

Start fetching a dataset in the background.

*Responses:* 202, 422

### `DELETE /datasets/{dataset_id}`

**Delete Dataset**

*Tags:* datasets

Delete a dataset row and, optionally, its cached audio files.

Returns 409 when at least one run references the dataset — delete or
retry those runs first to avoid orphaning them.

*Responses:* 204, 422

### `GET /datasets/{dataset_id}`

**Get Dataset**

*Tags:* datasets

Return a single dataset's metadata.

*Responses:* 200, 422


## models

### `GET /models`

**List Models**

*Tags:* models

List all registered models.

*Responses:* 200

### `POST /models`

**Register Model**

*Tags:* models

Register a new model. Idempotent — returns existing if name + backend match.

*Responses:* 201, 422

### `POST /models/{model_id}/load`

**Load Model**

*Tags:* models

Load a model into GPU/CPU memory.

*Responses:* 200, 422

### `POST /models/{model_id}/unload`

**Unload Model**

*Tags:* models

Unload a model from memory.

*Responses:* 200, 422


## optimize

### `GET /optimize/`

**List Studies**

*Tags:* optimize

List optimization studies, optionally filtered by status.

*Responses:* 200, 422

### `POST /optimize/global-config`

**Start Global Config**

*Tags:* optimize

Start a two-stage IAMS run aggregated across N datasets.

Every trial evaluates the candidate config on ALL listed datasets,
weights the per-dataset scores, and returns one aggregate. IAMS's
7-layer algorithm then produces a single global config that
minimises the weighted mean across the fleet — use this when
deploying to a product with a single shared preset.

*Responses:* 202, 422

### `POST /optimize/start`

**Start Study**

*Tags:* optimize

Create a single-dataset optimization study and kick off IAMS.

*Responses:* 202, 422

### `POST /optimize/two-stage`

**Start Two Stage**

*Tags:* optimize

Start a coarse→fine two-stage optimization run.

Creates two ``optimization_studies`` rows (one per stage) and runs
both in sequence via the library's ``run_two_stage`` orchestrator.
Budget/epsilon default to ``None`` — the library sizes them from the
space and the stage duration when left unset.

*Responses:* 202, 422

### `GET /optimize/{study_id}`

**Get Study**

*Tags:* optimize

Return a single study's metadata and final result (if completed).

*Responses:* 200, 422

### `POST /optimize/{study_id}/cancel`

**Cancel Study**

*Tags:* optimize

Force-cancel a stuck running or failed study.

Marks status as 'cancelled' so a new study can start. Safe to call
on a study whose background task has already died.

*Responses:* 200, 422

### `GET /optimize/{study_id}/trials`

**List Trials**

*Tags:* optimize

Paginated access to the full trial log for audit / UI replay.

*Responses:* 200, 422


## runs

### `GET /runs`

**List Runs**

*Tags:* runs

List runs with optional status/lang filters and a result cap.

*Responses:* 200, 422

### `GET /runs/compare`

**Compare Runs**

*Tags:* runs

Compare metrics of two or more runs side by side.

*Responses:* 200, 422

### `POST /runs/start`

**Start Run**

*Tags:* runs

Start a benchmark run in the background.

*Responses:* 202, 422

### `DELETE /runs/{run_id}`

**Delete Run**

*Tags:* runs

Delete a run and all its segments/aggregates. Rejects in-flight runs.

*Responses:* 204, 422

### `GET /runs/{run_id}`

**Get Run**

*Tags:* runs

Return a single run with aggregate metrics.

*Responses:* 200, 422

### `POST /runs/{run_id}/cancel`

**Cancel Run**

*Tags:* runs

Request cancellation of an in-progress run.

*Responses:* 200, 422

### `GET /runs/{run_id}/export`

**Export Run**

*Tags:* runs

Export a run's full results as JSON or CSV.

*Responses:* 200, 422

### `POST /runs/{run_id}/retry`

**Retry Run**

*Tags:* runs

Re-run a failed or cancelled run with the same params; creates a new run row.

*Responses:* 202, 422

### `GET /runs/{run_id}/segments`

**Get Segments**

*Tags:* runs

Paginated segment results for a run.

*Responses:* 200, 422


## system

### `GET /system/health`

**Health**

*Tags:* system

Liveness probe — always returns ok.

*Responses:* 200

### `GET /system/vram`

**Vram**

*Tags:* system

Report GPU VRAM usage. Returns empty list if no GPU is available.

*Responses:* 200
