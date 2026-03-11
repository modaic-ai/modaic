# modaic.batch

## Purpose

`modaic.batch` is the batch inference layer for Modaic + DSPy.

It has three main jobs:

1. Turn one or more `dspy.Predict` programs plus their examples into provider-specific batch requests.
2. Wait for provider completion and parse raw batch results back into DSPy predictions.
3. Persist returned rows as durable `ABatchResult` files inside `<modaic_cache>/batch`.

The top-level API is intentionally split:

- `modaic.batch.abatch(...)` is the grouped batch API.
- `modaic.Predict.abatch(...)` is the compatibility wrapper for the common single-predict case.

## Module Map

- `batch.py`
  Orchestration layer. Owns grouped input flattening, adapter selection, provider/client resolution, progress display, fallback retry logic, and result splitting.
- `types.py`
  Shared data model for `BatchRequest`, `ResultItem`, `ABatchRow`, `FailedPrediction`, and `ABatchResult`.
- `storage.py`
  Runtime storage layout under `<modaic_cache>/batch`.
- `clients/base.py`
  Provider-independent transport behavior such as cache checks, status polling, retry helpers, and JSONL creation.
- `clients/openai.py`, `clients/azure.py`, `clients/together.py`, `clients/anthropic.py`, `clients/fireworks.py`
  Provider-specific request formatting, status polling, result download, and raw-result parsing.
- `modal_client.py`
  Batch transport for Hugging Face models running on Modal.
- `modal_job.py`
  Remote Modal execution surface used by `ModalBatchClient`.
- `__init__.py`
  Lazy public exports.

## Runtime Storage

All runtime artifacts created by `modaic.batch` are stored under:

`<modaic_cache>/batch`

Layout:

- `<modaic_cache>/batch/tmp`
  Temporary transport files such as provider JSONL payloads and Modal parquet files.
- `<modaic_cache>/batch/<batch_id>/0.duckdb`
- `<modaic_cache>/batch/<batch_id>/1.duckdb`
- `<modaic_cache>/batch/<batch_id>/...`

Each `.duckdb` file is one persisted `ABatchResult` for one predict group from the grouped `abatch()` call. All files under the same `<batch_id>/` directory came from the same provider batch submission.

This layout replaced the older single-file `<batch_id>.duckdb` approach because grouped `abatch()` can now return multiple `ABatchResult` objects from one provider batch id.

## High-Level Flow

Grouped `abatch()` in `batch.py` does the following:

1. Accepts `inputs: list[tuple[dspy.Predict, list[dict]]]`.
2. Flattens all predict/example pairs into `BatchRequestContext` rows.
3. Assigns a stable global `request_id` like `request-0`, `request-1`, and so on.
4. Validates that the whole grouped call can run through one `BatchClient`.
5. Selects a `BatchAdapter` from the current DSPy adapter.
6. Formats one provider `BatchRequest`.
7. Submits one provider batch job.
8. Downloads raw results and sorts them back by `custom_id`.
9. Parses results into `dspy.Prediction` or `FailedPrediction`.
10. Optionally attaches `_messages` and `_outputs` when `return_messages=True`.
11. Splits rows back by original predict group.
12. Persists one `ABatchResult` file per predict group.

`modaic.Predict.abatch()` stays simple: it wraps `self` as `[(self, inputs)]`, calls the grouped API, and unwraps the first `ABatchResult`.

## Why There Is A `BatchRequestContext`

Grouped batching means one provider request can contain examples from multiple predictors.

That creates two requirements:

- The formatter must know which predictor owns each example so it can run that predictor’s `_forward_preprocess(...)`.
- The parser must replay postprocessing with the same predictor/signature pair that formatted the request.

`BatchRequestContext` is the glue for that. It carries:

- `predictor`
- `inputs`
- `group_index`
- `example_index`
- `request_index`
- `request_id`

This keeps request order, result merging, fallback retries, and final persistence deterministic.

## Adapter Layer

The adapter layer is DSPy-facing. It is responsible for prompt/message construction and DSPy output parsing. Provider clients do not know anything about DSPy signatures.

### `BatchAdapter`

`BatchAdapter` handles the normal one-pass path:

- `format(...)`
  Runs each predictor’s `_forward_preprocess(...)`, calls the active DSPy adapter’s preprocess/format hooks, and produces a common `BatchRequest`.
- `parse(...)`
  Replays the predictor’s preprocess logic for each row, then feeds provider text back through the DSPy adapter’s postprocess hooks.
- `_execute_rows(...)`
  Submits the request, collects results, normalizes ordering by `custom_id`, parses each row, and returns `ABatchRow` objects.

This split is deliberate. Provider clients should stay transport-oriented, while DSPy-specific prompt and parse rules stay centralized in `batch.py`.

### `BatchJSONAdapter`

Uses `dspy.JSONAdapter()` directly.

### `BatchChatAdapter`

Uses `dspy.ChatAdapter()` first, then retries only failed rows with `BatchJSONAdapter`.

Important behavior:

- The first pass is one giant grouped request.
- Failures across all predictors are collected into one aggregated second request.
- Retry merge happens by original flattened row index, not by predictor-local position.

This avoids N retry submits for N predictors and preserves the “one provider submit per pass” model.

## Client Layer

The client layer is provider-facing. Clients receive a provider-agnostic `BatchRequest` and handle transport details.

Shared responsibilities in `clients/base.py`:

- create JSONL payloads when providers want uploaded files
- check `dspy.cache` before submitting
- avoid resubmitting already-cached rows
- poll provider status
- send `status_callback` updates
- normalize retry behavior for transient network errors

Provider-specific responsibilities:

- request formatting
- provider job creation
- provider status mapping
- raw result download
- raw result parsing into `ResultItem`

The key architectural rule is:

- Adapters understand DSPy.
- Clients understand providers.

## Provider Validation Rules

Grouped `abatch()` must be able to run through one concrete client instance.

Current rules:

- Without an explicit `client=...`, all predictors in the grouped call must resolve to the same provider.
- If predictors carry explicit API keys, they must agree unless the caller passes a concrete client.
- With an explicit non-Modal client, every predictor must resolve to that same provider.
- With `ModalBatchClient`, every predictor must use a `huggingface/...` model and must match the client LM model.

There is also a provider-specific batch limitation worth keeping in mind:

- Fireworks requires one model for the whole batch job. The common `BatchRequest` tracks `model=None` when requests disagree so transports can reject unsupported mixed-model batches explicitly.

## Result Persistence

`ABatchResult` is intentionally file-backed instead of fully in-memory.

Each stored row contains:

- parsed prediction payload
- original request messages
- raw text/reasoning outputs
- original input example

`ABatchResult` supports:

- iteration
- random access via `__getitem__`
- lazy reconstruction of rows from disk

That keeps large batch outputs usable without loading the entire result set into Python objects immediately.

## Why Parquet For Modal Transport

Modal does not use the JSONL upload path that the hosted batch APIs use.

Instead, `ModalBatchClient` writes the request messages to parquet, uploads the parquet file into the Modal volume, runs the remote job, then downloads parquet output back.

This was a deliberate design choice:

- parquet is compressed and more compact than naive JSON for large message batches
- parquet is efficient for pandas-based read/write on both sides
- parquet is a better fit for shipping a table of rows through the network boundary to Modal
- the request and response datasets are naturally columnar: one row per batch item, one messages/response column per row

In practice that makes the Modal handoff smaller and faster than pushing equivalent raw JSON files around.

## Why DuckDB For Final `ABatchResult`

DuckDB is used for the persisted result files because the access pattern is different from the Modal transport pattern.

The final result store needs:

- fast append/write at the end of a batch
- cheap reopen from disk
- efficient row iteration
- efficient random access for `result[i]`
- good behavior with memory-mapped local file access

DuckDB fits that well. It gives us a compact local file with strong read ergonomics for post-batch analysis, retries, and inspection without forcing everything into memory.

Parquet is the transport format for Modal.
DuckDB is the final user-facing result format for `ABatchResult`.

Those are different jobs, so they use different storage choices.

## Progress And Status

When `show_progress=True`, `batch.py` uses `BatchProgressDisplay` with Rich.

Progress callbacks flow like this:

1. the orchestration layer creates a display object
2. the display’s `update(...)` method is passed into the client
3. the client emits provider status updates
4. the display updates both the terminal panel and the user-supplied callback

Modal is treated specially because its client already executes synchronously from the local process perspective and does not use the same remote polling loop.

## Cache Behavior

`BatchClient.submit(...)` checks `dspy.cache` row-by-row before sending anything upstream.

If some rows are already cached:

- cached rows are not resubmitted
- uncached rows are sent as the real provider request
- the client remembers enough metadata to merge cached and uncached results later

If everything is cached:

- no provider submit happens
- a synthetic cached batch id is returned

This is why request ids and per-row ordering must stay stable.

## Design Decisions

- Grouped `abatch()` is top-level.
  The grouped API belongs in `modaic.batch.abatch()` because it can combine multiple predictors into one provider request. `modaic.Predict.abatch()` remains a convenience wrapper.
- Adapter and client responsibilities are split.
  DSPy prompt/parse logic stays in the adapter layer; provider transport logic stays in the client layer.
- Fallback retry is aggregated.
  `ChatAdapter` fallback collects every failed row across all predictors into one second request instead of submitting retries per predictor.
- Final results are persisted per predict group.
  The grouped API returns one `ABatchResult` per original predict/input group, even when they share the same provider batch id.
- Runtime artifacts live in one batch cache tree.
  Everything generated by the batch subsystem is stored under `<modaic_cache>/batch` so cleanup and debugging stay localized.

## Future Direction: Lazy Inputs

The current grouped API still requires a fully materialized:

`list[tuple[dspy.Predict, list[dict]]]`

That is simple and deterministic, but it forces callers to build the whole request set in memory up front.

The intended future direction is to support lazy input feeding into `abatch()`.

Likely requirements for that design:

- accept iterables or async iterables of examples instead of only lists
- preserve stable global request ordering and reproducible request ids
- spill formatted requests to disk incrementally instead of materializing the entire flattened list in memory
- keep retry bookkeeping so failed rows can still be mapped back to the owning predictor/example
- preserve one-provider-submit semantics where the upstream API requires a complete uploaded file
- support chunked staging for very large batches without changing `ABatchResult` semantics

One plausible direction is:

- incrementally flatten predictor/example rows
- incrementally format them into disk-backed request manifests
- finalize one provider upload file at submit time
- retain a sidecar manifest mapping request id to predictor/group/example metadata

That would let callers generate examples lazily while keeping the current deterministic result reconstruction model.

## Editing Guidance

If you change this subsystem, verify at least these invariants:

- grouped `abatch()` still issues one initial provider submit for the whole grouped call
- `modaic.Predict.abatch()` still behaves like the legacy single-predict API
- request ids remain stable and sortable by `request-{n}`
- grouped fallback still retries all failures in one aggregated second pass
- runtime files still live under `<modaic_cache>/batch`
- grouped results still persist to `<batch_id>/<predict_index>.duckdb`
