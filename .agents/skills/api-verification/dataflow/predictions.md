# Predictions (single + batch)

Two paths produce predictions. They share a write target (the
`predictions` table in ClickHouse) and an annotation contract, but the
mechanics differ.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v2/arbiters/predictions` | Synchronous single prediction. Optionally pre-warms a confidence probe. |
| `POST` | `/api/v1/jobs/batch/predictions` | Async batch — Cartesian product of arbiters × examples. Returns a `job_id`. |
| `GET` | `/api/v1/jobs/batch/predictions/{job_id}` | Coarse status (`PENDING` / `PROGRESS` / `SUCCESS` / `FAILURE` / `REVOKED`). |
| `GET` | `/api/v1/jobs/batch/predictions/{job_id}/events` | SSE stream — `started`, `phase`, `prediction_completed`, `prediction_failed`, `done`. |
| `GET` | `/api/v1/jobs/batch/predictions/{job_id}/results` | NDJSON — one example per line, with all per-arbiter predictions. |
| `DELETE` | `/api/v1/jobs/batch/predictions/{job_id}` | Cancel a running job. Synthesizes a `done`/`aborted` event. |

Client surface:

- `Arbiter.predict` / `Arbiter.__call__` → `ModaicClient.predict`
- `Arbiter.predict_all` / `ModaicClient.predict_all` → returns a
  `BatchJob` and waits via SSE → polling fallback.

## Client → server flow

### Single (`POST /api/v2/arbiters/predictions`)

1. Client posts `{input, arbiter_repo, arbiter_revision, ground_truth?,
   ground_reasoning, compute_confidence}`.
2. Server resolves the arbiter, runs the LM call inline, persists one
   `examples` row + one `predictions` row to ClickHouse.
3. Response includes `example_id`, `prediction_id`, `output`,
   `reasoning`, `messages`. The client wraps it as `ArbiterPrediction`.
4. If `compute_confidence=true`, the server pre-warms a probe scorer
   container and enqueues confidence scoring. The response itself does
   not block on the score — fetch it via the confidence endpoints.

### Batch (`POST /api/v1/jobs/batch/predictions`)

1. Client posts `{arbiters: list[BatchArbiter], examples:
   list[BatchExample] | None, example_ids: list[str] | None,
   compute_confidence?}`. Exactly one of `examples` or `example_ids`
   must be set. Limits: 1–5 arbiters, 1–1000 rows.
2. The route validates write access on every arbiter repo, resolves
   per-org env vars, and (when `example_ids` is set) fail-fast 400s if
   any `(id, arbiter_repo)` pair is missing in ClickHouse. Dispatches a
   Celery task. Returns `{job_id, event: "start", status: "predicting",
   arbiters, total}`.
3. Worker drives a `BatchPredictionsJob` and forwards each yielded
   snapshot to (a) a Redis Stream for SSE and (b) the Celery `meta` for
   polling. Phases: `predicting` (load → run via `dspy.Parallel` →
   one-transaction CH persist) → optionally `scoring` (per-arbiter
   Modal scoring filtered to this job's `prediction_ids`) → terminal
   `done`/`failed`.
4. Snapshots have a single shape: `{job_id, event, status,
   predictions_progress, scores_progress, results, error, ts}`. SSE
   forwards this shape verbatim (per-prediction payloads stay
   server-side; clients fetch the canonical list via `/results`). The
   GET status endpoint returns the latest snapshot in the same shape.
5. Client (`BatchJob.wait`) opens the SSE stream, drives a tqdm bar
   from `predictions_progress` / `scores_progress` counters, returns
   when the awaited milestone fires (`wait_for="predictions"` or
   `"scores"`). Falls back to polling `/{job_id}` if SSE 404s or drops.
   Pass `wait_for=None` to get a `BatchJob` handle back without waiting.

## Invariants

1. **`split` on `BatchExample`** follows the annotations contract.
   Never send `"none"`, never send `null`. Omit the key. The
   `BatchExample.get_split` validator will randomly assign train/test
   (80/20) when `ground_truth` is present.
2. **One `example_id` per example, shared across all arbiters in the
   job.** One `prediction_id` per `(example, arbiter)` pair. The job
   row table maps `(job_id, position, arbiter_index)` → both ids so
   results stream in stable order.
3. **The batch worker writes `ex.split` verbatim to ClickHouse.** It
   does not re-derive split. If a future schema bypasses the
   `BatchExample` validator (e.g. a different request shape), it must
   either reuse the validator or replicate the assignment logic before
   inserting — `batch_utils.py` will not save you.
4. **Cancellation is best-effort.** `DELETE` on a running job revokes
   the Celery task; if the worker died before emitting `done`, the
   server synthesizes a terminal `done`/`aborted` event so the SSE
   client doesn't hang.
5. **Result streaming preserves arbiter order.** The NDJSON rows list
   `predictions` in the same order as `arbiters` in the original
   request. `BatchJob._arbiter_for_index` relies on this; don't reorder
   server-side.
6. **Batch confidence scoring is filtered to *this job*.** When
   `compute_confidence=True`, the worker passes the prediction_ids it
   just produced to Modal, so scoring touches only those rows — not
   any other NULL-confidence rows that may exist in the repo.

## Common pitfalls

### Annotated batch examples never reach GEPA / scoring

**Symptom:** A batch run with `ground_truth` populated completes
successfully, but the rows show `split='none'` in ClickHouse and never
get picked up by GEPA train/test extraction.

**Cause:** `BatchExample` had no `get_split` validator; the worker
wrote the default `"none"` straight through. Fixed — the validator now
mirrors the one on `PredictedExample`.

**How to reproduce a regression:** Send a batch with explicit
`split: "none"` and a non-null `ground_truth`. If those rows appear in
the train/test splits in ClickHouse, the validator is intact. If they
land as `"none"`, the validator is broken (most likely someone removed
it, or added a parallel ingest path that bypasses it).

### SSE bar finishes before results are ready

**Symptom:** tqdm shows 100% and then `wait()` hangs for a moment.

**Cause:** Expected. `event="prediction"` snapshots fire from the
worker's parallel pool; the actual ClickHouse insert happens in a
single transaction after the parallel loop drains. The terminal
`event="finish"` snapshot is emitted after the persist (and, when
`compute_confidence=True`, after Modal scoring finishes). Don't try
to short-circuit `wait()` on bar completion.

### `get_modaic_client()` returns stale token

**Symptom:** First call after `configure_modaic_client(...)` works;
later calls 401.

**Cause:** The httpx client is created per-context inside `get_client`
using `self.modaic_token`. If you swap the singleton's token on the
fly, only callers that re-acquire via `get_modaic_client()` see the
new value. Use `configure_modaic_client(...)` to replace the
singleton, not direct attribute writes.
