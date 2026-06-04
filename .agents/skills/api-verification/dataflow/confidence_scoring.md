# Confidence scoring

Per-prediction confidence scoring. Given a `prediction_id` already in
ClickHouse, run the calibrated probe scorer to produce a float in
`[0, 1]` and persist it on the prediction row.

The contract is built around three operations on one logical
"confidence resource" identified by `prediction_id`:

1. **Enqueue** (idempotent) — `POST` ensures the score will exist.
2. **Read** — `GET` returns whatever state is currently visible.
3. **Stream** — SSE delivers the terminal event without polling.

The client SDK exposes all three via `ArbiterPrediction.confidence`
(blocking property) and the lower-level
`ModaicClient.{request,get,wait_for}_confidence_score` methods.

## Endpoints

| Method | Path | Status codes | Purpose |
| --- | --- | --- | --- |
| `POST` | `/api/v1/arbiters/predictions/{id}/confidence` | 200 / 202 / 404 | Idempotent enqueue. 200 if already complete, 202 if newly queued or in flight. |
| `GET` | `/api/v1/arbiters/predictions/{id}/confidence` | 200 / 404 | Read state without enqueueing. 404 if no POST has happened. |
| `GET` | `/api/v1/arbiters/predictions/{id}/confidence/stream` | 200 (SSE) / 404 / 403 | SSE — terminal event when complete. |
| `POST` | `/api/v1/arbiters/predictions/confidence` | 200 | Synchronous online scoring keyed by request body (no `prediction_id`). |

`ConfidenceStatusResponse`:

```json
{
  "status": "queued" | "completed" | "failed",
  "prediction_id": "uuid",
  "score": 0.83 | null,
  "error": "..." | null
}
```

## Server flow

### `POST /.../{id}/confidence` (idempotent enqueue)

1. Permission check — caller needs `read` on the arbiter repo. The
   handler discovers the repo via either Redis (if the prediction is
   still pending insert) or the ClickHouse row.
2. If the ClickHouse row already has `confidence IS NOT NULL` →
   return 200 `completed` with the cached score. No new Celery task.
3. Otherwise call `enqueue_confidence_job`:
   - `SETNX` on `pred:confidence:job:{prediction_id}` (24h TTL) so
     concurrent POSTs don't dispatch twice.
   - On gate-win, `score_prediction_confidence_task.delay(...)` and
     stash the Celery task id in the gate value.
4. Return 202 `queued`.

### `GET /.../{id}/confidence`

1. If row has confidence → 200 `completed`.
2. Else, look up the Celery task id from Redis. If missing → 404.
3. Resolve the task: `FAILURE`/`REVOKED` → `failed` with error;
   anything else → `queued`. Never returns `running` — clients only
   need to know "done or not".

### `GET /.../{id}/confidence/stream`

1. Subscribes to whatever the worker writes to (Redis pub/sub or
   similar — implementation may evolve).
2. Emits initial `status` event on connect (`queued` or `completed`).
3. Emits a terminal `completed` / `failed` event when the worker
   finishes. Heartbeats every `STREAM_KEEPALIVE_INTERVAL_S` seconds.
4. Closes after `STREAM_WALL_CAP_S` seconds with a non-terminal
   `status: queued` so the client can reconnect cleanly. The client
   honors this by reopening the stream until it hits a terminal event
   or its own deadline.

### `POST /api/v1/arbiters/predictions/confidence` (online)

Synchronous variant. The body carries the input + output dict; the
server runs `online_score(...)` inline and returns
`{"confidence": float}`. No `prediction_id`, no row update beyond what
`online_score` does internally. Use only when you don't have a stored
prediction.

## Invariants

1. **`POST` is idempotent.** Calling it N times on the same
   `prediction_id` produces at most one Celery task. The Redis SETNX
   gate (`pred:confidence:job:{id}`) is the source of truth — don't
   bypass it from any other code path.
2. **A `prediction_id` lives in two places.** Pending predictions
   (mid-batch, not yet flushed to ClickHouse) sit in Redis; persisted
   predictions live in ClickHouse. The handlers query both, in that
   order, before 404'ing.
3. **`completed` is terminal and idempotent.** Once
   `confidence IS NOT NULL` in ClickHouse, every subsequent POST/GET
   returns the cached score. Re-scoring is not a supported operation
   on this endpoint — you'd need to null out the column out-of-band.
4. **The stream always reconnects.** The server caps each SSE
   connection at ~120s with a non-terminal `status: queued` close.
   The client (`_drain_confidence_stream` in `client.py`) is
   responsible for reopening until terminal-or-timeout. If you change
   the cap, change `wait_for_confidence_score`'s reconnect window
   accordingly.
5. **The probe scorer is per-arbiter.** Pre-warming a probe container
   for arbiter A does not help arbiter B. Batch confidence (when
   wired up) needs to pre-warm one container per distinct arbiter in
   the batch.
6. **Error string truncation.** The handler caps error messages at
   500 chars when surfacing Celery failures. Don't rely on a full
   traceback making it through; log on the worker if you need detail.

## Common pitfalls

### `confidence` property hangs forever

**Symptom:** `ArbiterPrediction.confidence` never returns even though
the score completed server-side.

**Causes:**

1. The SSE connection is being held open by a proxy that strips
   heartbeats — the client's reconnect path doesn't help if the
   stream itself never closes. Force a shorter `timeout=` on
   `wait_for_confidence_score` or fall back to GET polling.
2. The `prediction_id` is wrong. The server returns 404 on the POST,
   but the property only raises if `status != "completed"` AND
   `score is None`. Verify the id round-trips through
   `request_confidence_score` first.

### Score appears in ClickHouse but `status` says `queued`

**Symptom:** `GET /confidence` returns `queued`; `SELECT confidence
FROM predictions WHERE id=...` shows a value.

**Cause:** Replication lag on the read replica. The handler queries
the read DB; the writer wrote to the primary. Retry after a few
seconds. If it persists, the issue is upstream of this endpoint.

### "Already in flight" returns 202 forever

**Symptom:** Multiple POSTs return 202; the Celery task seems gone.

**Cause:** The SETNX gate's TTL is 24h, but the Celery task itself
may have crashed without flipping the gate. The handler trusts the
gate. Workaround: GET surfaces `failed` once the AsyncResult resolves
to FAILURE — call GET, not POST, to detect stuck-and-failed jobs.
