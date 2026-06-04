# Annotations

How a prediction row becomes "annotated" — meaning a human has supplied
ground truth — and how it gets a `split` so downstream consumers (GEPA,
scoring, dataset export) can find it.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/examples` | Ingest one or more predicted examples (NDJSON body). Optionally pre-annotated. |
| `PATCH` | `/api/v1/examples/{example_id}/annotation` | Add or update ground truth on an existing example, per arbiter. |
| `GET` | `/api/v1/examples/{example_id}` | Read a single example. |
| `GET` | `/api/v1/examples` | Page through examples for a repo. |

Client surface:

- `ModaicClient.ingest_examples`
- `Arbiter.annotate_example` → `ModaicClient.annotate_example`
- `Arbiter.list_examples` / `Arbiter.get_example`

## Client → server flow

### Ingest (`POST /api/v1/examples`)

1. Client sends NDJSON — one JSON object per line, no `Content-Type:
   application/json`. Each row is a `PredictedExample` (see
   `examples/schemas.py`).
2. Server validates each line as `PredictedExample`. The
   `model_validator(mode="after")` named `get_split` runs:
   - If `ground_truth is None` → `split = "none"`.
   - Else if `split is None or "none"` → `split = "train" if random()<0.8 else "test"`.
3. Rows are queued for the ClickHouse writer. The writer treats the
   payload's `split` as authoritative.

### Annotate (`PATCH /.../annotation`)

1. Client sends `AnnotateExampleRequest` — a list of
   `PredictionAnnotation` objects (`arbiter_repo`, `ground_truth?`,
   `ground_reasoning?`).
2. Server fetches the latest non-deleted prediction for each
   `(example_id, arbiter_repo)`.
3. **Split assignment lives in `_annotate_example`** (not the
   validator): if `existing.ground_truth is None and
   annotation.ground_truth is not None`, the helper rolls a fresh
   train/test split. Otherwise it preserves `existing.split`.
4. The helper inserts a new ClickHouse row (same `id`, bumped
   `event_ts`) with the merged ground truth and split.

## Invariants

1. **`split ∈ {"train", "test", "none"}` in storage; `{"train", "test"}` everywhere else.**
   `"none"` is the sentinel for unannotated rows. Every downstream
   consumer filters on `split in ('train','test')`. A row stuck at
   `"none"` is invisible.
2. **The client never sends `"none"` and never sends `null`.** Omit
   the key. The server-side validator decides.
3. **Split, once assigned, is immutable.** Re-annotating an already-
   annotated row preserves its split. This is what keeps train/test
   reproducible across re-labeling passes.
4. **Two split-assignment sites must agree.**
   - `examples/schemas.py::PredictedExample.get_split` — for
     `POST /api/v1/examples` ingest.
   - `examples/utils.py::_annotate_example` — for the PATCH path.
   - `jobs/schemas.py::BatchExample.get_split` — for the batch
     predictions path.
   If you change the ratio in one, change it in all three. There is no
   shared helper today; that is technical debt, not a feature.
5. **Annotations are scoped to `(example_id, arbiter_repo)`.** One
   example may have predictions from multiple arbiters; an annotation
   only updates the rows for the arbiter it names.

## Common pitfalls

### Annotated rows land with `split='none'`

**Symptom:** Examples annotated via the batch predictions path never
appear in GEPA train/test datasets, never get scored, never export.

**Cause:** `BatchExample` originally had no `get_split` validator —
`batch_utils.py` wrote `ex.split` (defaulting to `"none"`) verbatim into
ClickHouse. The validator was added; if you create another path that
flows annotated rows into ClickHouse, replicate the validator.

### Client sends `split: "none"` explicitly

**Symptom:** Same as above. Pydantic accepts the value; the validator
doesn't reroll because the payload looked intentional.

**Cause:** Older client versions defaulted `ex.get("split", "none")`.
Don't put `"none"` on the wire. Don't set it to `null`. Omit the key.

### Annotation request silently no-ops

**Symptom:** PATCH returns 200 but the row's `ground_truth` doesn't
change.

**Cause:** `PredictionAnnotation.arbiter_repo` doesn't match any
existing prediction for that `example_id`. `_annotate_example` skips
unmatched rows and only commits if at least one matched. Verify the
arbiter_repo string before reporting a server bug.
