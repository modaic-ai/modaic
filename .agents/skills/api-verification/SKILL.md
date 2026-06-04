---
name: api-verification
description: Reference for how each end-to-end Modaic dataflow is *intended* to work across client (modaic) and server (modaic-dev). Use this skill BEFORE planning or implementing any change that touches the API surface — predictions, annotations, alignment (GEPA + repredict), or confidence scoring. The dataflow docs capture the contract and the cross-repo gotchas (split semantics, idempotency keys, validator placement) that are not visible from one side of the wire alone.
---

# API verification

Modaic's public API is split across two repos:

- **Client** (`core/modaic`) — Python SDK, `modaic_client.ModaicClient`,
  schemas the user constructs.
- **Server** (`core/modaic-dev/server`) — FastAPI routes, ClickHouse
  writes, Celery / Redis / sandbox orchestration.

Every public dataflow has invariants that span both sides. A change on
one side that ignores the contract on the other silently corrupts data
(e.g. annotated rows landing with `split='none'` and being skipped by
GEPA forever). This skill is the one place those contracts live.

## When to consult this skill

Invoke it whenever you are about to:

- **Plan** a change that crosses the wire — anything where the client
  builds a request body, hits an endpoint, or parses a response. Read
  the relevant dataflow doc first; reconcile your plan against it.
- **Add or modify code** in any of:
  - `src/modaic-client/modaic_client/client.py` (any method that does
    HTTP)
  - `src/modaic-client/modaic_client/schemas.py`
  - `modaic-dev/server/src/api/v1/**` route handlers, schemas, utils
  - `modaic-dev/server/src/api/v1/jobs/**` (batch / GEPA / repredict)
- **Review a PR** that touches the surface above. Use the docs as the
  acceptance bar — the contract is what merges, not the diff.

If the change is purely internal to the server (no public-API
implication) or purely client-side (e.g. adding a tqdm option), you do
not need the skill — but a 30-second sanity check costs nothing.

## How to use the docs

Each `dataflow/*.md` describes one end-to-end path:

| File | Path |
| --- | --- |
| [`dataflow/annotations.md`](dataflow/annotations.md) | How examples become annotated and how `split` is assigned. |
| [`dataflow/alignment.md`](dataflow/alignment.md) | GEPA optimization + the repredict-and-score follow-up. |
| [`dataflow/predictions.md`](dataflow/predictions.md) | Single (`/v2/arbiters/predictions`) and batch (`/jobs/batch/predictions`) inference. |
| [`dataflow/confidence_scoring.md`](dataflow/confidence_scoring.md) | Per-prediction confidence scoring (POST/GET/SSE). |

Each doc follows the same shape:

1. **Endpoints** — exact paths, methods, request/response models.
2. **Client → server flow** — what the client sends, what the server
   does on receipt, what it stores.
3. **Invariants** — the rules that must hold across the wire. Bugs in
   this area always trace back to one of these being violated.
4. **Common pitfalls** — historical foot-guns, with the symptom and the
   fix.

## Updating the docs

The docs are the source of truth for intended behavior. If you discover
that the code disagrees with a doc:

- If the doc is wrong, update it.
- If the code is wrong, fix the code AND add a "Common pitfall" entry
  with the symptom so the next agent doesn't reintroduce the bug.

Do not delete a "Common pitfall" entry just because the bug is fixed —
that's exactly the institutional memory the skill exists to preserve.

When you add a new public dataflow (a new endpoint family or a new
job type), add a new file under `dataflow/` and link it from the table
above.
