# Alignment (GEPA + repredict)

"Alignment" is the end-to-end optimization flow: take an arbiter's
annotated train/test examples, run GEPA prompt optimization, push the
optimized prompt back to the arbiter repo as a new commit, then
re-predict on the test set with the new prompt and re-score
confidence on the resulting predictions. The whole thing surfaces to
the user as one job.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/jobs/gepa` | Start an alignment run. Returns `{job_id, status: "queued", repo, revision}`. |
| `GET` | `/api/v1/jobs/gepa/{job_id}` | Poll status. Returns `GepaJobResponse` with `status`, `phase`, `commit_hash`, `results`. |
| `DELETE` | `/api/v1/jobs/gepa/{job_id}` | Cancel a running optimization sandbox. |

`phase` values: `optimizing` → `re_predicting` → `scoring` → `done`.
`status` values: `in_progress` → `completed` | `failed`.

The client SDK does not yet expose this directly — the modaic-hub UI
and the `modaic-hub` jobs runner are the canonical callers. When
adding client support, mirror the polling behavior: GET the job until
`phase == "done"` AND `status in {"completed","failed"}`.

## Server flow

### `POST /api/v1/jobs/gepa`

1. Permission check — caller needs `write` / `admin` / `owner` on
   the arbiter repo.
2. Resolve env vars: user's encrypted env vars, overridden by org-
   level vars if the repo is org-owned. `MODAIC_TOKEN` is injected.
3. Build the GEPA train/val sets from ClickHouse:
   - `_make_dataset(arbiter_repo, "train", token)` — predictions with
     `split = 'train'` and non-null `ground_truth`.
   - `_make_dataset(arbiter_repo, "test", token)` — same with
     `split = 'test'`. (Yes, the *test* split is used as GEPA's
     `valset`. This is intentional — GEPA's `valset` drives Pareto
     selection, not held-out evaluation.)
4. Spin up a sandbox (`_start_gepa_sandbox`) with the trainset,
   valset, and `gepa_kwargs`. The sandbox runs GEPA, pushes the
   optimized program to the arbiter repo on `push_branch` with an
   optional `push_tag`, and reports back via Redis.
5. Persist sandbox metadata to Redis (`gepa:sandbox:{id}`, 24h TTL):
   user id, token, repo, slack flag, env vars. The polling endpoint
   needs all of these to trigger the repredict follow-up.

### `GET /api/v1/jobs/gepa/{job_id}` — phase composition

This endpoint is doing more than reporting status. It is the trigger
point for the second half of alignment.

1. Read GEPA result cache (`gepa:result:{id}`). On a hit, the
   underlying optimizer already finished.
2. If GEPA completed AND `commit_hash` is present AND no repredict
   has been triggered yet, call `_trigger_repredict`:
   - Redis `SETNX` gate (`gepa:repredict:{id}`) so concurrent polls
     race-but-only-one-wins.
   - Set `gepa:repredict:phase:{id}` = `"re_predicting"`.
   - Dispatch `repredict_and_score_task` (Celery) with the cached
     metadata.
   - Stash the task id in `gepa:repredict:task:{id}`.
3. Compose the response so the frontend keeps polling until both
   GEPA *and* repredict/scoring are done:
   - If `phase` not in Redis (no repredict needed, e.g. no
     improvement) → return `phase = "done"`.
   - If `phase = "done"` → return `status = "completed"`.
   - If `phase = "failed"` → return `status = "failed"` and surface
     the Celery task error if available.
   - Otherwise → override `status` back to `in_progress` with the
     current phase.

### Repredict + score worker

1. Re-run predictions for every annotated example on the new commit.
   Inserts new `predictions` rows (same `id`, bumped `version`).
2. Set `phase = "scoring"`. Enqueue confidence scoring for the new
   predictions (see `confidence_scoring.md`).
3. On completion, set `phase = "done"`, optionally fire a Slack
   notification.

## Invariants

1. **GEPA only sees rows where `split in ('train','test')` and
   `ground_truth IS NOT NULL`.** Anything stuck at `split='none'` is
   invisible — see `annotations.md` for why this is a recurring trap.
2. **The repredict trigger is idempotent.** Multiple concurrent polls
   of the GEPA status endpoint must dispatch at most one
   `repredict_and_score_task`. The `SETNX` gate is what enforces
   this — don't bypass it.
3. **Sandbox metadata has a 24h TTL.** If a user lets a job sit for
   over 24h before checking it, the repredict trigger will see no
   metadata and skip silently with a warning. This is a deliberate
   bound, not a bug.
4. **`status` in the response is composed, not raw.** The Celery /
   sandbox-level status alone is insufficient — the polling endpoint
   overlays Redis-tracked phase. Don't read GEPA Celery status
   directly from a client and assume the run is done.
5. **`gepa_kwargs.auto`, `max_full_evals`, and `max_metric_calls`
   are mutually exclusive.** Exactly one defines the budget. The
   schema does not enforce this — the sandbox does. Set only one.
6. **Reflection LM defaults matter.** `reflection_lm` defaults to
   `openai/gpt-5.2-2025-12-11`. The user's `OPENAI_API_KEY` env var
   on Modaic Hub must resolve, or the sandbox fails inside GEPA.

## Common pitfalls

### "GEPA done" but UI still spinning

**Symptom:** Status flips to `completed` briefly, then back to
`in_progress` with `phase = "re_predicting"` or `"scoring"`.

**Cause:** Working as intended. The status endpoint composes GEPA's
result with the repredict phase. The job is *truly* done when
`phase = "done"`. Wait for that, not for an intermediate `completed`.

### Repredict never fires

**Symptom:** GEPA completes; status sits at `phase = "done"` without
ever showing `re_predicting`. New commit hash is on the repo, but no
new predictions appear in ClickHouse.

**Causes (in order of likelihood):**

1. `commit_hash` was empty or absent in the GEPA result — sandbox
   reported "no improvement" and the trigger correctly skipped.
2. Sandbox metadata expired (24h TTL).
3. `_trigger_repredict` failed silently. Check worker logs for
   `[repredict] No metadata for sandbox ...` or task failures.

### `gepa_kwargs` not respected

**Symptom:** User passes a custom `reflection_minibatch_size`, but the
sandbox uses the default.

**Cause:** Pydantic's behavior when `GepaKwargs` is constructed from a
partial dict — only fields the user names override defaults. Verify
the request body has the right path: `{"gepa_kwargs": {"...": ...}}`
and not the field at the top level.
