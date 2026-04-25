---
name: modaic-hub
description: >
  How modaic-hub is laid out and how to run its two main jobs — align (GEPA prompt
  optimization) and score (confidence scoring). Use when working in modaic-dev,
  touching the jobs API, debugging a stuck align/score run, or explaining the
  hub's server/client/SDK split to someone new.
---

modaic-hub
==========

modaic-hub is Modaic's product for hosting **arbiters** and **probes** in
git-backed repos and running async jobs against them. The two user-visible jobs
are:

- **align** — run GEPA to optimize an arbiter's prompts against labeled
  train/test predictions. Runs in a Modal sandbox.
- **score** — compute confidence scores (via the trained probe) for every
  prediction in the repo whose `confidence` is still `NULL`. Runs as a Celery
  task.

Both are kicked off by the web UI today; there is no first-class CLI or SDK
helper — users (or you, debugging) hit the FastAPI endpoints directly.

Where stuff is
--------------

Everything below is rooted at `/Users/tytodd/Desktop/Modaic/code/core/`.

```
modaic-dev/                       # the hub product (server + client + infra)
  server/src/
    api/v1/jobs/
      index.py                    # POST/GET/DELETE for /jobs/gepa and /jobs/confidence-scores
      schemas.py                  # GepaJobRequest, GepaKwargs, ConfidenceScoreJobRequest, GepaJobResponse
      gepa_client.py              # HTTP shim that talks to the Modal GEPA sandbox
      gepa_utils.py               # builds train/val dspy.Examples from ClickHouse + the gepa_metric
      probe_client.py             # HTTP shim to the runtime probe (embeddings + confidence)
      utils.py                    # create_confidence_scores() — the score job's actual loop
    workers/celery.py             # submit_confidence_scores_task — the Celery task
    api/v1/envvars/service.py     # decrypted env vars passed into Modal sandboxes
    lib/gitea.py                  # gitea_client — perms, file reads, branch/tag pushes
    db/{pg,ch,redis}.py           # Postgres / ClickHouse / Redis sessions
    service/slack_notify.py       # job completion notifications
  client/src/
    components/repo/tabs/
      annotation-dialogs.tsx      # "Align Model" + "Score Annotations" buttons (lines ~70-189)
      annotation-job-status.tsx   # polling status bars
    hooks/annotation.ts           # useStartOptimizationJob / useStartConfidenceScoring etc.
  gitea/                          # the bundled Gitea instance (arbiter repos live here)
  gpu/, modal/, services/         # external worker glue

modaic/                           # the SDK + user-facing docs
  src/modaic-sdk/modaic/hub.py    # client-side hub.* helpers (push/pull repos, auth)
  src/modaic-client/              # lower-level HTTP client
  docs/docs/modaic_hub/           # user docs (currently only repos.mdx + bundling_a_program.mdx)
  docs/docs/arbiters/
    aligning_your_arbiter.mdx     # conceptual guide for the align workflow
    confidence_scoring.mdx        # conceptual guide for confidence/probes

mo-cli/                           # internal benchmarking CLI — NOT user-facing for jobs
```

Storage map:
- **Gitea** holds arbiter source (`owner/repo`, branches, tags).
- **ClickHouse** holds predictions, examples, embeddings — both jobs read/write
  here.
- **Postgres** holds users, orgs, repo metadata, encrypted env vars.
- **Redis** is used for the Celery broker, the example flush queue, and GEPA
  job metadata / Slack-dedup keys (`gepa:sandbox:{id}`, `gepa:notified:{id}`,
  24h TTL).
- **Modal sandboxes** run the GEPA optimizer (one sandbox per align job, so
  cancel = sandbox terminate).

Auth + permissions
------------------

Both endpoints require a session (`managerv2.required`) and check that the
caller has `write | admin | owner` on the arbiter repo via
`gitea_client.get_authenticated_user_permissions(...)`. Without it you'll see
`ForbiddenError`. The session token is extracted with
`extract_session_user_token(session_user)` and forwarded to Gitea / Modal so
the job can pull the repo and push results back.

Slack notification on completion is opt-out via the repo's `README.md`
frontmatter:

```yaml
---
slack_notifications: false
---
```

Default is `true`. Read by `_get_slack_notifications_flag()` in
`server/src/api/v1/jobs/index.py`.

---

Align job (GEPA optimization)
-----------------------------

**What it does.** Pulls all annotated predictions for the repo from ClickHouse
(`split = "train"` and `split = "test"`, `ground_truth IS NOT NULL`,
`is_deleted = 0`, latest version per id), wraps them as `dspy.Example`s, spawns
a Modal sandbox, and runs GEPA to evolve the arbiter's prompts. The metric is
exact-match equality between the predicted output dict and `ground_truth`
(`gepa_metric` in `gepa_utils.py:70`). On success the sandbox pushes optimized
prompts to `push_branch` (and tags `push_tag` if set) on the same Gitea repo.

**Endpoint.**

```
POST /api/v1/jobs/gepa
```

Body — `GepaJobRequest` (see `server/src/api/v1/jobs/schemas.py`):

```json
{
  "arbiter_repo": "owner/repo",         // required, must contain "/"
  "arbiter_revision": "main",           // branch/tag/sha to optimize from
  "push_branch": "main",                // where to push optimized prompts
  "push_tag": null,                     // optional tag on success
  "gepa_kwargs": {
    "auto": "light",                    // "light" | "medium" | "heavy" | null
    "max_full_evals": null,             // budget — set ONE of auto/full_evals/metric_calls
    "max_metric_calls": null,
    "reflection_minibatch_size": 3,
    "candidate_selection_strategy": "pareto",   // or "current_best"
    "reflection_lm": {"model": "openai/gpt-5.2-2025-12-11", "kwargs": null},
    "skip_perfect_score": true,
    "add_format_failure_as_feedback": false,
    "component_selector": "round_robin",         // or "all"
    "use_merge": true,
    "max_merge_invocations": 5,
    "seed": 0,
    "gepa_kwargs": null                 // arbitrary GEPA passthrough
  }
}
```

Response: `{"job_id": <modal_sandbox_id>, "status": "queued", "repo": ..., "revision": ...}`.

**Status, logs, cancel.**

```
GET    /api/v1/jobs/gepa/{job_id}        -> GepaJobResponse {status, message, results}
GET    /api/v1/jobs/gepa/{job_id}/logs   -> {logs: "<raw modal stdout/stderr>"}
DELETE /api/v1/jobs/gepa/{job_id}        -> cancels sandbox
```

`status` is `"in_progress" | "completed" | "failed"`. The status endpoint also
fires the Slack notification on terminal states (deduplicated via Redis).

**Example via curl.**

```bash
curl -X POST "$MODAIC_HUB/api/v1/jobs/gepa" \
  -H "Cookie: $SESSION_COOKIE" \
  -H "Content-Type: application/json" \
  -d '{
    "arbiter_repo": "tyrin/my-arbiter",
    "arbiter_revision": "main",
    "push_branch": "gepa/auto",
    "gepa_kwargs": {"auto": "medium", "seed": 42}
  }'
```

**Common gotchas.**

- The dataset is built fresh on every request, so a job with no `train` /
  `test` annotations in ClickHouse will still start but immediately fail in
  the sandbox — check `make_dataset` in `gepa_utils.py:17` if the count looks
  wrong.
- Cancel = `sandbox.terminate()` via the `-cancel.modal.run` endpoint in
  `gepa_client.py`. The Postgres/Redis state is not cleaned up — the
  `gepa:sandbox:{id}` key just expires after 24h.
- For org-owned repos, env vars are merged: user's vars first, then the org's
  vars override (see `start_optimization_job` in `index.py:192-197`).

---

Score job (confidence scoring)
------------------------------

**What it does.** Finds every prediction in the repo with `confidence IS NULL`
(deduped to the latest version per id), pulls the message arrays, and calls
the runtime probe via Modal to get `(embedding, confidence)` for each. Writes
the new rows back to ClickHouse. Reports progress to the Celery task state so
the UI can show a progress bar.

The actual loop is `create_confidence_scores()` in
`server/src/api/v1/jobs/utils.py:16`. The Celery wrapper is
`submit_confidence_scores_task` in `server/src/workers/celery.py:81`.

**Endpoint.**

```
POST /api/v1/jobs/confidence-scores
```

Body — `ConfidenceScoreJobRequest`:

```json
{ "arbiter_repo": "owner/repo" }
```

Response: `{"job_id": <celery_task_id>, "status": "queued", "repo": ...}`.

**Status, cancel.**

```
GET    /api/v1/jobs/confidence-scores/{job_id}
DELETE /api/v1/jobs/confidence-scores/{job_id}
```

Status payload mirrors Celery state:

```json
{
  "job_id": "...",
  "status": "PENDING" | "PROGRESS" | "SUCCESS" | "FAILURE",
  "progress": 42,           // present when status == "PROGRESS"
  "current": 7, "total": 16,
  "logs": ["batch 7/16 done", ...],
  "result": { "processed": 32, "total_found": 32, ... },   // when SUCCESS
  "error": "..."                                            // when FAILURE
}
```

DELETE calls `AsyncResult.revoke(terminate=True)` — hard kill of the worker
process running that task.

**Example via curl.**

```bash
curl -X POST "$MODAIC_HUB/api/v1/jobs/confidence-scores" \
  -H "Cookie: $SESSION_COOKIE" \
  -H "Content-Type: application/json" \
  -d '{"arbiter_repo": "tyrin/my-arbiter"}'
```

**Common gotchas.**

- The job first **flushes the Redis example queue into ClickHouse** before
  querying — newly-annotated examples not yet flushed will still get scored
  this run.
- "Nothing happened" usually means there were zero predictions matching the
  filters. The diagnostic block at the top of `create_confidence_scores`
  prints counts for each filter (total, non-deleted, with messages, null
  confidence) — read those logs.
- The probe endpoint is per-arbiter — if no probe has been trained for the
  repo yet, `probe_modal()` errors out and the task fails. Check
  `probe_client.py` for the upstream call.

---

Quick "where do I look if X is broken" cheatsheet
-------------------------------------------------

| Symptom                                  | Look at                                                           |
| ---                                      | ---                                                               |
| 403 starting either job                  | `gitea_client.get_authenticated_user_permissions` in `index.py`   |
| Align job hangs in `queued` forever      | `gepa_client.start_gepa_sandbox` + Modal dashboard                |
| Align job fails immediately              | `make_dataset` in `gepa_utils.py` — likely empty trainset         |
| Score job says SUCCESS but nothing changed | The diagnostic counts in `create_confidence_scores` (utils.py)  |
| Score job stuck in PROGRESS              | Celery worker logs; `submit_confidence_scores_task` in `celery.py`|
| No Slack notification fired              | README frontmatter `slack_notifications`, `_maybe_send_gepa_notification`, `gepa:notified:*` Redis key |
| UI button does nothing                   | `client/src/hooks/annotation.ts` — check the mutation hook        |

For deeper conceptual background on *why* these jobs exist, point users at:

- `modaic/docs/docs/arbiters/aligning_your_arbiter.mdx`
- `modaic/docs/docs/arbiters/confidence_scoring.mdx`
