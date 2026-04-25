---
name: modaic-api
description: Reference for the Modaic REST API — every public endpoint with a working curl + Python example. Use this skill when the user is calling Modaic over HTTP directly (not via the Python SDK), debugging an API request, or wiring Modaic into a non-Python service.
---

# Modaic REST API

Base URL: `https://api.modaic.dev`

All endpoints require a bearer token in the `Authorization` header. Get a
token at https://modaic.dev/settings/tokens and set it as `MODAIC_TOKEN`.

```bash
export MODAIC_TOKEN=mk_...
```

The Python examples below use the `httpx` client and assume the env var
is set. They are written so you can paste them straight into a script.

> The arbiter run endpoints require the **provider's API key** (e.g.
> `TOGETHER_API_KEY`) to be set as an Environment Variable on Modaic Hub
> at https://modaic.dev/settings/env-vars. Without it, `POST
> /api/v1/arbiters/predictions` will fail server-side regardless of how
> you're calling it.

---

## Arbiters

### Initialize an arbiter

`POST /api/v1/arbiters` — creates a new arbiter repository from a
JSON-schema-style field spec. Use this when you don't have a Python
environment available; otherwise prefer the Signature + `modaic.Predict`
flow from the SDK.

```bash
curl -X POST https://api.modaic.dev/api/v1/arbiters \
  -H "Authorization: Bearer $MODAIC_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "repo": "your-org/support-triage",
    "inputs": [
      {"name": "ticket", "type": "string", "description": "Support ticket text"}
    ],
    "outputs": [
      {
        "name": "queue",
        "type": "string",
        "allowed_values": ["billing", "technical", "account"],
        "description": "Which queue the ticket belongs to"
      }
    ],
    "instructions": "Route the ticket to the correct queue.",
    "model": "qwen3-vl-32b-instruct"
  }'
```

```python
import os, httpx

httpx.post(
    "https://api.modaic.dev/api/v1/arbiters",
    headers={"Authorization": f"Bearer {os.environ['MODAIC_TOKEN']}"},
    json={
        "repo": "your-org/support-triage",
        "inputs": [{"name": "ticket", "type": "string"}],
        "outputs": [
            {
                "name": "queue",
                "type": "string",
                "allowed_values": ["billing", "technical", "account"],
            }
        ],
        "instructions": "Route the ticket to the correct queue.",
        "model": "qwen3-vl-32b-instruct",
    },
).raise_for_status()
```

### Get arbiter metadata

`GET /api/v1/arbiters/?repo=...` — returns the arbiter's metadata.

```bash
curl "https://api.modaic.dev/api/v1/arbiters/?repo=your-org/support-triage" \
  -H "Authorization: Bearer $MODAIC_TOKEN"
```

### Get arbiter schema

`GET /api/v1/arbiters/schema?repo=...` — returns the arbiter's
input/output schema.

```bash
curl "https://api.modaic.dev/api/v1/arbiters/schema?repo=your-org/support-triage" \
  -H "Authorization: Bearer $MODAIC_TOKEN"
```

### Update arbiter metadata

`PATCH /api/v1/arbiters/metadata` — patch arbiter metadata (e.g.
confidence threshold, slack notifications).

```bash
curl -X PATCH https://api.modaic.dev/api/v1/arbiters/metadata \
  -H "Authorization: Bearer $MODAIC_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "repo": "your-org/support-triage",
    "metadata": {"confidence_threshold": 0.8}
  }'
```

### Run a prediction

`POST /api/v1/arbiters/predictions` — run one or more arbiters on a
single input. This is what the Python `Arbiter.__call__` calls under the
hood, and what Braintrust scorers + LangSmith integrations end up
hitting.

```bash
curl -X POST https://api.modaic.dev/api/v1/arbiters/predictions \
  -H "Authorization: Bearer $MODAIC_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {"ticket": "My payment failed twice in a row."},
    "arbiters": [
      {"arbiter_repo": "your-org/support-triage", "arbiter_revision": "main"}
    ]
  }'
```

```python
import os, httpx

r = httpx.post(
    "https://api.modaic.dev/api/v1/arbiters/predictions",
    headers={"Authorization": f"Bearer {os.environ['MODAIC_TOKEN']}"},
    json={
        "input": {"ticket": "My payment failed twice in a row."},
        "arbiters": [{"arbiter_repo": "your-org/support-triage"}],
    },
    timeout=60,
)
r.raise_for_status()
print(r.json()["predictions"][0]["output"])
```

To attach ground-truth at prediction time (e.g. when running labeled data
through an arbiter for evaluation), include `ground_truth` and
`ground_reasoning` per arbiter:

```json
{
  "input": {"ticket": "..."},
  "arbiters": [{
    "arbiter_repo": "your-org/support-triage",
    "ground_truth": "billing",
    "ground_reasoning": "Refund request"
  }]
}
```

### Dispatch predictions across arbiters

`POST /api/v1/arbiters/predictions/dispatch` — fan one input out across
multiple arbiters/groups. Use this when one record needs to be routed
through several arbiters at once.

```bash
curl -X POST https://api.modaic.dev/api/v1/arbiters/predictions/dispatch \
  -H "Authorization: Bearer $MODAIC_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {"ticket": "My payment failed twice in a row."},
    "alt_id": "ticket-12345",
    "arbiters_map": {
      "triage": [{"arbiter_repo": "your-org/support-triage"}],
      "sentiment": [{"arbiter_repo": "your-org/sentiment"}]
    }
  }'
```

### Chat completions (OpenAI-compatible)

`POST /api/v1/arbiters/chat/completions` — OpenAI-compatible chat
completions surface. The `model` field is your Arbiter repo path. This is
the endpoint LangSmith and other OpenAI-compatible tools point at.

```bash
curl -X POST https://api.modaic.dev/api/v1/arbiters/chat/completions \
  -H "Authorization: Bearer $MODAIC_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-org/support-triage",
    "messages": [
      {"role": "user", "content": "<run><ticket>My payment failed</ticket></run>"}
    ]
  }'
```

The endpoint parses the user message for XML tags whose names match the
arbiter's signature input fields, then runs the arbiter. See the
`langsmith.md` reference in `modaic-sdk` for the templating contract.

---

## Examples (predictions store)

Each arbiter run produces an **example** stored in Modaic's prediction
store. These endpoints manage that store.

### Create examples

`POST /api/v1/examples` — bulk-insert examples (NDJSON body).

```bash
curl -X POST https://api.modaic.dev/api/v1/examples \
  -H "Authorization: Bearer $MODAIC_TOKEN" \
  -H "Content-Type: text/plain" \
  --data-binary $'{"input":{"ticket":"refund please"},"ground_truth":"billing"}\n{"input":{"ticket":"login broken"},"ground_truth":"technical"}'
```

### List examples

`GET /api/v1/examples` — paginated, filterable list.

```bash
curl "https://api.modaic.dev/api/v1/examples?program=your-org/support-triage&page=1&page_size=50&max_confidence=0.6" \
  -H "Authorization: Bearer $MODAIC_TOKEN"
```

Useful filters: `user`, `program`, `version`, `commit_hash`, `search`,
`sort_by`, `sort_order`, `max_confidence` (great for "show me the
arbiter's least-certain predictions").

### Get a single example

`GET /api/v1/examples/{example_id}` — fetch one prediction.

```bash
curl "https://api.modaic.dev/api/v1/examples/ex_01HZ9K2F8V" \
  -H "Authorization: Bearer $MODAIC_TOKEN"
```

### Annotate an example

`PATCH /api/v1/examples/{example_id}/annotation` — attach
`ground_truth` / `ground_reasoning` to an existing prediction. This is
the supervision signal Modaic uses to align the arbiter and re-train the
confidence probe.

```bash
curl -X PATCH https://api.modaic.dev/api/v1/examples/ex_01HZ9K2F8V/annotation \
  -H "Authorization: Bearer $MODAIC_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "annotations": [{
      "arbiter_repo": "your-org/support-triage",
      "ground_truth": "escalate",
      "ground_reasoning": "Repeated failures suggest a payment-provider outage."
    }]
  }'
```

### Delete examples

`DELETE /api/v1/examples` — bulk-delete by IDs scoped to one arbiter.

```bash
curl -X DELETE https://api.modaic.dev/api/v1/examples \
  -H "Authorization: Bearer $MODAIC_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "arbiter_repo": "your-org/support-triage",
    "example_ids": ["ex_01HZ9K2F8V", "ex_01HZ9K2F8W"]
  }'
```

### Check for uncalibrated examples

`GET /api/v1/examples/has-uncalibrated` — does the program still have
examples awaiting confidence scoring?

```bash
curl "https://api.modaic.dev/api/v1/examples/has-uncalibrated?program=your-org/support-triage" \
  -H "Authorization: Bearer $MODAIC_TOKEN"
```

---

## Jobs

Long-running background work. Both confidence scoring and prompt
optimization (GEPA) follow the same start/poll/cancel pattern.

### Confidence scoring

`POST /api/v1/jobs/confidence-scores` — kick off scoring for any unscored
predictions in an arbiter's repo.

```bash
curl -X POST https://api.modaic.dev/api/v1/jobs/confidence-scores \
  -H "Authorization: Bearer $MODAIC_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"arbiter_repo": "your-org/support-triage"}'
```

`GET /api/v1/jobs/confidence-scores/{job_id}` — poll status.

```bash
curl "https://api.modaic.dev/api/v1/jobs/confidence-scores/job_abc123" \
  -H "Authorization: Bearer $MODAIC_TOKEN"
```

`DELETE /api/v1/jobs/confidence-scores/{job_id}` — cancel.

```bash
curl -X DELETE "https://api.modaic.dev/api/v1/jobs/confidence-scores/job_abc123" \
  -H "Authorization: Bearer $MODAIC_TOKEN"
```

### Prompt optimization (GEPA)

`POST /api/v1/jobs/gepa` — kick off prompt optimization on labeled
examples and push the result back to the repo.

```bash
curl -X POST https://api.modaic.dev/api/v1/jobs/gepa \
  -H "Authorization: Bearer $MODAIC_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "arbiter_repo": "your-org/support-triage",
    "arbiter_revision": "main",
    "push_branch": "main",
    "push_tag": "optimized-v1",
    "gepa_kwargs": {
      "auto": "light",
      "candidate_selection_strategy": "pareto",
      "component_selector": "round_robin"
    }
  }'
```

`GET /api/v1/jobs/gepa/{job_id}` — poll status. Response includes
`status` (`in_progress` | `completed` | `failed`), an optional `message`,
and a `results` payload when finished.

```bash
curl "https://api.modaic.dev/api/v1/jobs/gepa/job_xyz" \
  -H "Authorization: Bearer $MODAIC_TOKEN"
```

`DELETE /api/v1/jobs/gepa/{job_id}` — cancel.

```bash
curl -X DELETE "https://api.modaic.dev/api/v1/jobs/gepa/job_xyz" \
  -H "Authorization: Bearer $MODAIC_TOKEN"
```
