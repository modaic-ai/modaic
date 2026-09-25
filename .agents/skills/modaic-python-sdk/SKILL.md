---
name: modaic-python-sdk
description: Use the Modaic Python HTTP SDK to define decision questions, run inference, manage models and examples, and start or monitor alignment and batch-decision jobs. Covers synchronous and asynchronous clients and typed Pydantic responses.
---

# Modaic Python SDK

Install `modaic>=0.48.0` with Python 3.11+. The SDK is HTTP-only: it runs
inference on Modaic and does not invoke Git or read/write model files.
Read [quickstart.py](quickstart.py) for an executable typed direct-inference example.

## Configuration

Use `MODAIC_API_KEY` or `Modaic(api_key=...)`; never embed real keys in code.
The API base defaults to `https://modaic.dev/api/v1`. Precedence:
explicit `base_url`, then `MODAIC_API_URL`, then the default.
For the local development API use `http://localhost:3001/v1`.
`timeout` is in seconds. Use a context manager to close the HTTP client.

Typesafe System One HTTP requests can migrate without changing state or
questions: use a Modaic key and endpoint, create a saved model, and set
`model` to its `workspace/slug`. Core answers/usage stay compatible.
For URL-joining details when retaining a Typesafe client, see `modaic-api`.
When using this SDK, call `decisions.create` with the versioned base above.

## Decisions and questions

```python
from modaic import Modaic, Noul, Choice, Score

with Modaic() as client:
    result = client.decisions.create(
        model="typesafe/jev-latest",
        state={"description": "Waterproof hiking boots, size 42"},
        questions={
            "outdoor": Noul(instructions="Is this product intended for outdoor use?"),
            "department": Choice(criteria={
                "apparel": "Clothes and footwear",
                "electronics": "Electronic devices",
                "other": "Other products",
            }),
            "specificity": Score(criteria=[
                "No identifiable product", "Product category only",
                "Specific product with useful details",
            ]),
        },
        idempotency_key="catalog-item-42-v1",
    )
    print(result.choices["department"].choice)
```

Question objects and dictionaries can be mixed. Accept them in
`decisions.create`, `models.create`, and `models.update`. `Choice`
requires at least one option with nonempty keys; `Score` requires at least
one ordered level. Score values can be fractional and use zero-based levels.
Instructions are optional JSON, including null. `state` accepts JSON values,
including false, zero, lists, and null; do not silently replace falsy values.

Direct inference needs no saved model. Saved model paths use
`workspace/slug`, for example `acme/support-priority`. Their saved questions
are used automatically; supplied questions override matching IDs and add new
ones. Pass `revision` explicitly on a decision to pin a version.

## Typed responses

Subclass `DecisionResponse` (a Pydantic model), with fields typed
`NoulAnswer`, `ChoiceAnswer`, or `ScoreAnswer`, and pass it as
`response_model=YourResponse`. This validates the response locally; it does
not change the questions sent to the API. Use `Field(alias="question-id")`
for non-identifier question names. Keep response metadata names reserved.

`response_model` accepts any Pydantic model. A plain `BaseModel` declaring
its own shape works, including a nested `answers` model; Pydantic drops the
fields it does not declare, and none of the `DecisionResponse` extras
(`request_id`, the typed views) are added. Prefer `DecisionResponse` unless
the caller wants to own the whole schema. `BaseModel` itself is rejected.

Responses retain `answers`, typed `nouls`/`choices`/`scores` maps,
`usage`, and optional capture metadata. `request_id` comes from the HTTP
header and is excluded from `model_dump()`. Missing or mistyped required
answers raise `ModaicConnectionError`.

## Models and bound resources

```python
with Modaic() as client:
    model = client.models.create(
        workspace="acme", slug="support-priority",
        model="typesafe/jev-latest",
        questions={"urgent": Noul(instructions="Does this require urgent action?")},
    )
    result = model.decisions.create(state="The production checkout is unavailable.")
    examples = model.examples.list(page=1, page_size=20)
    jobs = model.jobs.batch_decisions.list()
```

Creation accepts the question schema directly; do not assemble model files.
The create response has a workspace slug string. `models.get(workspace=...,
model=...)` returns a model whose `workspace` is an entity object. Do not
change one response shape to match the other.

Models from create/get/update bind `decisions`, `examples`, and `jobs`.
Bound calls omit the model ID/path. Keep their originating client open.
Top-level `client.examples`, `client.batch_decisions`, and
`client.alignments` methods take the model UUID for model-scoped calls.

`model.examples`: `ingest(examples=[...])`, `list`, `get`,
`annotate`, `list_decisions`. Examples carry `state` and optional
`annotation={"ground_truth": {...}, "ground_reasoning": "..."}`.
Ground truth may contain false or zero. Ingesting a labeled example alone
does not run inference. Nested `latest_decision` is optional and does not
require an `example_id`; history records have their own shape.

## Jobs

`model.jobs.batch_decisions.create` requires `idempotency_key` and exactly
one of `example_ids`, `examples`, or `scope="all"`. Explicit lists have
1–1,000 examples. The default branch is main.

`model.jobs.alignments.create` requires `branch`, `source_commit_sha`,
`max_metric_calls`, and `idempotency_key`. Use an actual source commit.
Optional reflection settings are `reflection_model`,
`reflection_minibatch_size`, and `seed`.

Use top-level job resources for `get(id)`, `wait(id)`, `cancel(id)`;
alignment also exposes `logs(id)`. `wait(..., progress=True)` renders tqdm.
It returns terminal failed/cancelled jobs too: check `status` and `error`.
A wait timeout does not cancel the job. Alignment progress is stage/budget
usage, not an overall completion percentage. Progress is opt-in terminal IO.

## Async and retries

`AsyncModaic` has the same resource names, arguments, and typed responses.
Use `async with` and await every network method. Type names and public
Python parameters use snake_case.

`decisions.create` calls `POST /systemone`. Reuse an explicit idempotency
key only for the exact same request. For `409 decision_in_progress`, the
SDK retries a keyed decision at most five times with bounded backoff.
Other conflicts are not automatically retried. Capture may become visible
shortly after a successful response; do not make inference wait for history.

HTTP errors raise `ModaicAPIError` with `status_code`, `code`,
`request_id`, `details`, and `body`. Network/parsing errors raise
`ModaicConnectionError`; timeouts raise `ModaicTimeoutError`.
