---
name: modaic-python-sdk
description: Use the Modaic Python HTTP SDK to define decision questions, run inference, manage models and examples, and start or monitor alignment and batch-decision jobs. Covers synchronous and asynchronous clients and typed Pydantic responses.
---

# Modaic Python SDK

Default to `mo-latest` for inference. Use `mo-2.0` when pinning a version. Use `mo-1.3.1` only for fine-tuning. Do not select a different base model unless the user explicitly requests it.

Install `modaic` with Python 3.11+.
Read [quickstart.py](quickstart.py) for an executable typed direct-inference example.

## Configuration

Use `MODAIC_API_KEY` or `Modaic(api_key=...)`; never embed real keys in code.
`timeout` is in seconds. Use a context manager to close the HTTP client.

Typesafe System One HTTP requests can migrate without changing state or
questions: use a Modaic key and endpoint, create a saved model, and set
`model` to its `workspace/slug`. Core answers/usage stay compatible.
For URL-joining details when retaining a Typesafe client, see `modaic-api`.
Use `decisions.create` for inference.
Use `mo-latest` by default. Use `mo-2.0` to pin the current version.

## Interactive CLI authentication

The Python package includes `modaic login --workspace acme` and
`modaic login --headless`. `--workspace` binds the refreshable CLI session to
the account's personal workspace or an organization after verifying membership;
omit it for an unbound personal session.
Match the terminal code and device before approving a request the user started.
Browser sessions authenticate CLI commands only; library clients, CI, and
deployed services use API keys. CLI credential precedence is `MODAIC_API_KEY`,
then a stored API key from `modaic login --token-stdin`, then the browser profile selected by
`MODAIC_PROFILE`. Auth subcommands require an explicit `--profile` for
non-default profiles. OS keyring is the default; `--storage file` is an
explicit POSIX plaintext fallback with private permissions, unsupported on Windows.
Never expose session files, access tokens, refresh tokens, or API keys in
output, logs, prompts, or commits. See https://docs.modaic.dev/docs/cli.

## Decisions and questions

```python
from modaic import Modaic, Noul, Choice, Score

with Modaic() as client:
    result = client.decisions.create(
        model="mo-latest",
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
`workspace/slug`, for example `acme/support-priority`. A task-specific model
has saved questions and must be called without request questions; passing
questions returns `task_specific_model_questions_forbidden`. A general model
has no saved questions and must be called with questions; omitting them
returns `questions_required` (422).
Pass `revision` explicitly on a decision to pin a version.

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
        model="mo-latest",
        questions={"urgent": Noul(instructions="Does this require urgent action?")},
    )
    result = model.decisions.create(state="The production checkout is unavailable.")
    examples = model.examples.list(page=1, page_size=20)
    jobs = model.jobs.batch_decisions.list()
```

Creation requires a base `model` and accepts an optional question schema
directly; omit questions to create a general model. Do not assemble model files.
The create response has a workspace slug string. `models.get(workspace=...,
model=...)` returns a model whose `workspace` is an entity object and whose
`configuration` carries the saved `model` and `questions` (None only when
the model has no manifest yet). Do not change one response shape to match
the other. `models.update` with questions identical to the stored ones is a
no-op that succeeds: `commit.commit_sha == commit.previous_sha` and the
checkpoint is unchanged. Compare `configuration.questions` first to skip
the call, and never catch a 500 as "already up to date".

If `configuration.checkpoint > 0`, alignment has rewritten the questions.
`models.update(..., questions=...)` with different questions raises
`ModaicAPIError` with `code == "alignment_would_be_discarded"` (409) rather
than reverting the model. Keep aligned instructions by omitting `questions`;
pass `discard_alignment=True` only when the user explicitly wants to replace
them and reset the checkpoint.

`models.get(...).commit.commit_sha` is the default branch head. Pass it as
`expected_head_sha=` to `models.update`; a stale value raises
`ModaicAPIError` with `code == "expected_head_mismatch"`. Version control
on `client.models`: `list_branches`, `create_branch(name=, source_ref=)`,
`delete_branch`, `list_commits(branch=)`, `list_tags`,
`create_tag(name=, commit_sha=)`, `delete_tag`, and
`rollback(branch=, target_commit_sha=, expected_head_sha=, message=)`. Tag
an alignment's result commit and use the tag as `revision`; restore an
overwritten version with `rollback`, not by re-committing prompt text.

Models from create/get/update bind `decisions`, `examples`, and `jobs`.
Bound calls omit the model path. Keep their originating client open.
Top-level `client.examples`, `client.batch_decisions`, and
`client.alignments` methods take `workspace/model` for collection calls.
Example item calls (`get`, `annotate`, `list_decisions`) take only the globally
unique example ID.

`model.examples`: `ingest(examples=[...])`, `list`, `get`,
`annotate`, `list_decisions`. Examples carry `state` and optional
`annotation={"ground_truth": {...}}`.
Ground truth may contain false or zero. Ingesting a labeled example alone
does not run inference. The returned annotation carries a `split` the
service assigns and preserves; it is not an input, so do not curate
train/test membership yourself. Nested `latest_decision` is optional and does not
require an `example_id`; history records have their own shape.

## Jobs

`model.jobs.batch_decisions.create` requires `idempotency_key` and exactly
one of `example_ids`, `examples`, or `scope="all"`. Explicit lists have
1–1,000 examples. The default branch is main.

`model.jobs.alignments.create` requires `branch`, `source_commit_sha`,
and `idempotency_key`. Use an actual source commit. `max_metric_calls` is
optional (modaic >= 0.52.0): when omitted the API sizes the budget to the
validation split at launch, which is the recommended default. Optional
reflection settings are `reflection_model`, `reflection_minibatch_size`,
and `seed`.

Use top-level job resources for `get(id)`, `wait(id)`, `cancel(id)`;
alignment also exposes `logs(id)`. `wait(..., progress=True)` renders tqdm.
It returns terminal failed/cancelled jobs too: check `status` and `error`.
A wait timeout does not cancel the job. Alignment progress is stage/budget
usage, not an overall completion percentage. Progress is opt-in terminal IO.

## Async and retries

Put images in JSON `state` as
`{"type": "image_url", "image_url": {"url": "https://..."}}` or use a
`data:image/...;base64,` URL. `mo-latest` accepts up to four images
(PNG, JPEG, GIF, or WebP, 20 MB each). No separate image parameter is needed.

`AsyncModaic` has the same resource names, arguments, and typed responses.
Use `async with` and await every network method. Type names and public
Python parameters use snake_case.

`decisions.create` calls `POST /systemone`. Reuse an explicit idempotency
key only for the exact same request. For `409 decision_in_progress`, the
SDK retries a keyed decision at most five times with bounded backoff.
Approved ZDR workspaces making covered direct base-model calls ignore the key:
each retry is a new, potentially billable inference, not a replay. ZDR requires
a separate signed agreement, approval, and technical activation; saved-model
calls, batch jobs, and training are not covered. `capture=False` does not make
a saved-model call ZDR. See
[Zero Data Retention](https://docs.modaic.dev/docs/api/zero-data-retention).
Other conflicts are not automatically retried. Capture may become visible
shortly after a successful response; do not make inference wait for history.

HTTP errors raise `ModaicAPIError` with `status_code`, `code`,
`request_id`, `details`, and `body`. Network/parsing errors raise
`ModaicConnectionError`; timeouts raise `ModaicTimeoutError`.

## Workspace resources and fine-tuning


Use `client.uploads` for resumable datasets, eval-only datasets,
and artifacts. Prefer `upload_file(workspace, path, name=, kind=...)` for a new local file: it computes the
whole-file SHA-256, creates the session, streams server-sized parts, confirms
receipts, completes, and waits for validation. Pass `wait_for_validation=False`
for an asynchronous return and `progress=` for byte progress. It makes a
best-effort cancellation if transfer fails before completion.
`create(workspace, kind=, sha256=, size_bytes=, name=, repos=)` creates a session;
`upload_part(workspace, id, part_number=, data=)` hashes one bytes chunk, uploads it with
the file-scoped grant, and confirms its receipt. Then call `complete` and poll
`get` or `wait` until validation is terminal. `resources` provides
`list`, `get`, `update`, and `download`; `list` accepts a `glob` filter and
`update` replaces repository links and/or the description. Resource names are
unique across all kinds, so get/download do not take a kind.

Use `client.protege.create(prompt=, repo="workspace/model", eval_only_dataset=...)`
to start optimization, then `get`, `cancel`, or `wait` by job ID.

Use `client.fine_tunes.create` with `training_file` and
`training_type="head_only" | "lora"`. Base model `mo-1.3.1` also requires
`destination`. The JSONL dataset supplies `instructions`, `type`, `criteria`,
and `answer` on each inline question; rows do not contain partition labels.
Pass `validation_file` and `eval_only_dataset` to use separate uploaded files.
When either is omitted, the API selects that partition deterministically from
the training file. Rows may also contain up to ten named PNG/JPEG/WebP assets as raw
base64 or public HTTPS URLs. Asset IDs are placeholders inside state,
instructions, and criteria strings; URL contents are snapshotted when training
starts. The `mo-1.3.1` training base is text-only and returns
`fine_tune_modality_unsupported` for referenced images instead of discarding
them. This works for head-only and LoRA. Training controls are passed in the `controls` mapping. LoRA
also accepts `lora_r` (maximum 64), `lora_alpha` (maximum 128), and
`lora_dropout`; trainable modules are not configurable. Use `list`, `get`,
`logs`, `cancel`, and `wait` for jobs. Successful training publishes a normal
model commit. Invoke `workspace/model` for main or
`workspace/model@<full-commit-sha>` to pin it.


See https://docs.modaic.dev/docs/sdks/jobs/fine-tuning for dataset requirements.

## Mo 2.0 extraction and boxes

Select `mo-2.0` explicitly. `Span` (`type: "span"`) extracts an exact
substring from a string state; `Box` (`type: "box"`) localizes an image region.
Both require nonempty string instructions and reject criteria.
Span answers contain `span: {start, end, text}` and confidence; offsets count
UTF-16 code units and end is exclusive. Box answers contain
`box: {top, bottom, left, right}` normalized to [0, 1], and confidence.
Pass inline `assets` to decisions.create, mapping an asset ID to
`{type: "image", format: "png", data_base64: "..."}`. Reference IDs in
state, instructions or supported criteria; no separate upload is required.
`mo-latest` selects `mo-2.0`.
Run a JSON inference request (including inline assets) with
`modaic --json decisions create --request request.json`.

The CLI accepts these question shapes in `modaic models create WORKSPACE SLUG --config model.json` and `modaic models update WORKSPACE/MODEL --config model.json`. Put the base model and questions in the configuration; supply image assets on inference requests.
