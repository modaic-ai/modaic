---
name: modaic-sdk
description: Build, push, and run Modaic Arbiters (LLM judges with calibrated confidence scores) using the Modaic Python SDK. Use this skill whenever the user is creating an Arbiter, defining a `dspy.Signature`, calling `modaic.Predict`, pushing a judge to Modaic Hub, or running a deployed Arbiter.
---

# Modaic SDK

## What is Modaic?

Modaic helps you build LLM judges that emit **a decision and a calibrated
confidence score**. The confidence is computed from the model's *hidden
states* (mechanistic interpretability), not from verbalized confidence or
token logprobs, so it is well-calibrated and continuously improves as more
labeled data flows in.

These judges are called **Arbiters**. An Arbiter is meant for any semantic
transformation task with a **discrete output space**:

- single-class / binary classification
- multi-class classification
- single-label / multi-label extraction
- subjective rating on a finite scale
- triage and routing

If the output space is a fixed enumeration, it can be an Arbiter.

## Defining an Arbiter

The preferred way to define an Arbiter is the `dspy.Signature` + `modaic.Predict` + `.as_arbiter()` pattern.

```python
import dspy
import modaic
from typing import Literal


class TicketTriage(dspy.Signature):
    """Classify the support ticket into the right queue.

    Use `billing` for payment, refund, or invoice issues.
    Use `technical` for product bugs and integration errors.
    Use `account` for login, profile, or permission questions.
    """

    ticket: str = dspy.InputField(desc="The user-submitted support ticket")
    queue: Literal["billing", "technical", "account"] = dspy.OutputField(
        desc="Which queue should own this ticket"
    )


arbiter = modaic.Predict(
    TicketTriage,
    lm=dspy.LM(model="together_ai/openai/gpt-oss-120b"),
).as_arbiter()

arbiter.push_to_hub("your-org/support-triage", private=True)
```

### Important rules for signatures

- **Output fields MUST use `Literal`** (or another enumerable type — see
  `signatures.md`). Modaic needs to know the *finite* output space. A bare
  `str`/`int` output field gives Modaic no enum to calibrate against.
- **Do NOT add a `reasoning` output field yourself.** `as_arbiter()` injects
  one automatically. Adding your own collides with it.
- **The signature's docstring is the prompt.** It defines the task and the
  initial system instructions sent to the LLM. Write it like a rubric, not
  like a Python docstring.
- For very long instructions, don't paste them into the docstring. Use
  `.with_instructions(...)` instead:

  ```python
  long_rubric = open("rubric.txt").read()
  TicketTriage = TicketTriage.with_instructions(long_rubric)
  ```

### `modaic.Predict` is the standard judge

`modaic.Predict` is essentially a single-shot LLM call with built-in output
parsing against the signature. It is the standard primitive for an LLM judge
in Modaic.

### Always call `.as_arbiter()` before `push_to_hub`

If you `push_to_hub` a `modaic.Predict` without calling `.as_arbiter()`,
the repo will not be recognized as an arbiter on Modaic Hub — it will not
get an annotations page, confidence scoring, or the `Arbiter(...)` loader
behavior.

```python
# WRONG — will not be recognized as an arbiter
modaic.Predict(Sig).push_to_hub("org/judge")

# RIGHT
modaic.Predict(Sig).as_arbiter().push_to_hub("org/judge")
```

### Prefer Signature + `modaic.Predict` over `ModaicClient.create_arbiter`

`ModaicClient.create_arbiter(...)` exists, but you should generally **prefer
the Signature + `modaic.Predict(...).as_arbiter()` pattern**. The signature
gives you typed inputs/outputs, native Literal handling, an LM bound to the
judge, and proper version control via `push_to_hub`. Only reach for
`create_arbiter` when you have no Python environment that can run DSPy and
you're scripting against the API.

## Running an Arbiter

Once an Arbiter is on Modaic Hub, the preferred entry point is to construct an `Arbiter` directly — it discovers the shared `ModaicClient` automatically, so you don't need to call `client.get_arbiter(...)`.

There are four common ways to run a deployed Arbiter. Pick by **how many inputs** and **how many arbiters** you have, plus whether you need confidence inline.

| Method | When to use |
| --- | --- |
| `Arbiter.predict` (single call) | One input, one arbiter — ad-hoc calls and request-path inference |
| `Arbiter.predict_all` | Many inputs, one arbiter — offline scoring or bulk jobs |
| `ModaicClient.predict_all` | Many inputs, multiple arbiters — ensembling or judge comparisons |
| `compute_confidence=True` | You need a calibrated confidence score in the response path |

### Single Call

Use `Arbiter.predict` (or call the arbiter directly) for one input against one arbiter. This is the simplest entry point and the run is logged on Modaic Hub. Prefer this for online/request-path inference and ad-hoc evaluation.

```python
from modaic import Arbiter

arbiter = Arbiter("your-org/support-triage")

# kwargs MUST match the input fields of the signature
prediction = arbiter.predict(ticket="My payment failed twice in a row.")

print(prediction.output.queue)   # "billing"
print(prediction.reasoning)      # auto-generated reasoning
print(prediction.confidence)     # calibrated confidence (lazy)
```

`arbiter(...)` and `arbiter.predict(...)` are equivalent.

> Do **not** invoke `modaic.Predict.__call__` (the local DSPy module) when you want a tracked run — that path skips the Modaic backend entirely, so no example is logged and no confidence is computed. Always go through `Arbiter(...)` for tracked runs.

### Multiple Examples

Use `Arbiter.predict_all` to run one arbiter against many examples in a single batch job. Prefer this for offline scoring, backfills, or any bulk workload — it dispatches one server-side batch instead of N separate requests, which is dramatically cheaper and faster than looping over `predict`.

By default the call blocks until results are ready. Pass `wait_for=None` to get a `BatchJob` handle you can poll with `job.status()` or `job.wait(...)`.

```python
from modaic import Arbiter

arbiter = Arbiter("your-org/support-triage")

results = arbiter.predict_all(
    examples=[
        {"input": {"ticket": "My payment failed twice in a row."}},
        {"input": {"ticket": "How do I change my plan?"}},
        {"input": {"ticket": "The app crashes when I open settings."}},
    ],
)

for row in results:
    pred = row.predictions[0]
    print(row.example_id, pred.output)
```

### Multiple Arbiters

Use `ModaicClient.predict_all` when you need to **run multiple arbiters against the same set of examples** — for example, ensembling judges, A/B-comparing two prompts, or scoring with a triage + sentiment pair on each ticket. All arbiters are dispatched in a single batch job and evaluated concurrently on the server, so this is the right primitive whenever the unit of work spans more than one judge.

For a single arbiter, prefer `Arbiter.predict` / `Arbiter.predict_all` — they're simpler and don't require building a `ModaicClient` explicitly.

```python
from modaic import Arbiter, ModaicClient

client = ModaicClient()

triage = Arbiter("your-org/support-triage")
sentiment = Arbiter("your-org/sentiment")

results = client.predict_all(
    examples=[
        {"input": {"ticket": "My payment failed twice in a row."}},
        {"input": {"ticket": "Thanks, your team is amazing!"}},
    ],
    arbiters=[triage, sentiment],
)

for row in results:
    for pred in row.predictions:
        print(row.example_id, pred.arbiter_repo, pred.output)
```

### Re-predicting on existing examples

Pass `example_ids` instead of `examples` to re-run predictions on examples you've already ingested. The server fails fast (400) if any id doesn't exist for a requested arbiter; the new predictions land as a fresh version on the existing example rows.

```python
results = arbiter.predict_all(
    example_ids=["ex_01HZ9K2F8V", "ex_01HZ9K2F8W"],
)
```

### Batch Confidence Scoring

Pass `compute_confidence=True` to `predict_all` (on either `Arbiter` or `ModaicClient`) to kick off batch confidence scoring after every prediction in the batch is persisted. Scoring is filtered to *this job's* predictions — it won't touch unrelated NULL-confidence rows in the same repo.

```python
results = arbiter.predict_all(
    examples=[
        {"input": {"ticket": "My payment failed twice."}},
        {"input": {"ticket": "Thanks!"}},
    ],
    compute_confidence=True,
    wait_for="scores",   # block until scoring finishes; defaults to "predictions"
)

for row in results:
    for pred in row.predictions:
        print(pred.output, pred.confidence)
```

`wait_for` controls how long `predict_all` blocks: `"predictions"` (default) returns once predictions are persisted (confidence may still populate later); `"scores"` blocks until scoring finishes; `None` returns a `BatchJob` handle right away.

> **Billing:** batch confidence scoring is cheaper than online (single-prediction) confidence scoring. Use it whenever you don't need a confidence score on the response path.

### Online Confidence Scoring

Pass `compute_confidence=True` to `Arbiter.predict` to kick off confidence scoring as soon as the prediction completes. Reading `prediction.confidence` then blocks only on whatever work is left, instead of starting from scratch on first access.

Prefer this **only** when you need confidence on the response path (e.g. routing low-confidence predictions to a human reviewer in real time). For offline scoring of already-stored predictions, use the batch confidence APIs.

> **Billing:** online confidence scoring is billed per call and is **more expensive** than batch confidence scoring, which is priced separately at a lower per-prediction rate. If you don't need confidence inline, run predictions first and score them in batch later.

```python
from modaic import Arbiter

arbiter = Arbiter("your-org/support-triage")

prediction = arbiter.predict(
    ticket="My payment failed twice in a row.",
    compute_confidence=True,
)

print(prediction.output.queue)   # "billing"
print(prediction.confidence)     # 0.87
```

### Required: provider API key on Modaic Hub

For a run to actually execute on the backend, **the user must add the API
key for the configured model provider as an Environment Variable on Modaic
Hub** (https://modaic.dev/settings/env-vars). For example, an Arbiter using
`together_ai/...` needs `TOGETHER_API_KEY` set on Modaic Hub. Without it,
runs will fail server-side.

## `modaic` vs `modaic_client`

`modaic_client` is re-exported from `modaic`. **By default, import from
`modaic`** — even for client-only things like `Arbiter`, `ModaicClient`,
`configure`, and `track`:

```python
from modaic import Arbiter, ModaicClient   # preferred
```

Only depend on `modaic_client` directly when one of these is true:

1. Your script does not use anything from the full SDK and you specifically
   want to avoid the litellm/dspy import cost (e.g. cold-start-sensitive
   serverless functions, Braintrust scorers).
2. You're in a restrictive environment (sandbox, locked-down container)
   where the full `modaic` package can't be installed.

In those cases:

```python
from modaic_client import Arbiter, ModaicClient
```

The public API is identical between the two.

## Reference files

- `signatures.md` — supported input/output types for `dspy.Signature`
- `repositories.md` — Modaic Hub repos, branches, tags, `push_to_hub`,
  `from_precompiled`
- `models.md` — supported models and which to pick
- `arbiter.py` — runnable example: define + push an Arbiter
- `run_arbiter.py` — runnable example: load + call an Arbiter
- `braintrust.md` — using Modaic Arbiters as Braintrust scorers
- `langsmith.md` — using Modaic Arbiters in LangSmith evals
