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

Once an Arbiter is on Modaic Hub, load and call it via the `Arbiter` class.

```python
from modaic import Arbiter

arbiter = Arbiter("your-org/support-triage")

# kwargs MUST match the input fields of the signature
result = arbiter(ticket="My payment failed twice in a row.")

# result.output has fields matching the signature's output fields
print(result.output.queue)        # "billing"
print(result.reasoning)           # auto-generated reasoning
print(result.confidence)          # calibrated confidence (lazy)
```

`arbiter(...)` and `arbiter.predict(...)` are equivalent.

### Use the `Arbiter` class — not `modaic.Predict.__call__` — when you want runs tracked

If you call `modaic.Predict.__call__` (i.e. invoke the local DSPy module),
the run is **not tracked** on Modaic Hub: no example is logged, no
confidence is computed. To get tracking, you must run the judge through the
Modaic backend, which means using the `Arbiter` class:

- `Arbiter("repo")(...)` — preferred
- `Arbiter("repo").predict(...)` — equivalent

Use `ModaicClient.predict_all(...)` only when you want to **fan out one
input across multiple Arbiters in parallel** in a single server-side
request. For single-Arbiter calls, `Arbiter(...)` is simpler.

```python
from modaic import Arbiter, ModaicClient

triage = Arbiter("your-org/support-triage")
sentiment = Arbiter("your-org/sentiment")

client = ModaicClient()
response = client.predict_all(
    input={"ticket": "My payment failed twice in a row."},
    arbiters=[triage, sentiment],
)
for pred in response.predictions:
    print(pred.arbiter_repo, pred.output)
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
