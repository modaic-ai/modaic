# Modaic Arbiters in Braintrust

You can use a Modaic Arbiter as a **code scorer** in
[Braintrust](https://www.braintrust.dev/). The Arbiter runs on Modaic's
backend and returns a calibrated score, which Braintrust attaches to the
trace.

## 1. Set MODAIC_TOKEN in Braintrust

In Braintrust → **Settings** → **Env variables**, add `MODAIC_TOKEN` with
a token from https://modaic.dev/settings/tokens. Without this, the scorer
cannot authenticate against Modaic Hub.

## 2. Write the scorer

Use `modaic_client` (not the full `modaic` SDK) here. Braintrust scorers
are deployed in a sandboxed environment where cold-start latency matters,
and `modaic_client` ships without DSPy/litellm.

```python
# scorer.py
import braintrust
from pydantic import BaseModel
from modaic_client import Arbiter

project = braintrust.projects.create(name="My Project")


class CorrectnessParams(BaseModel):
    input: dict
    output: dict


def correctness_scorer(input: dict, output: dict):
    arbiter = Arbiter("modaic/correctness")
    result = arbiter.predict(input=input, output=output)
    # result.output is a Pydantic-style object whose fields match the
    # arbiter's signature outputs. Return whichever score field your
    # arbiter produces.
    return result.output.score
    # Or return both score and reasoning:
    # return {"score": result.output.score, "reasoning": result.reasoning}


project.scorers.create(
    name="Correctness Scorer",
    slug="correctness-scorer",
    description="Check if the output is correct",
    parameters=CorrectnessParams,
    handler=correctness_scorer,
    metadata={"__pass_threshold": 0.5},
)
```

> The Arbiter must accept the schema you pass in. In the example above,
> `modaic/correctness` is expected to declare `input: dict` and
> `output: dict` as input fields on its signature.

## 3. Declare the dependency

```txt
# requirements.txt
modaic-client
```

## 4. Push the scorer

```bash
# uv
uv run braintrust push scorer.py --requirements requirements.txt

# pip / global
braintrust push scorer.py --requirements requirements.txt
```

## Why `modaic_client` and not `modaic`?

Braintrust scorers run in restricted sandboxes that do not allow dspy to be installed. Also since we don't need any modaic stuff here its nice to use modaic_client to avoid the cold start. 
