# Modaic Python SDK

The official, HTTP-only Python client for the public Modaic API. It includes
matching synchronous and asynchronous clients and never invokes Git or reads or
writes repository files.

## Install

```bash
pip install modaic
```

Set an API key from <https://modaic.dev/settings/keys>:

```bash
export MODAIC_API_KEY="mdc_..."
```

## Quickstart

```python
from modaic import Modaic

with Modaic() as modaic:
    result = modaic.decisions.create(
        state={
            "ticket": "I was charged twice for order 4832. Please refund one charge."
        },
        model="typesafe/jev-latest",
        questions={
            "needs_refund": {
                "type": "noul",
                "instructions": "Should this customer receive a refund?",
                "criteria": {
                    "true": "A duplicate or invalid charge should be refunded.",
                    "false": "The charge is valid or more information is required.",
                },
            },
            "priority": {
                "type": "choice",
                "instructions": "Choose the support priority.",
                "criteria": {
                    "low": "No financial or time-sensitive impact.",
                    "normal": "Routine customer issue.",
                    "high": "Financial impact or an urgent blocker.",
                },
            },
        },
        idempotency_key="ticket-4832-v1",
    )

print(result.answers["priority"])
```

The API URL defaults to `https://modaic.dev/api/v1`. Set `MODAIC_API_URL` to
override it, or pass `base_url` to the client. The client option takes precedence
over the environment variable. This applies to both `Modaic` and `AsyncModaic`.

For a local server, pass the versioned API URL explicitly:

```python
modaic = Modaic(api_key="mdc_...", base_url="http://localhost:3001/v1")
```

## Question objects and typed responses

Use `Noul`, `Choice`, and `Score` to define questions. Dictionary questions still
work, including alongside question objects. Both forms are accepted by
`decisions.create`, `models.create`, and `models.update`.

`Choice` requires at least one option; `Score` requires at least one rubric
level. Empty choice criteria sent as a dictionary are rejected by the API with
`422` and code `validation_error`.

Define a Pydantic response model by extending `DecisionResponse` with answer
fields matching your question names:

```python
from modaic import DecisionResponse, Modaic, Noul, NoulAnswer


class BillingResponse(DecisionResponse):
    billing: NoulAnswer


with Modaic() as modaic:
    result = modaic.decisions.create(
        model="typesafe/jev-latest",
        state="I was charged twice.",
        questions={"billing": Noul(instructions="Is this about billing?")},
        response_model=BillingResponse,
    )
    assert result.billing == result.nouls["billing"]
    print(result.billing.noul)
    print(result.request_id)
```

Use `ChoiceAnswer` and `ScoreAnswer` for choice and score fields. `answers`
retains every answer, with `nouls`, `choices`, and `scores` providing typed
views. Usage and capture metadata remain available. Required answer fields are
validated; missing answers or mismatched types raise `ModaicConnectionError`.
Use Pydantic `Field(alias="question-name")` for names that are not Python
identifiers. Keep response metadata names, such as `model` and `usage`, reserved.

`response_model` also works with `AsyncModaic` and `model.decisions.create`.
It controls local response parsing only and is not sent to the API. The
`request_id` comes from the HTTP response header and is excluded from
`model_dump()` and `model_dump_json()`.

## Async

The async client has the same resources and methods:

```python
import asyncio
from modaic import AsyncModaic


async def main() -> None:
    async with AsyncModaic() as modaic:
        models = await modaic.models.list()
        print([model.name for model in models.models])


asyncio.run(main())
```

## Model-bound resources

Models returned by `models.create`, `models.get`, and `models.update` can run
decisions directly:

```python
with Modaic() as modaic:
    model = modaic.models.get(workspace="acme", model="support-priority")
    result = model.decisions.create(
        state={"ticket": "Please refund my duplicate charge."},
    )
    examples = model.examples.list(page_size=10)
    batches = model.jobs.batch_decisions.list()
    alignments = model.jobs.alignments.list()
```

With `AsyncModaic`, await both calls. The bound method accepts every decision
option except `model` and uses the same client; keep that client open while
running decisions. Pass `revision` to pin a version. `model_dump()` and
`model_dump_json()` contain only response data. The top-level
`modaic.decisions.create` remains available.

`model.examples` exposes `ingest`, `list`, `get`, `annotate`, and `list_decisions`
without a model ID argument. `model.jobs.alignments` and
`model.jobs.batch_decisions` expose model-bound `create` and `list`. Use the
top-level job resources to retrieve, wait for, or cancel a job by its ID.

## Job progress

Pass `progress=True` to either job resource's `wait()` method for a tqdm display:

```python
finished = modaic.batch_decisions.wait(job.id, progress=True)
finished = modaic.alignments.wait(alignment.id, progress=True)
```

With `AsyncModaic`, use `await` with the same option. Progress is off by default.
Batch jobs show processed examples and failures; alignment shows its stage and
metric-call budget usage, not an overall completion percentage. Updates use the
existing polling interval. Timing out stops waiting without cancelling the job.

## Resources

| Resource | Methods |
| --- | --- |
| `decisions` | `create` |
| `models` | `list`, `create`, `get`, `update`, `delete` |
| `examples` | `ingest`, `list`, `get`, `annotate`, `list_decisions` |
| `batch_decisions` | `create`, `list`, `get`, `cancel`, `wait` |
| `alignments` | `create`, `list`, `get`, `logs`, `cancel`, `wait` |

Responses are Pydantic models. Python attributes use `snake_case` even when the
wire format uses `camelCase`.

## Errors

Non-2xx responses raise `ModaicAPIError`, which exposes `status_code`, `code`,
`request_id`, `details`, and the decoded response `body`. Network failures raise
`ModaicConnectionError`; request and polling deadlines raise
`ModaicTimeoutError`.

See the complete API documentation at <https://docs.modaic.dev>.
