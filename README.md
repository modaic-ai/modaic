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

## CLI

The Python package installs the `modaic` CLI:

```bash
modaic login
modaic auth status
modaic logout
```

Use `modaic login --headless` to sign in from another device. Match the code
and device before confirming a request you started. Session secrets use the OS
keyring by default; `--storage file` explicitly opts into private plaintext
storage on POSIX. Sessions refresh automatically.

CLI sessions are for interactive commands. Programmatic `Modaic`/`AsyncModaic`
clients and CI use API keys. `MODAIC_API_KEY` takes precedence over stored
credentials. See the [CLI reference](https://docs.modaic.dev/docs/cli).

## Quickstart

```python
from modaic import Modaic

with Modaic() as modaic:
    result = modaic.decisions.create(
        state={
            "ticket": "I was charged twice for order 4832. Please refund one charge."
        },
        model="mo-latest",
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
        model="mo-latest",
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

`response_model` accepts any Pydantic model, not only `DecisionResponse`
subclasses. Extending `DecisionResponse` is the shorter path, since it
supplies `model`, `answers`, `usage`, the typed views, and `request_id`, and
lifts your answer fields to the top level. To control the whole schema
instead, declare a plain `BaseModel` and spell out the parts you want:

```python
from pydantic import BaseModel

from modaic import NoulAnswer


class BillingAnswers(BaseModel):
    billing: NoulAnswer


class BillingEnvelope(BaseModel):
    answers: BillingAnswers
```

Pydantic drops response fields the model does not declare, so this validates
only `answers.billing` and reads as `result.answers.billing.noul`. Nothing is
grafted on: `request_id` and the typed views exist only on `DecisionResponse`
subclasses. `BaseModel` itself is rejected, because Pydantic cannot validate
into it.

`response_model` also works with `AsyncModaic` and `model.decisions.create`.
It controls local response parsing only and is not sent to the API. The
`request_id` comes from the HTTP response header and is excluded from
`model_dump()` and `model_dump_json()`.

## Workspace uploads and fine-tuning

Large training datasets and artifacts use resumable upload sessions. The
high-level helper streams the file twice: once to compute its whole-file
SHA-256 and once to upload server-sized parts. It never materializes the whole
file in memory, and it waits for validation by default:

```python
with Modaic() as modaic:
    upload = modaic.uploads.upload_file(
        "acme",
        "training.jsonl",
        kind="dataset",
        name="support-hard-labels",
        repos=["support-priority"],
        progress=lambda uploaded, total: print(f"{uploaded}/{total}"),
    )
```

Set `wait_for_validation=False` to return as soon as completion starts, or use
the lower-level `create`, `prepare_part`, `upload_part`, `confirm_part`,
`complete`, and `wait` methods to resume an existing session. Resources are
available through `resources.list`, `get`, `update`, and
`download`. Every call names its workspace. Listing accepts `glob=` with `*`
and `?`; uploads also support `kind="eval-only-dataset"` for held-out sets.

Start head-only or LoRA tuning with the uploaded dataset name. The JSONL
dataset defines each question and answer on the row; a base model requires a
destination, while an existing repository publishes back to itself:

```jsonl
{"state":"Duplicate charge","questions":{"route":{"instructions":"Route?","type":"choice","criteria":{"support":"Support","payments":"Payments"},"answer":"payments"},"severity":{"instructions":"Severity?","type":"score","criteria":["low","high"],"answer":[0.1,0.9]}}}
```

Keyed criteria use string hard labels or probability objects with exactly the
same keys. Positional criteria use zero-based integer labels or probability
arrays of the same length. Soft-label probabilities must sum to 1.
Rows do not contain split labels. Use `validation_file` and
`eval_only_dataset` for separate uploaded validation and held-out files; omitted
partitions are selected deterministically from the training file.

```python
job = modaic.fine_tunes.create(
    model="mo-1.3.1",
    destination="acme/support-priority",
    training_file="support-hard-labels",
    training_type="lora",
    lora_r=8,
    lora_alpha=16,
)
finished = modaic.fine_tunes.wait(job.id)
```

Once `finished.status == "done"`, call `acme/support-priority` normally. Pin an
immutable published checkpoint with `acme/support-priority@<full-commit-sha>`.
Fine-tuning requires `mo-1.3.1`.

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

`models.get` includes `configuration` with the model and questions saved on
the default branch (`None` until the model has a manifest), so you can diff
before updating. `models.update` with questions that already match is a
no-op: it returns `unchanged=True`, a `commit` whose `commit_sha` equals
`previous_sha`, and the stored configuration. Nothing is committed and the
checkpoint is not reset.

Alignment writes its optimized instructions into the same `questions`, so a
setup script that re-pushes the schema as authored would revert an aligned
model. Once a model has a checkpoint, `models.update` with questions that
differ from the stored ones fails with `409 alignment_would_be_discarded`.
Omit `questions` to keep the aligned instructions, or pass
`discard_alignment=True` to replace them and reset the checkpoint to 0.

`modaic.__version__` is the installed SDK version, for logging alongside
results. `models.get` also returns `commit`, the head of the default branch. Pass
`commit.commit_sha` back as `expected_head_sha` on `models.update` and a
stale client gets `409 expected_head_mismatch` instead of overwriting newer
commits.

### Branches, tags, and rollback

Every model is a Git repository, and `models` exposes its history:

```python
model = modaic.models.get(workspace="acme", model="support-priority")

# Name the aligned commit so production can pin to it.
modaic.models.create_tag(model.id, name="v1", commit_sha=model.commit.commit_sha)
modaic.decisions.create(model="acme/support-priority", revision="v1", state={...})

# Undo a bad update: move main back to the aligned commit.
commits = modaic.models.list_commits(model.id, branch="main")
modaic.models.rollback(
    model.id,
    branch="main",
    target_commit_sha=commits.commits[1].sha,
    expected_head_sha=commits.commits[0].sha,
)
```

Rollback commits the target's files back onto the branch, so `model.json`
returns with its checkpoint and metrics intact. `list_branches`,
`create_branch(name=, source_ref=)`, `delete_branch`, `list_tags`, and
`delete_tag` round out the surface.

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
| `protege` | `create`, `get`, `cancel`, `wait` |
| `resources` | `list`, `get`, `update`, `download` |

Responses are Pydantic models. Python attributes use `snake_case` even when the
wire format uses `camelCase`.

## Errors

Non-2xx responses raise `ModaicAPIError`, which exposes `status_code`, `code`,
`request_id`, `details`, and the decoded response `body`. Network failures raise
`ModaicConnectionError`; request and polling deadlines raise
`ModaicTimeoutError`.

See the complete API documentation at <https://docs.modaic.dev>.

## Signatures and nullable answers

Pass a `Signature` class as `questions` to infer requests from native Python
annotations and receive an instance of that signature. Sync, async, and
model-bound decisions support the same interface. No `response_model` is needed.

```python
from datetime import date
from typing import Literal
from modaic import Modaic, Question, Signature

class Review(Signature):
    country: Literal["fr", "us"] | None = Question(
        instructions="Which country is the address in?",
        criteria=["France", "United States"],
    )
    relevant: bool = Question(instructions="Is this address relevant?")
    count: int = Question(instructions="How many locations?", min=0)
    price: float = Question(instructions="What is the price?", min=0)
    published: date = Question(instructions="What is the publication date?", format="YYYY-MM-DD")
    summary: str | None = Question(instructions="Summarize the address.", nullable=0.8)

with Modaic() as client:
    result = client.decisions.create(
        model="mo-latest", state={"address": "Paris, France"}, questions=Review,
    )
    print(result.country)  # "fr", "us", or None; result is typed as Review.
```

String `Literal` annotations produce Choice questions; criteria lists correspond
to literal keys in order, and criteria dictionaries must match those keys.
`bool` produces Noul and returns `True` at probability >= 0.5. `str` produces
Text. `int` and `float` produce Number. Optional Number `min` and `max` bounds
are sent as exact decimal strings; Number does not accept `step`.
`date`, `datetime`, and `time` use the native datetime primitive with type-based format defaults.
An `int` with a criteria list produces Score instead; signature scores round to
the nearest integer (ties to even). Existing dictionary Score answers retain
their fractional values.

`T | None` automatically adds an answerability Noul with threshold 0.5.
`Question(nullable=0.8)` enables nullability or overrides that default. It also
works without `| None`, but include the union for accurate editor types.
Thresholds must be finite numbers in [0, 1]; equality accepts the answer.

Existing question dictionaries and explicit `response_model` validation still
work. For dictionaries, use `question.nullable()`, `question.nullable(0.8)`,
`question | None`, or `Nullable(question, threshold=0.8)`. A rejected answer is
`None` in `result.answers`; a named response field should allow `None` too.
`Text` and `Number` constructors are also available for dictionary questions.
`QuestionLike` describes supported question inputs for explicit annotations.

Nullable composition adds a Noul alongside the original question in one HTTP
request, then removes the helper answer and applies nullability before response
validation. Missing or malformed gates raise `ModaicConnectionError`.
`__modaic_answerable_<question-id>` is reserved for each generated gate; explicit
collisions fail before HTTP. Gates count as additional questions and both
questions run, even when the final answer is null.

Signatures and nullable wrappers are request-local features of
`decisions.create`; they are not saved-model configuration. Captured API history
contains the original answers and helper Nouls, rather than the SDK's projected
values. A signature cannot be combined with `response_model`.


### Native dates, times, text, and spans

```python
from datetime import date, datetime, time
from modaic import Question, Signature, SpanAnswer

class Document(Signature):
    published: date = Question(instructions="Publication date?")
    event: datetime | None = Question(instructions="Event time?")
    opening: time = Question(instructions="Opening time?")
    month: str = Question(instructions="Which month?", format="YYYY-MM")
    summary: str = Question(instructions="Summarize the document")
    quote: SpanAnswer = Question(instructions="Find the exact supporting passage")
```

Native `date`, `time`, and `datetime` annotations default to `YYYY-MM-DD`,
`HH:MM`, and `YYYY-MM-DDTHH:MM`, respectively, including nullable annotations.
Explicit `format` overrides these defaults. String fields with `type="datetime"`
still require an explicit format; public datetime classes supply their own defaults.
The selected format is sent as `criteria.format` on a native `datetime` question.
Supported formats: `YYYY`, `YYYY-MM`, `YYYY-MM-DD`,
`HH`, `HH:MM`, `HH:MM:SS`, `YYYY-MM-DDTHH`, `YYYY-MM-DDTHH:MM`,
`YYYY-MM-DDTHH:MM:SS`. No timezone suffixes or offsets are supported.
`date` requires the full date; `datetime` requires date plus time; `time`
requires a time-only pattern. These return native Python objects without a
timezone. Use `str` with `format` to preserve partial dates without inventing
missing fields. Missing lower time components become zero in native objects.
Invalid formats fail before HTTP; malformed or calendar-invalid answers fail
response validation. Python cannot represent year zero in native date objects;
use a string field if year zero must be preserved.

`str` without format produces Text. `SpanAnswer` produces a native `span`
question and returns `.span`, up to ten confidence-ordered disjoint candidates
in `.spans`, and top-level `.confidence`. Each candidate has `text`, `start`,
`end`, and `confidence`; `.span` is always `.spans[0]`. For only the
extracted string, use `str = Question(type="span", instructions=...)`.
Dictionary requests support `Datetime(instructions=..., format=...)`,
`Text(instructions=...)`, and `Span(instructions=...)`; each supports `.nullable()`.
Dictionary datetime answers retain the formatted string.


### Unified primitives

Public primitive classes represent both question definitions and answered objects.
Answers expose `Choice.choice`, `Score.score`, `Span.span.text`/`.span.start`/`.span.end`,
`Text.text`, `Number.number`, `Noul.noul`, and `Datetime.datetime`.
Existing `*Answer` classes remain supported for response typing and validation;
answered public objects subclass them.

`questions` accepts a dictionary or a Python Signature class. It may still be
omitted for saved models, as before. Individual primitive instances or classes
are not accepted as `questions`.

```python
result = client.decisions.create(
    model="mo-2.0", state="The event is in Paris.",
    questions={"evidence": Span(instructions="Find the city name")},
)
span = result.answers["evidence"]
print(span.span.text, span.span.start, span.span.end)
print(span.name, span.instructions)
```

Names are injected from dictionary keys or signature field names. Constructor
`name` is optional metadata; the dictionary key or signature field takes
precedence. Input question objects are not mutated by inference.

Signatures can use `evidence: Span = Question(instructions="Find evidence")`
and access `result.evidence.span.text`. Existing native annotations retain their
value extraction behavior. Existing dictionary response envelopes and explicit
`response_model` validation remain supported.

Answer fields are unavailable before inference. Explicit request serialization
excludes answer fields and local metadata. Reusing an answered object inside a
question dictionary sends its original definition. If that definition is
unavailable (e.g. omitted saved-model questions), instructions are `None`, and
resending it raises `ValueError` instead of guessing a configuration.


### Primitive signatures and generic Choice

Prefer primitive annotations when you want the API answer and its confidence or
probabilities. Native annotations remain supported for existing callers.

```python
from typing import Literal
from modaic import Choice, Date, Datetime, Noul, Question, Signature, Time

class Review(Signature):
    country: Choice[Literal["fr", "us"]] | None = Question(
        instructions="Which country?", criteria=["France", "United States"],
    )
    relevant: Noul = Question(instructions="Is this relevant?")
    published: Date = Question(instructions="Publication date?")
    opening: Time = Question(instructions="Opening time?")
    event: Datetime = Question(instructions="Event time?")
```

`result.country.choice` is statically `Literal["fr", "us"]` after a null
check. Bare `Choice` defaults to `Choice[str]`. Generic criteria may be omitted
(descriptions default to their keys), supplied as a list in literal order, or
as a dictionary with exactly the literal keys. Unexpected choice values fail
response validation. Use `Choice[Literal[...]]`, not `Choice["fr", "us"]`,
for standard static type checking.

`Date` and `Time` subclass `Datetime`. Their format defaults are `YYYY-MM-DD`,
`HH:MM`, and `YYYY-MM-DDTHH:MM` for `Datetime`. All return `.datetime` strings
plus confidence and probabilities. Explicit format overrides preserve more or
less precision. `Date` supports date-only patterns and `Time` time-only patterns.
Generic Choice and datetime subclasses are preserved in returned dictionary
answers as well as signature fields. Native `date`/`time`/`datetime` annotations
still return native Python objects with the same defaults.


### Image boxes

Use `Box(instructions=...)` inside a question dictionary, or annotate a signature
field as `Box = Question(instructions=...)`. The answer is a `Box` instance with
`.box.top`, `.box.bottom`, `.box.left`, `.box.right`, and `.confidence`.
Coordinates are normalized to [0, 1], with top < bottom and left < right.
`Box` supports nullable composition. `BoxAnswer` is available for response typing.
The wire question remains `{ "type": "box", "instructions": "..." }`.


Answered Python primitives remain mutable, and `model_dump()` / `model_dump_json()`
serialize their current answer fields. Their original question definition is
kept separately for request serialization. `bool(noul)` is true exactly when
`noul.noul > 0.5` (exactly 0.5 is false). This does not change the numeric
probability or the separate nullable answerability threshold. Evaluating an
unanswered Noul question as a boolean raises `AttributeError`.


### Response envelopes and named answers

`Signature` extends `DecisionResponse`. A signature request retains `answers`,
`model`, `usage`, request/capture metadata, and typed views, and adds statically
typed top-level fields. For primitive annotations,
`result.relevant is result.answers["relevant"]`; both expose `.noul`.
This is the same access pattern provided by a `DecisionResponse` subclass passed
as `response_model`. Signature fields also define the request questions, so no
separate response model is needed. Metadata names are reserved for the envelope.
Legacy native annotations still project values at the named field, while the
raw primitive answer remains available in `answers`.


Python Score answers use integer keys for `.legend` and `.probabilities`, matching
TypeSafe's Python SDK. This intentionally changes earlier Modaic string-keyed
maps: use `answer.probabilities[0]`, not `["0"]`. `model_dump()` preserves integer
keys; JSON serialization converts them to strings as required by JSON.
Box confidence may be `None`. Custom response-model `before` validators receive
raw answer dictionaries; unification occurs after response validation. Custom
Answer subclasses and their validation results are preserved.
## Mo 2.0 extraction and boxes

Select `mo-2.0` explicitly. `Span` extracts an exact
substring from a string state; `Box` (`type: "box"`) localizes an image region.
Both require nonempty string instructions and reject criteria.
Span answers expose `.span.text`, `.span.start`, `.span.end`, and `.confidence`; offsets count
UTF-16 code units and end is exclusive. Box answers contain
`box: {top, bottom, left, right}` normalized to [0, 1], and confidence.
Pass inline `assets` to decisions.create, mapping an asset ID to
`{type: "image", format: "png", data_base64: "..."}`. Reference IDs in
state, instructions or supported criteria; no separate upload is required.
`mo-latest` selects `mo-2.0`.
Run a JSON inference request (including inline assets) with
`modaic --json decisions create --request request.json`.

The CLI accepts these question shapes in `modaic models create WORKSPACE SLUG --config model.json` and `modaic models update WORKSPACE/MODEL --config model.json`. Put the base model and questions in the configuration; supply image assets on inference requests.

Span answers contain a nested value: `.span.text`, `.span.start`, and `.span.end`.
The API and SDK use the same `type: "span"` contract.

Span `.probabilities` is a dictionary keyed by string IDs. Currently it contains
only `"0"`, with `text`, `start`, `end`, and `probability` equal to the answer
confidence. Python entries expose attributes; TypeScript entries expose properties.

### Number answers

Number answers expose `.number`, `.confidence`, `.variance`, and optional
`.probabilities`, matching the API. Datetime answers likewise expose optional
`.probabilities`. The field is `None` unless experimental sampling was requested.
Confidence is separate from the sampled alternatives distribution and variance.

`result.numbers`, `result.texts`, and `result.datetimes` filter `result.answers`
by primitive type. They contain the same answer objects; signature fields and
explicit response-model fields remain available as before. Native `int` and
`float` signature fields extract the `.number` value.

Raw request typing is available through `NumberQuestion`, `NumberCriteria`,
`TextQuestion`, `DatetimeQuestion`, `DatetimeCriteria`, and `SpanQuestion`.
