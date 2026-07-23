<p align="center">
  <img alt="Modaic" src="docs/images/modaic-banner.png" width="900" />
</p>

<p align="center">
  <a href="https://docs.modaic.dev">
    <img alt="Docs" src="https://img.shields.io/badge/docs-available-brightgreen.svg" />
  </a>
  <a href="https://pypi.org/project/modaic/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/modaic" />
  </a>
  <a href="https://pypi.org/project/modaic-client/">
    <img alt="Client PyPI" src="https://img.shields.io/pypi/v/modaic-client?label=modaic-client" />
  </a>
  <a href="https://deepwiki.com/modaic-ai/modaic">
    <img alt="Ask DeepWiki" src="https://deepwiki.com/badge.svg" />
  </a>
</p>

# Modaic

Modaic is the decision layer for AI systems. It helps teams build, deploy, measure, and improve language models for discriminative tasks.

Modaic builds on [DSPy](https://dspy.ai) for its declarative AI programming interface and adds 

- Production deployment
- Git versioning
- Labeling queues
- Confidence estimation
- Batch inference jobs
- Convenient Python SDKs

## CTA

Read the [docs](https://docs.modaic.dev) <br>
Visit [modaic.dev](https://www.modaic.dev) <br>
Create an access token from [Modaic Platform](https://modaic.dev/settings/tokens).

### Arbiters

An Arbiter is a deployed language model with a finite output space. You define the inputs, outputs, and rubric; Modaic stores each prediction with the model output, reasoning, optional ground truth, and optional confidence score.

Common use cases include:

- LLM-as-a-judge evaluators for agents, assistants, and generated content
- Support triage and routing classifiers
- Content moderation and policy classifiers
- Risk, fraud, KYC, underwriting, and compliance reviewers
- Semantic tagging and data-quality gates
- Model, query, tool, and escalation routers

Arbiter outputs must be discrete so they can be measured and calibrated. In Python, use `Literal`, `modaic.Enum[...]`, or `modaic.Scale[lo, hi]`.

### Modaic Platform

Modaic Platform stores Arbiters as versioned repositories addressed as `owner/name`. Use the platform to push new revisions, load a specific branch/tag/commit, review examples, annotate ground truth, compare confidence and agreement metrics, and manage repository settings.

### Confidence and Alignment

Modaic confidence scores help you prioritize review. Instead of labeling random samples or inspecting every prediction, start with the cases where the Arbiter is least certain. Add `ground_truth` and short `ground_reasoning` annotations, then use automatic prompt optimization to compile that feedback into improved Arbiter instructions.

## Quickstart

### Install

Use the full Python SDK when you are creating and pushing Arbiters:

```bash
uv add modaic
```

or:

```bash
pip install modaic
```

Use the lightweight Python client when production code only needs to call existing
Arbiters and manage examples:

```bash
pip install modaic-client
```

### Authenticate

Generate a token in [Modaic Platform](https://modaic.dev/settings/tokens), then
export it before running SDK or CLI workflows:

```bash
export MODAIC_TOKEN=...
```

Modaic manages inference for supported models out of the box. If you bring your own provider for a supported model, configure that provider key in [Modaic Platform environment variables](https://www.modaic.dev/settings/env-vars).

## Create an Arbiter

Define your task as a DSPy signature, wrap it with `modaic.Predict`, mark it as an Arbiter, and push it to Hub.

```python
from typing import Literal

import dspy
import modaic


class SupportTriageSignature(dspy.Signature):
    """Decide how a support ticket should be handled."""

    ticket: str = dspy.InputField(desc="The incoming support ticket")
    action: Literal["refund", "faq", "escalate"] = dspy.OutputField(
        desc="How to handle the ticket"
    )


if __name__ == "__main__":
    arbiter = modaic.Predict(
        SupportTriageSignature,
        lm=dspy.LM(model="modaic/openai/gpt-oss-120b"),
    ).as_arbiter()

    arbiter.push_to_hub("your-org/support-triage")
```

## Run an Arbiter

Load the Arbiter by repo name and call it with inputs matching the signature.

```python
from modaic_client import Arbiter

arbiter = Arbiter("your-org/support-triage")

prediction = arbiter.predict(
    ticket="My payment failed twice in a row.",
    compute_confidence=True,
)

print(prediction.output)
print(prediction.reasoning)
print(prediction.confidence)
```

Open a specific revision by passing a branch, tag, or commit hash:

```python
arbiter = Arbiter("your-org/support-triage", revision="v1")
```

## Run Batch Jobs

Use `predict_all` to run one Arbiter over many examples. By default it waits until predictions are ready; pass `compute_confidence=True` and `wait_for="scores"` when you want confidence values populated before results return.

```python
from modaic_client import Arbiter

arbiter = Arbiter("your-org/support-triage")

results = arbiter.predict_all(
    examples=[
        {"input": {"ticket": "My payment failed twice in a row."}},
        {"input": {"ticket": "How do I change my plan?"}},
        {"input": {"ticket": "The app crashes when I open settings."}},
    ],
    compute_confidence=True,
    wait_for="scores",
)

for row in results:
    pred = row.predictions[0]
    print(row.example_id, pred.output, pred.confidence)
```

For advanced workflows, pass `wait_for=None` to get a `BatchJob` handle:

```python
job = arbiter.predict_all(examples=[...], wait_for=None)

print(job.status())
for event in job.events():
    print(event.event, event.status)
results = job.results()
```

To run multiple Arbiters against the same examples in a single call, use the low-level `ModaicClient.predict_all(arbiters=[...])`.

## Manage Examples and Feedback

Examples are the stored inputs, predictions, model reasoning, ground truth, and confidence scores for an Arbiter. They are the dataset you review and calibrate against.

```python
arbiter.ingest_examples([
    {
        "input": {"ticket": "My payment failed twice in a row."},
        "ground_truth": {"action": "refund"},
        "ground_reasoning": "Billing failures should be handled directly.",
        "split": "train",
    }
])

page = arbiter.list_examples(page=1, page_size=50)
example = arbiter.get_example(page.items[0].id)

arbiter.annotate_example(
    example.id,
    ground_truth={"action": "escalate"},
    ground_reasoning="Repeated payment failures need human review.",
)
```

Use this loop for production improvement:

1. Run the Arbiter on representative data.
2. Score predictions with confidence.
3. Review low-confidence or high-impact cases.
4. Add `ground_truth` and concise `ground_reasoning`.
5. Align the Arbiter and push a new version.

## SDK Surfaces

- `modaic`: full Python SDK for creating, pushing, and running Arbiters.
- `modaic-client`: lightweight Python client for services that only need to call Arbiters, manage examples, and poll jobs.
- REST and gRPC APIs: direct interfaces for non-SDK integrations.

Modaic also ships coding-agent skills for Claude Code, Cursor, Codex, and other skill-compatible agents:

```bash
npx skills add modaic-ai/modaic
```

## Documentation

- [Modaic Docs](https://docs.modaic.dev)
- [Quickstart](https://docs.modaic.dev/docs/getting_started/quickstart)
- [Python SDK Installation](https://docs.modaic.dev/docs/python_sdk/installation)
- [Python SDK Arbiters](https://docs.modaic.dev/docs/python_sdk/arbiters)
- [Python SDK Jobs](https://docs.modaic.dev/docs/python_sdk/jobs)
- [Python SDK Examples](https://docs.modaic.dev/docs/python_sdk/examples)
- [Using Arbiters](https://docs.modaic.dev/docs/arbiters/create_an_arbiter)
- [Confidence Estimation](https://docs.modaic.dev/docs/arbiters/confidence_estimation)
- [Aligning Arbiters](https://docs.modaic.dev/docs/arbiters/aligning_your_arbiter)
- [API Reference](https://docs.modaic.dev/api_reference/arbiters/run-prediction)

## Development

Install development dependencies:

```bash
uv sync --all-extras
```

Run the default test suite:

```bash
uv run pytest
```

## Support

- GitHub Issues: [github.com/modaic-ai/modaic/issues](https://github.com/modaic-ai/modaic/issues)
- Docs: [docs.modaic.dev](https://docs.modaic.dev)
- Contact: [modaic.dev/contact](https://www.modaic.dev/contact)

## License

MIT License with additional terms. See [LICENSE](LICENSE).
