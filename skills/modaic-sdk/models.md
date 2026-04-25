# Models

The `model=` argument to `dspy.LM(...)` is just a **LiteLLM model id** in
the form:

```
<provider>/<model-id>
```

So `together_ai/openai/gpt-oss-120b` means "the `openai/gpt-oss-120b`
model served via the `together_ai` provider", and Modaic / DSPy will
route the request through LiteLLM accordingly.

For the run to succeed on Modaic Hub, the **API key for the chosen
provider must be set as an Environment Variable on Modaic Hub**
(https://modaic.dev/settings/env-vars). For example:

| Provider prefix | Env var Modaic Hub needs |
|---|---|
| `together_ai/...` | `TOGETHER_API_KEY` |
| `openai/...` | `OPENAI_API_KEY` |
| `anthropic/...` | `ANTHROPIC_API_KEY` |
| `fireworks_ai/...` | `FIREWORKS_API_KEY` |

See LiteLLM docs for the full provider list:
https://docs.litellm.ai/docs/providers

## Supported probed models

Modaic only computes calibrated confidence scores for models it has
trained probes on. Pick one of these for an Arbiter:

| Model id (LiteLLM) | Good for |
|---|---|
| `together_ai/openai/gpt-oss-120b` **(recommended)** | General-purpose judging. Strongest reasoning of the supported set. |
| `together_ai/Qwen/Qwen3-32B` | Solid mid-tier reasoning, cheaper than gpt-oss-120b. |
| `together_ai/Qwen/Qwen3-VL-32B-Instruct` | Same tier but supports `dspy.Image` inputs (multimodal). |
| `together_ai/Qwen/Qwen3.5-4B` | Fast, cheap, low-IQ tasks. Good for testing pipeline logic and simple classification. |

If you need a model that isn't in this list (your own fine-tune, a
different base model), contact Modaic at https://modaic.dev/contact.

## Picking a model

- **Default**: `together_ai/openai/gpt-oss-120b`. It's the recommended
  general-purpose choice.
- **Multimodal input**: `together_ai/Qwen/Qwen3-VL-32B-Instruct`.
- **Cost-sensitive or trivial tasks**: `together_ai/Qwen/Qwen3.5-4B`.
  Useful for smoke-testing the pipeline before paying for a larger
  model run.

## Example

```python
import dspy
import modaic

arbiter = modaic.Predict(
    MySignature,
    lm=dspy.LM(model="together_ai/openai/gpt-oss-120b"),
).as_arbiter()
```
