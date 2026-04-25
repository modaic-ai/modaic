# `dspy.Signature` types in Modaic

A Modaic Arbiter is built from a `dspy.Signature`. The signature defines
input fields, output fields, and the prompt (its docstring). For Modaic to
recognize the output space and calibrate confidence, the **output fields
must use enumerable types** — almost always `typing.Literal`.

## Supported field types

These types work in both `dspy.InputField` and `dspy.OutputField`.

| Type | Usage | Notes |
|---|---|---|
| `str` | `name: str = dspy.InputField()` | Plain text. |
| `int` | `score: int = dspy.OutputField()` | Use `Literal[1, 2, 3, 4]` instead for outputs. |
| `float` | `temperature: float = dspy.InputField()` | |
| `bool` | `is_spam: bool = dspy.OutputField()` | Equivalent to `Literal[True, False]`. |
| `Literal[...]` | `queue: Literal["billing", "tech"] = dspy.OutputField()` | **Required for output enums.** |
| `list[T]` | `tags: list[str] = dspy.InputField()` | Lists of any supported type. |
| `list[Literal[...]]` | `labels: list[Literal["a","b","c"]] = dspy.OutputField()` | Multi-label classification. |
| `dict` / `dict[str, T]` | `meta: dict = dspy.InputField()` | Free-form mappings. |
| `pydantic.BaseModel` | nested structured input or output | See below. |
| `dspy.Image` | `screenshot: dspy.Image = dspy.InputField()` | Multimodal input. |
| `dspy.Audio` | `clip: dspy.Audio = dspy.InputField()` | Multimodal input. |

`Optional[T]` / `T | None` is also accepted and is serialized as a
nullable field.

## Output fields must enumerate

Modaic's confidence estimator and the Hub UI both depend on knowing the
*finite* set of valid outputs. Always type output fields with `Literal`,
`bool`, a `BaseModel` whose own fields use `Literal`, or a `list[Literal]`
for multi-label tasks.

```python
from typing import Literal
import dspy

class Risk(dspy.Signature):
    """Rate content risk."""
    text: str = dspy.InputField()
    risk: Literal["safe", "mild", "moderate", "severe"] = dspy.OutputField()
```

## `BaseModel` outputs

`pydantic.BaseModel` is allowed as an output field type. This is useful
when you want a structured object back. Use `Literal` inside the model so
the output space stays finite.

```python
from typing import Literal
from pydantic import BaseModel
import dspy

class Verdict(BaseModel):
    label: Literal["accept", "reject", "needs_review"]
    severity: Literal[1, 2, 3, 4, 5]

class Review(dspy.Signature):
    """Review a code change."""
    diff: str = dspy.InputField()
    verdict: Verdict = dspy.OutputField()
```

## Multimodal inputs

`dspy.Image` and `dspy.Audio` are first-class input types. Useful when the
judge needs to look at a screenshot or listen to an audio clip.

```python
import dspy
from typing import Literal

class ImageSafety(dspy.Signature):
    """Decide if the image is safe for a children's product."""
    image: dspy.Image = dspy.InputField()
    label: Literal["safe", "unsafe"] = dspy.OutputField()
```

## Do NOT add a `reasoning` output

`as_arbiter()` injects a reasoning field automatically. Adding your own
`reasoning: str = dspy.OutputField()` collides with it and breaks
serialization. Just leave it off — it will appear on `result.reasoning` at
runtime.

## Instructions live in the docstring

The signature's docstring is the system prompt. Treat it as the rubric.
For long instructions, prefer `Signature.with_instructions(...)` over
pasting:

```python
SignatureCls = SignatureCls.with_instructions(open("rubric.txt").read())
```
