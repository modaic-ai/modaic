# Signatures (TypeScript)

A `Signature` defines an Arbiter's interface: optional `instructions` (the
prompt/rubric), an `input` schema, and an `output` schema. Both schemas are zod
objects, so install `zod` alongside `modaic`.

```typescript
import { Signature } from "modaic";
import { z } from "zod";

const signature = new Signature({
  instructions: "Decide whether the answer correctly addresses the question.",
  input: z.object({
    question: z.string().describe("The user's question"),
    answer: z.string().describe("The answer to judge"),
  }),
  output: z.object({
    verdict: z.string().describe("correct | incorrect"),
  }),
});
```

- **Field descriptions** come from zod `.describe(...)`. They serialize into
  `config.json` and surface to the model.
- **`instructions`** is the system prompt. If omitted, a default
  (`Given the fields ..., produce the fields ...`) is generated — always write
  real instructions for a judge.
- **Don't declare a `reasoning` output field.** `Arbiter.create` / `arbiter.update`
  inject one automatically (the analog of Python `as_arbiter()`).
- `Signature.parse("question -> answer", "instructions")` builds a simple
  signature from a string, mirroring DSPy's string signatures.

## Special field types

Import these from `modaic`. Each serializes to the **exact** JSON the Python SDK
produces, so Arbiters round-trip between the two languages with no Python-side
changes.

| TS                 | Python analog        | Serializes to                                  | Use for |
| ------------------ | -------------------- | ---------------------------------------------- | ------- |
| `Scale(lo, hi)`    | `modaic.Scale[lo,hi]`| integer enum `{"enum":[lo..hi],"type":"integer"}` | finite integer rating |
| `Enum(...values)`  | `modaic.Enum[...]`   | string enum `{"enum":[...],"type":"string"}`   | fixed set of string choices |
| `Image.field()`    | `dspy.Image`         | `dspy.Image` `$ref`                            | multimodal image input |
| `Audio.field()`    | `dspy.Audio`         | `dspy.Audio` `$ref`                            | multimodal audio input |

Prefer `Scale` / `Enum` for **outputs** — Arbiters need an enumerable output
space to calibrate confidence against; a bare `z.string()` / `z.number()` output
gives Modaic no enum.

```typescript
import { Signature, Scale, Enum, Image } from "modaic";
import { z } from "zod";

const signature = new Signature({
  instructions: "Rate the code completion and classify its language.",
  input: z.object({
    prompt: z.string().describe("The prompt"),
    screenshot: Image.field().describe("Screenshot of the editor"),
  }),
  output: z.object({
    quality: Scale(1, 4).describe("Quality from 1 (worst) to 4 (best)"),
    language: Enum("luau", "python", "other").describe("Detected language"),
  }),
});
```

Carry a multimodal value at call time by constructing the class:
`arbiter.predict({ screenshot: new Image({ url: "https://..." }) })`.
