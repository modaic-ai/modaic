---
name: modaic-typescript-sdk
description: Build, push, and run Modaic Arbiters (LLM judges with calibrated confidence scores) from TypeScript or JavaScript using the `modaic` npm package. Use this skill whenever the user is creating an Arbiter with a zod `Signature`, calling `Arbiter.create`, `arbiter.predict`, or `arbiter.update`, or running a deployed Arbiter from a Node/Bun/Deno app. For Python, use the `modaic-python-sdk` skill instead.
---

# Modaic TypeScript SDK

## What is Modaic?

Modaic helps you build LLM judges that emit **a decision and a calibrated
confidence score**. The confidence is computed from the model's *hidden states*
(mechanistic interpretability), not verbalized confidence or token logprobs, so
it stays well-calibrated. These judges are called **Arbiters** and are meant for
any task with a **discrete / enumerable output space** (classification,
extraction, finite-scale rating, triage/routing).

The TypeScript `Arbiter` is a thin wrapper over the Modaic REST API and git — it
**never runs an LLM locally**. `predict()` calls the API and the Modaic server
runs the model; `create()` / `update()` write the judge's `config.json`
(signature schema) and `program.json` (stored prompt) and push them to Modaic Hub
via git. These files are byte-compatible with the Python SDK, so an Arbiter
authored in either language round-trips into the other with no changes.

## Install

The npm package is named `modaic`. Signatures are built with [zod](https://zod.dev), so install both:

```bash
npm add modaic zod      # or: pnpm add / bun add / yarn add
```

ESM-only; targets Node.js 18+, Bun, and Deno. In a CommonJS project, load it with
`await import("modaic")`.

## Authentication & configuration

The SDK reads the access token from `MODAIC_TOKEN` (or pass `access_token` to
`create` / `update`). Get a token at https://modaic.dev/settings/tokens.

```bash
export MODAIC_TOKEN=...
```

| Env var          | Default                  | Purpose                                 |
| ---------------- | ------------------------ | --------------------------------------- |
| `MODAIC_TOKEN`   | —                        | Access token (API + git auth)           |
| `MODAIC_API_URL` | `https://api.modaic.dev` | Modaic REST API base URL                |
| `MODAIC_GIT_URL` | `https://git.modaic.dev` | Modaic Hub git host                     |
| `MODAIC_CACHE`   | `~/.cache/modaic`        | Staging dir for git working trees       |

## Creating an Arbiter

Define inputs/outputs with a `Signature` (backed by zod), then call
`Arbiter.create`. It writes `config.json` + `program.json` and pushes them to the
hub (private by default).

```typescript
import { Arbiter, Signature, Enum } from "modaic";
import { z } from "zod";

const signature = new Signature({
  instructions:
    "Classify the support ticket into the right queue. Use `billing` for " +
    "payment/refund/invoice issues, `technical` for bugs, `account` for login.",
  input: z.object({
    ticket: z.string().describe("The user-submitted support ticket"),
  }),
  output: z.object({
    queue: Enum("billing", "technical", "account").describe("Which queue owns this ticket"),
  }),
});

const arbiter = await Arbiter.create({
  repo: "your-org/support-triage",
  signature,
  model: "together_ai/openai/gpt-oss-120b", // LiteLLM "<provider>/<model>" string
  private: true,
  commit_message: "initial judge",
});
```

### Rules for signatures

- **Output fields should be enumerable** so Modaic has a finite space to
  calibrate against. Use `Enum(...values)` (string choices) or `Scale(lo, hi)`
  (integer rating) for outputs rather than a bare `z.string()` / `z.number()`.
  See `signatures.md`.
- **Do NOT add a `reasoning` output field yourself.** `Arbiter.create` /
  `arbiter.update` inject one automatically (the `reasoning` field the server
  reads on every prediction), exactly like Python's `as_arbiter()`.
- **`instructions` is the prompt.** Write it like a rubric — it's the task
  definition sent to the model. Defaults are generated if omitted, but always
  write real instructions for a judge.
- **`model` is required** on `create` (and on `update` when you pass a new
  `signature`). It's a LiteLLM model string in `<provider>/<model>` form and is
  written into `program.json`. An Arbiter with no model can't be run.
- Field descriptions come from zod `.describe(...)`.

### Required: provider API key on Modaic Hub

The server runs the model, so the **provider API key for the Arbiter's model
must be set as an Environment Variable on Modaic Hub**
(https://modaic.dev/settings/env-vars). A `together_ai/...` model needs
`TOGETHER_API_KEY`. Without it, `predict()` fails server-side.

## Running an Arbiter

Construct a handle to an existing repo and call `predict()`. No network call
happens on construction; `predict()` runs the judge server-side in one HTTP
request. Keys must match the signature's input fields.

```typescript
import { Arbiter } from "modaic";

const arbiter = new Arbiter("your-org/support-triage");

const result = await arbiter.predict({
  ticket: "My payment failed twice in a row.",
});

console.log(result.output?.queue); // "billing"
console.log(result.reasoning);     // auto-generated reasoning
```

`arbiter.call(...)` is an alias for `predict(...)`. Open a specific revision with
`new Arbiter("owner/name", { rev: "v1" })` (a branch, tag, or commit).

### Predict options & confidence

`predict(input, opts)` accepts `ground_truth`, `ground_reasoning`, and
`compute_confidence`. The prediction response is
`{ exampleId, predictionId, output, reasoning, messages }` — note it does **not**
carry a `confidence` field inline (unlike the Python SDK's lazy
`prediction.confidence`). Pass `compute_confidence: true` to enqueue calibrated
confidence scoring, which runs asynchronously after the prediction is persisted;
read the score back via the REST API / Modaic Hub once scoring finishes.

## Updating an Arbiter

Push a new revision. Pass a new `signature` (with `model`) to rewrite the schema
and prompt, or just `metadata` to update the Arbiter card.

```typescript
await arbiter.update({
  signature,
  model: "together_ai/openai/gpt-oss-120b",
  commit_message: "tweak prompt",
});
```

## Low-level REST client

For direct typed access to every endpoint, use `ModaicClient`. Methods are
grouped into resources — `examples`, `jobs`, `predictions`, `arbiters`. Each
method throws `ModaicClientError` on HTTP errors and is also exported as a
tree-shakeable standalone function (e.g. `predictionsCreate`).

```typescript
import { ModaicClient } from "modaic";

const client = new ModaicClient({ token: process.env.MODAIC_TOKEN ?? "" });

// the call Arbiter.predict() makes under the hood:
const result = await client.predictions.create({
  input: { ticket: "..." },
  arbiterRepo: "your-org/support-triage",
});
```

> `client.predictions.create` is the single-arbiter v2 endpoint
> (`POST /api/v2/arbiters/predictions`) — what `Arbiter.predict` calls. The
> original v1 multi-arbiter endpoint is now `client.predictions.createV1`. (Both
> replaced the old flat `client.createPredictionApiV2ArbitersPredictionsPost`
> naming.)

`Arbiter.predict` is the high-level path and is preferred for running judges;
reach for `ModaicClient` only when you need an endpoint `Arbiter` doesn't wrap.

> **Running multiple arbiters at once:** the `Arbiter` class only runs one judge
> per call. To score the same input with several arbiters in a single request, use
> the low-level `client.predictions.createV1({ input, arbiters: [...] })` — it's the
> only way to run multiple arbiters at the same time.

### Method reference

- **`client.examples`** — `ingest` (`POST /api/v1/examples`), `list`
  (`GET /api/v1/examples`), `delete` (`DELETE /api/v1/examples`), `export`
  (`POST /api/v1/examples/export`), `downloadExport`
  (`GET /api/v1/examples/export/download`), `getDistinctHashes`
  (`GET /api/v1/examples/distinct-hashes`), `hasUncalibrated`
  (`GET /api/v1/examples/has-uncalibrated`), `getGradedCount`
  (`GET /api/v1/examples/graded-count`), `patchAnnotation`
  (`PATCH /api/v1/examples/{example_id}/annotation`), `get`
  (`GET /api/v1/examples/{example_id}`)
- **`client.jobs`** — `startConfidenceScore` / `getConfidenceScoreStatus` /
  `cancelConfidenceScore` (`/api/v1/jobs/confidence-scores[/{job_id}]`),
  `startOptimization` / `getOptimizationStatus` / `cancelOptimization` /
  `getOptimizationLogs` (`/api/v1/jobs/gepa[/{job_id}[/logs]]`),
  `startBatchPredictions` / `getBatchPredictionStatus` / `cancelBatchPrediction` /
  `streamBatchEvents` / `streamBatchResults`
  (`/api/v1/jobs/batch/predictions[/{job_id}[/events|/results]]`)
- **`client.predictions`** — `create` (`POST /api/v2/arbiters/predictions`,
  single arbiter — what `Arbiter.predict` calls), `createV1`
  (`POST /api/v1/arbiters/predictions`, v1 multi-arbiter), `getConfidence`
  (`POST /api/v1/arbiters/predictions/confidence`), `enqueueConfidence` /
  `getConfidenceStatus` / `streamConfidence`
  (`/api/v1/arbiters/predictions/{prediction_id}/confidence[/stream]`)
- **`client.arbiters`** — `get` (`GET /api/v1/arbiters/`), `getSchema`
  (`GET /api/v1/arbiters/schema`), `getSupportedModels`
  (`GET /api/v1/arbiters/supported-models`), `updateMetadata`
  (`PATCH /api/v1/arbiters/metadata`)

## Reference files

- `signatures.md` — `Signature` shape, field descriptions, and the special field
  types (`Scale`, `Enum`, `Image`, `Audio`) with how they map to the Python SDK.
- `create_arbiter.ts` — runnable example: define + push an Arbiter.
- `run_arbiter.ts` — runnable example: load + call an Arbiter.
