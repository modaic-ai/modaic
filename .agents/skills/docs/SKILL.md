---
name: docs
description: Use this skill whenever you are writing, editing, or reviewing pages in the Modaic public documentation (Mintlify). Covers the API reference under `docs/api_reference/`, the conceptual guides under `docs/docs/`, and the DSPy guide under `docs/dspy_guide/`. The skill encodes Modaic's house rule that public docs MUST NOT mention backend architecture, internal infrastructure, internal file paths, or internal helper functions — only the public contract.
---

# Modaic public-docs style

Modaic's public documentation describes **the public contract**: what users
send, what they get back, and what observable behavior they can rely on.
It must not describe **how Modaic implements it** behind that contract.

> This skill itself lives in a public repo, so it follows the same rule:
> **don't enumerate which specific services or vendors Modaic uses** in
> the skill body. Phrase guidance in terms of categories. If you're tempted
> to write "we use Foo for queues", stop — say "the queue / worker layer"
> instead.

## Why this rule exists

- Implementation choices change. Today's queue framework, today's column
  store, today's execution sandbox — any of these can be swapped without
  changing the public API. Doc that names them rots the moment one is
  replaced, and users build mental models around tools we never promised
  to keep.
- Naming an internal service in a public doc invites support questions
  about something we haven't committed to as part of our API surface.
- Self-hosters and on-prem deployments may run a different backend
  entirely; arch-specific doc becomes misleading for them.

## What counts as "architecture-specific" — strip these

These are internal implementation details. They MUST NOT appear in public
docs (this includes the SKILL.md itself):

- **Vendor / product names of internal infrastructure**, in any of these
  categories: queue / task / worker frameworks; in-memory caches and
  brokers; column stores, OLAP / OLTP databases; auth providers; on-demand
  execution sandboxes; git hosting backends; payments processors; chat /
  notification platforms used as internal infra (as opposed to a
  user-facing integration). Do not name the tool we currently use; do not
  even list the categories side-by-side with our names.
- **Queue / worker mechanics**: "the X task", "X worker", "X result TTL",
  task-revocation flags, signal names like SIGKILL, "the broker".
- **Storage internals**: specific table or column names from the prediction
  / example / job stores; transaction-boundary descriptions
  ("all inserts happen in a single transaction at the end"); cache-key
  formats.
- **Internal Python helpers** — anything starting with an underscore in the
  server tree, plus task-name symbols like `submit_*_task`,
  `_run_*`, `online_*`, `_peek_*`, etc.
- **Internal file paths**: anything under `server/src/...`,
  `client/src/...`, etc.
- **Process model / topology**: "the gRPC server runs in-process with the
  HTTP app", "the worker pool", "the sandbox spins up a fresh container".
- **Internal dependency wrapping** that the user doesn't call directly:
  "uses dspy.Parallel under the hood", "wraps gepa-ai/gepa". (`GEPA`
  itself is fine because it's a user-facing optimizer name and
  `gepa_kwargs` is a public request field — but don't describe the
  wrapping.)

## What stays — these are user-facing and required for accuracy

- HTTP method, path, status codes, headers, content types
  (`application/x-ndjson`, `application/vnd.apache.parquet`, etc.).
- Request body / query / path parameter names and types **exactly** as
  they appear on the wire. Same for response field names.
- Behavior the user can observe: idempotency, ordering, pagination,
  rate limits, auth requirements, what triggers each error code.
- Public Python SDK names (`modaic.Predict`, `modaic.Predict.from_precompiled`,
  `dspy.Signature`, etc.) when used as the recommended caller-side API.
- Public concepts: arbiters, examples, predictions, ground truth,
  confidence scores, GEPA, Modaic Hub, repo (in the `owner/repo` sense).
- **Concrete state values that the API actually returns** — e.g. job
  status strings like `PENDING`, `PROGRESS`, `SUCCESS`, `FAILURE`,
  `REVOKED`, or `queued`, `completed`, `failed`. Document the values
  without naming the framework that produces them.
- S3 mentions for the export/download endpoints — exported files are
  user-visible S3 objects accessed via presigned URL, so calling out
  S3 is part of the public contract for those endpoints.
- Wire-format snippets — the literal `.proto` definition is fine; the
  path to the proto file in the server tree is not.

## Substitutions

When the underlying implementation does need to be alluded to, prefer
generic phrasing:

| Internal phrasing | Use instead |
| --- | --- |
| Named queue/task framework + "task" | "background job" |
| "X worker" | "the worker" |
| "X task id" | "job id" |
| Named column store + "row" | "the stored prediction" / elide |
| Named cache + "key" | elide; or "the in-flight job tracker" if you must |
| Named execution sandbox | "the optimization sandbox" / "the job runtime" |
| "revoke(terminate=True)" | "cancels the running job" |
| "X result TTL" | "after the result cache window expires" or elide |

## Process

When writing or editing a page:

1. Start from the route handler / proto / SDK function. Lift only the
   public contract (signature, validation rules, response shape, status
   codes, observable behavior).
2. Re-read the draft and grep your own text for any vendor / product /
   internal-symbol name. If one slipped in, swap for the generic phrasing
   above.
3. The litmus test: *if Modaic swapped any one piece of its backend
   tomorrow, would this page still be correct?* If no, the page is
   leaking architecture.
4. The same litmus test applies to anything you write **into this skill
   file**.

## Mirror doc updates into `./skills`

Any change to a public docs page must also be reflected in the corresponding
skill under `./skills/` (e.g. `modaic-sdk`, `modaic-api`,
`modaic-troubleshooting`, `modaic-hub`). The skills are the agent-facing
counterpart to the human-facing docs and drift between them silently breaks
agent behavior.

After editing a docs page, **ask the user how they'd like the change
reflected in `modaic/skills`** before writing anything there. Options to
offer:

- Update an existing skill in place (name which one).
- Add a new section to an existing skill.
- Create a new skill (rare — confirm scope first).
- Skip if the change is purely cosmetic (typo, formatting).

Don't guess. The mapping from doc page to skill isn't always 1:1, and the
user may want different phrasing or depth for the agent audience.

## When in doubt

If you can't tell whether a piece of information is part of the public
contract or an implementation leak, ask the user. Defaults:

- **Field names** the user sends or receives: keep (they are the contract).
- **Status / state values** the API returns in JSON: keep (also part of
  the contract).
- **Sentences describing how we process the request internally**:
  rewrite to describe what the user observes, not what the system does.
