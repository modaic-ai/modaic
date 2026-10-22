---
name: modaic-cli
description: Use the Modaic CLI for browser or headless login, model and example management, alignment jobs, workspace resource uploads, fine-tuning, and Protégé jobs. Use for terminal workflows rather than programmatic Python or TypeScript integrations.
---

# Modaic CLI

Install with `pip install modaic`. This skill includes the command references,
examples, input formats, and flags needed for CLI workflows. Read the relevant
local file before using a command group; no website lookup is required.

## Commands

| Command | Purpose | Local reference |
| --- | --- | --- |
| modaic login | Sign in through a browser or save an API key from stdin. | [auth.md](auth.md#login) |
| modaic logout | Revoke/remove a browser session or remove a saved API key. | [auth.md](auth.md#logout) |
| modaic auth | Inspect the effective authentication source and session status. | [auth.md](auth.md#status) |
| modaic models | Create, list, inspect, update, and delete decision models. | [models.md](models.md) |
| modaic examples | Ingest, list, inspect, label, and view decisions for model examples. | [examples.md](examples.md) |
| modaic alignments | Start, list, inspect, read logs, and cancel alignment jobs. | [alignments.md](alignments.md) |
| modaic resources | Upload, list, inspect, and download datasets and artifacts. | [resources.md](resources.md) |
| modaic fine-tunes | Train existing models, monitor results, read logs, and cancel jobs. | [fine-tunes.md](fine-tunes.md) |
| modaic protege | Start, inspect, wait for, and cancel Protégé jobs. | [protege.md](protege.md) |

## Shared behavior

- Global --json and --quiet precede the command: `modaic --json models list`.
  --json produces compact JSON; --quiet suppresses progress output.
- Model paths use workspace/model; job commands use returned job IDs.
- Examples/annotations and fine-tuning dataset uploads have different formats.
  Use the schema in the relevant reference; do not substitute one for the other.
- Never expose API keys, session files, access tokens, or refresh tokens in
  prompts, shared output, logs, or commits. Browser confirmation belongs to the
  user who started the login. See [auth.md](auth.md) for the full flow.
- Keep aligned questions unless the user explicitly wants to discard alignment.
  Run deletion/cancellation only within the user's authorized task.
- Waiting can return failed/cancelled jobs; check status and error. A timeout
  stops waiting without cancelling the remote job.
- References describe the current CLI. Use installed `modaic --help` or
  `modaic <command> --help` to check release compatibility when needed.
