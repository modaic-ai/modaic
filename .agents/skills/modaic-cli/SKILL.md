---
name: modaic-cli
description: Use the Modaic CLI for browser or headless login, models, examples, alignment jobs, dataset and artifact uploads, and fine-tuning jobs. Use for terminal workflows rather than programmatic Python or TypeScript integrations.
---

# Modaic CLI

Install the Python package with `pip install modaic`. Check `modaic --help`
and `modaic <command> --help` for options available in the installed release.

## Login and credentials

Use `modaic login`; on headless hosts use `modaic login --headless`. The user
must match the terminal code/device and confirm the request they started.
Never approve an unrelated login request or expose session files, access tokens,
refresh tokens, or API keys in output, prompts, logs, or commits.

OS keyring stores sessions by default. `--storage file` explicitly selects
private plaintext storage on POSIX; unsupported on Windows. Sessions refresh
automatically; log in again if refresh fails. Library clients and CI use API
keys rather than browser sessions.

Credential precedence is `MODAIC_API_KEY`, a stored API key, then the selected
browser profile. `MODAIC_PROFILE` selects the profile for normal commands;
login, logout, and `auth status` require `--profile` for non-default profiles.

```bash
modaic login --headless --profile work
modaic auth status --profile work
modaic logout --profile work
```

Status reports stored metadata, not remote validity. Logout attempts remote
revocation and removes local data; unconfirmed revocation exits 3.
`--local-only` removes local data without remote revocation. Environment keys
remain set. `login --token-stdin` stores an API key; `logout --api-key` removes it.
`auth login` and `auth logout` are compatibility aliases.

## Models and examples

Model paths use `workspace/model`. `models` provides list, create, get, update,
and delete. Create/update accept `--config` with model/questions JSON. Preserve
aligned questions unless the user wants to discard alignment. Delete prompts
for confirmation; use `--yes` only when deletion is authorized.

`examples` provides create, list, summary, get, annotate, and decisions.
`examples create WORKSPACE/MODEL --file` accepts JSONL, a JSON array, or an
object with an examples array. Annotation files contain `ground_truth` and/or
`ground_reasoning`. These are model examples, separate from training datasets.

## Jobs and resources

`alignments` provides create/list by model path and get/logs/cancel by job ID.
Creation uses the model lookup commit and main branch by default.

`resources upload PATH --kind dataset|artifact` uploads and waits for validation.
Use `--name` for a unique resource name and `--workspace` when needed.
Check completed status before using the resource. List/get/download inspect or
retrieve resources; pagination requires both cursor and commit.

```bash
modaic fine-tunes create --model acme/support-priority \
  --training-path train.jsonl --task-id urgent --wait
```

Use exactly one of `--training-path` (local upload) or `--training-file`
(uploaded name). Task ID must match dataset labels. Fine-tuning supports
`mo`/`mo-1.3.1`, with head_only or lora; Mo Fast is inference-only.
`fine-tunes` also provides list, get, logs, wait, and cancel. Check terminal
status/error: waiting can return failed/cancelled jobs and timeouts do not cancel.

Global `--json` and `--quiet` precede the command. For full per-command examples
and flag tables, read https://docs.modaic.dev/docs/cli.
