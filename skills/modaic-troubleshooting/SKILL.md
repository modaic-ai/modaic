---
name: modaic-troubleshooting
description: Diagnose common Modaic SDK / Hub failures — provider API key not set on Modaic Hub, MODAIC_TOKEN missing locally, missing git/rsync, restrictive sandboxes that can't install the full SDK. Use this skill when a Modaic call fails, an arbiter run errors out, or `push_to_hub` won't go through.
---

# Modaic Troubleshooting

Quick triage for the most common Modaic failure modes. Each section
lists the symptom, the root cause, and the fix.

## 1. Provider API key not set on Modaic Hub

**Symptom**

- `arbiter(...)` / `arbiter.predict(...)` fails server-side with an
  authentication or "missing key" error from the model provider
  (Together, OpenAI, Fireworks, Anthropic, etc.).
- The error references `TOGETHER_API_KEY` / `OPENAI_API_KEY` / etc. even
  though you never call that provider directly from your code.
- The arbiter was created and pushed successfully, but no run has ever
  completed.

**Cause**

Arbiter runs execute on Modaic's backend, not on your machine. Modaic
needs the **provider's API key for the configured model** to be set as
an Environment Variable on Modaic Hub. The local environment is
irrelevant for runs.

**Fix**

1. Go to https://modaic.dev/settings/env-vars.
2. Add the env var for the provider in your `dspy.LM(model=...)` string.
   - `together_ai/...` → `TOGETHER_API_KEY`
   - `openai/...` → `OPENAI_API_KEY`
   - `anthropic/...` → `ANTHROPIC_API_KEY`
   - `fireworks_ai/...` → `FIREWORKS_API_KEY`
3. Re-run the arbiter.

## 2. `MODAIC_TOKEN` not set locally

**Symptom**

- `from modaic import Arbiter; Arbiter("org/name")` raises an
  authentication / 401 error.
- `push_to_hub` fails with "no access token" or 401.
- `ModaicClient()` raises `AuthenticationError`.

**Cause**

The Modaic SDK reads `MODAIC_TOKEN` from your environment to
authenticate to Modaic Hub. It is **not** the same as the model
provider's API key.

**Fix**

```bash
# 1. Generate a token at https://modaic.dev/settings/tokens
# 2. Set it
export MODAIC_TOKEN=mk_...
```

For non-shell environments (cron, CI, Braintrust scorers, LangSmith),
set `MODAIC_TOKEN` in that environment's secret store. Do not commit it.

You can also pass it explicitly:

```python
from modaic import ModaicClient
client = ModaicClient(modaic_token="mk_...")
```

## 3. `git` not installed

**Symptom**

- `push_to_hub` errors with `git: command not found`, "git executable
  not on PATH", or a subprocess error referencing `git`.
- `from_precompiled` fails to clone a repo.

**Cause**

Modaic uses `git` under the hood to push and pull program repositories.
A `git` binary on `PATH` is required.

**Fix**

- macOS: `xcode-select --install` or `brew install git`.
- Debian/Ubuntu: `sudo apt-get install git`.
- Windows: install [Git for Windows](https://git-scm.com/download/win).
- Restricted environments where you can't install `git` → see issue 5
  below; switch to `modaic_client`-only and the REST API.

## 4. `rsync` (or `robocopy` on Windows) not installed

**Symptom**

- `push_to_hub` / `save_precompiled` fails with
  `rsync: command not found` on macOS / Linux.
- On Windows, the same code path fails complaining about `robocopy`
  (the equivalent that Modaic uses on Windows).

**Cause**

Modaic uses `rsync` (Linux/macOS) or `robocopy` (Windows) to stage
repository files when bundling and pushing. If neither is on `PATH`,
the bundling step fails.

**Fix**

- macOS: `brew install rsync` (a newer version than the system one).
- Debian/Ubuntu: `sudo apt-get install rsync`.
- Windows: `robocopy` ships with Windows by default — if it's missing,
  open an elevated PowerShell and check `where robocopy`. If it really
  isn't there, repair the Windows install or run from a standard
  Windows shell rather than a stripped-down container.
- Restricted environments where you can't install either → see issue 5.

## 5. Restrictive environment / sandbox

**Symptom**

- You're in a serverless function, a Braintrust scorer, a sandboxed
  CI runner, or a locked-down container.
- `pip install modaic` fails or times out (litellm / dspy pull a lot
  of transitive deps).
- You don't have `git` / `rsync` available and can't install them.
- You only need to **call** an existing arbiter, not build or push one.

**Cause**

The full `modaic` package brings in DSPy and LiteLLM. That stack is
heavy, slow to cold-start, and unnecessary if you're just hitting the
Modaic API.

**Fix**

Switch to the lightweight client-only package:

```bash
pip install modaic-client
```

```python
from modaic_client import Arbiter, ModaicClient

arbiter = Arbiter("your-org/support-triage")
result = arbiter.predict(ticket="My payment failed twice in a row.")
```

`modaic_client` exposes the same `Arbiter` / `ModaicClient` classes as
`modaic`, but ships without DSPy or LiteLLM, has no `git` / `rsync`
dependency, and starts up fast.

If even `modaic-client` can't be installed, you can call the REST API
directly with `httpx` / `requests` / `fetch` — see the `modaic-api`
skill for endpoint examples.

## Quick decision tree

```
arbiter run failing server-side?         → 1. Provider key on Modaic Hub
local SDK call returning 401?            → 2. MODAIC_TOKEN
push_to_hub: "git: command not found"?   → 3. install git
push_to_hub: "rsync: command not found"? → 4. install rsync (or robocopy on Windows)
can't install full SDK at all?           → 5. use modaic-client (or REST API)
```
