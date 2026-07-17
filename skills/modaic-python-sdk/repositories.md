# Modaic Hub Repositories

Modaic Hub is organized around **repositories**. A repository is a
versioned, shareable artifact addressed by `entity/name`:

```
acme/support-triage
your-username/code-completions
```

A program repository contains:

- `README.md` — the program card. YAML frontmatter on this file holds
  arbiter-level metadata such as `is_arbiter`, `arbiter_probe`,
  `confidence_threshold`, and `slack_notifications`.
- `program.json` — serialized DSPy program (signature, LM, structure).
- `config.json` — serialized program config / signature state.
- `pyproject.toml` *(optional)* — only present when bundled with code.

Arbiter repositories additionally expose an **Annotations** page where
each prediction is logged with its input, output, reasoning, confidence,
and optional `ground_truth` / `ground_reasoning`.

## Pushing a repository

Use `push_to_hub`. For Arbiters, **always call `.as_arbiter()` first** or
the repo will not be recognized as an arbiter.

```python
import dspy
import modaic
from typing import Literal

class TicketTriage(dspy.Signature):
    """Route the ticket."""
    ticket: str = dspy.InputField()
    queue: Literal["billing", "technical", "account"] = dspy.OutputField()

arbiter = modaic.Predict(
    TicketTriage,
    lm=dspy.LM(model="together_ai/openai/gpt-oss-120b"),
).as_arbiter()

arbiter.push_to_hub(
    "your-org/support-triage",
    commit_message="initial release",
    private=True,
)
```

`push_to_hub` keyword arguments worth knowing:

- `private=True` — make the repo private.
- `branch="main"` — target branch.
- `tag="v1.0"` — also push a tag at this revision.
- `commit_message="..."` — appears in the repo's commit history.
- `metadata={...}` — merged into the README frontmatter.
- `extra_files=[...]` — extra files to ship in the repo (e.g. a
  `README.md` if you want to override the auto-generated one).

For non-Arbiter `PrecompiledProgram` pushes, `with_code=True` bundles
the source tree so `AutoProgram` can rehydrate it. (Not applicable to
plain `modaic.Predict` arbiters — `with_code` is ignored there.)

## Branches and tags

Every program repository is a git repository on Modaic Hub. You can:

- push to any branch via `branch=`
- create tags via `tag=`
- browse commits and revisions in the Hub UI

To pin a load to a specific revision, pass `revision=` (for `Arbiter`) or
`rev=` (for `AutoProgram` / `PrecompiledProgram.from_precompiled`). Both
accept a branch name, tag, or commit hash.

```python
from modaic import Arbiter, AutoProgram, PrecompiledProgram

# pin to a tag
arbiter = Arbiter("acme/support-triage", revision="v1.2")

# pin to a branch
program = AutoProgram.from_precompiled("acme/support-router", rev="experiment")

# pin to a commit
program = PrecompiledProgram.from_precompiled("acme/support-router", rev="3fa1b2c")
```

## Loading

Three load entry points, in increasing specificity:

```python
# 1. Arbiter wrapper — for live, server-side execution
from modaic import Arbiter
arbiter = Arbiter("acme/support-triage")

# 2. AutoProgram — resolves the right PrecompiledProgram subclass at load time
from modaic import AutoProgram
program = AutoProgram.from_precompiled("acme/support-triage")

# 3. Concrete class — when you know the type and want a typed instance
from modaic import PrecompiledProgram
program = PrecompiledProgram.from_precompiled("acme/support-triage")
```

`from_precompiled` also accepts a local directory path, which is useful
for testing a `save_precompiled` artifact before pushing.

## Versioning workflow

Typical lifecycle for an Arbiter:

```python
# 1. First release
arbiter.push_to_hub("acme/support-triage", commit_message="v1", tag="v1")

# 2. Iterate
arbiter.push_to_hub("acme/support-triage", branch="experiment",
                    commit_message="try a stricter rubric")

# 3. Promote
arbiter.push_to_hub("acme/support-triage", branch="main",
                    commit_message="promote experiment", tag="v2")
```

Consumers can then pin to `v1` or `v2` and upgrade deliberately.

## Visibility and settings

Per-repo settings (visible in the Hub UI under **Settings**):

- **Visibility** — public/private toggle.
- **Storage usage** — current size of stored artifacts.
- **Confidence threshold** — when an arbiter prediction gets flagged for
  human review.
- **Slack notifications** — workspace pings on job completion.
- **Delete** — destructive; requires typing the full repo name.

Confidence threshold and Slack notifications are persisted as YAML
frontmatter in `README.md` and travel with the repo.
