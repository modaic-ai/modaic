# models

## list

List your models.

```bash
modaic models list
```

| Flag / argument | What it does |
| --- | --- |
| `-h, --help` | Show usage and options. |

## create

Create a model from a configuration file.

```bash
modaic models create acme support-priority --config model.json
```

| Flag / argument | What it does |
| --- | --- |
| `WORKSPACE` | Required workspace slug. |
| `SLUG` | Required model name. |
| `--config` | Path to a JSON model configuration. |
| `--model` | Base model; overrides the configuration file value. |
| `--description` | Model description. |
| `--default-branch` | Default branch name. |
| `-h, --help` | Show usage and options. |

`model.json` supplies the model and question schema:

```json
{
  "model": "mo",
  "questions": {"urgent": {"type": "noul", "instructions": "Does this need urgent action?"}}
}
```

## get

Read a model and its configuration.

```bash
modaic models get acme/support-priority
```

| Flag / argument | What it does |
| --- | --- |
| `WORKSPACE/MODEL` | Required model path, such as `acme/support-priority`. |
| `-h, --help` | Show usage and options. |

## update

Update a model configuration or description.

```bash
modaic models update acme/support-priority --description "Support triage"
```

| Flag / argument | What it does |
| --- | --- |
| `WORKSPACE/MODEL` | Required model path, such as `acme/support-priority`. |
| `--config` | JSON configuration containing model and/or questions. |
| `--description` | New description. |
| `--branch` | Branch to update. |
| `--message` | Commit message. |
| `--discard-alignment` | Allow changed questions to replace aligned instructions and reset the checkpoint. |
| `-h, --help` | Show usage and options. |

## delete

Delete a model.

```bash
modaic models delete acme/support-priority
```

| Flag / argument | What it does |
| --- | --- |
| `WORKSPACE/MODEL` | Required model path, such as `acme/support-priority`. |
| `--yes` | Skip typing the model name to confirm deletion. |
| `-h, --help` | Show usage and options. |
