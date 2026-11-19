# datasets

## upload

Upload a file and wait for validation.

```bash
modaic datasets upload acme train.jsonl --kind dataset --name support-labels --repo support-model
```

| Flag / argument | What it does |
| --- | --- |
| `WORKSPACE` | Required workspace slug. |
| `PATH` | Required local file path. |
| `--kind` | Required: `dataset`, `eval-only-dataset`, or `artifact`. |
| `--name` | Required unique dataset name. |
| `--description` | Optional dataset description. |
| `--repo` | Repository name to link; repeat to link multiple repositories. |
| `--timeout` | Validation wait in seconds. |
| `-h, --help` | Show usage and options. |

Check the returned status before using the dataset. For other files, use `--kind artifact`. See [dataset format](fine-tunes.md#dataset-format).

## list

List uploaded datasets.

```bash
modaic datasets list acme --kind dataset --glob 'support-*'
```

| Flag / argument | What it does |
| --- | --- |
| `WORKSPACE` | Required workspace slug. |
| `--commit-sha` | Read a specific dataset revision. |
| `--kind` | Filter by `dataset`, `eval-only-dataset`, or `artifact`. |
| `--glob` | Filter names with `*` and `?` wildcards. |
| `--cursor` | Next-page cursor. Pass the previous response commit with `--commit-sha`. |
| `-h, --help` | Show usage and options. |

## get

Read dataset metadata.

```bash
modaic datasets get acme support-labels
```

| Flag / argument | What it does |
| --- | --- |
| `WORKSPACE` | Required workspace slug. |
| `NAME` | Required uploaded dataset name. |
| `--commit-sha` | Read a specific dataset revision. |
| `-h, --help` | Show usage and options. |

## download

Save dataset bytes to a local file.

```bash
modaic datasets download acme support-labels --output train-copy.jsonl
```

| Flag / argument | What it does |
| --- | --- |
| `WORKSPACE` | Required workspace slug. |
| `NAME` | Required uploaded dataset name. |
| `--commit-sha` | Read a specific dataset revision. |
| `--output` | Output path. Defaults to the dataset name. |
| `-h, --help` | Show usage and options. |

## Dataset identity and pagination

Names are identifiers, not paths: 1–128 lowercase letters, digits, underscores,
or hyphens, starting with a letter or digit. Names are unique across datasets
eval-only datasets, and artifacts. Use a new name for each upload.

Uploads stream file parts and wait for asynchronous validation. Check completed
status and a non-null result before using the dataset name. File size limit is
8 GiB. A successful transfer is not proof of successful dataset validation.
For resumable part-level operations use the SDK; the CLI upload command starts
a new session.

For another list page, pass both nextCursor and commitSha from the previous
response:

```bash
modaic datasets list acme --cursor NEXT_CURSOR --commit-sha COMMIT_SHA
```

Downloads write to --output, or the dataset name if omitted; choose an output
path that will not overwrite a file the user needs.


## delete

Permanently delete a dataset, including its retained versions and large-file
parts. Content still used by another dataset is retained for that dataset.
Queued, running, and paused jobs using this dataset block deletion with HTTP 409.

```bash
modaic datasets delete acme support-labels
```

The command waits for completion by default. `--no-wait` returns the deletion
request immediately; `--timeout SECONDS` limits the wait without cancelling the
cleanup. Reads are revoked immediately and physical cleanup follows after
outstanding upload grants have expired. This is irreversible.
