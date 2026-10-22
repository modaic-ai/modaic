# resources

## upload

Upload a file and wait for validation.

```bash
modaic resources upload acme train.jsonl --kind dataset --name support-labels --repo support-model
```

| Flag / argument | What it does |
| --- | --- |
| `WORKSPACE` | Required workspace slug. |
| `PATH` | Required local file path. |
| `--kind` | Required: `dataset`, `eval-only-dataset`, or `artifact`. |
| `--name` | Required unique resource name. |
| `--description` | Optional resource description. |
| `--repo` | Repository name to link; repeat to link multiple repositories. |
| `--timeout` | Validation wait in seconds. |
| `-h, --help` | Show usage and options. |

Check the returned status before using the resource. For other files, use `--kind artifact`. See [dataset format](fine-tunes.md#dataset-format).

## list

List uploaded resources.

```bash
modaic resources list acme --kind dataset --glob 'support-*'
```

| Flag / argument | What it does |
| --- | --- |
| `WORKSPACE` | Required workspace slug. |
| `--commit-sha` | Read a specific resource revision. |
| `--kind` | Filter by `dataset` or `artifact`. |
| `--glob` | Filter names with `*` and `?` wildcards. |
| `--cursor` | Next-page cursor. Pass the previous response commit with `--commit-sha`. |
| `-h, --help` | Show usage and options. |

## get

Read resource metadata.

```bash
modaic resources get acme support-labels
```

| Flag / argument | What it does |
| --- | --- |
| `WORKSPACE` | Required workspace slug. |
| `NAME` | Required uploaded resource name. |
| `--commit-sha` | Read a specific resource revision. |
| `-h, --help` | Show usage and options. |

## download

Save resource bytes to a local file.

```bash
modaic resources download acme support-labels --output train-copy.jsonl
```

| Flag / argument | What it does |
| --- | --- |
| `WORKSPACE` | Required workspace slug. |
| `NAME` | Required uploaded resource name. |
| `--commit-sha` | Read a specific resource revision. |
| `--output` | Output path. Defaults to the resource name. |
| `-h, --help` | Show usage and options. |

## Resource identity and pagination

Names are identifiers, not paths: 1–128 lowercase letters, digits, underscores,
or hyphens, starting with a letter or digit. Names are unique across datasets
eval-only datasets, and artifacts. Use a new name for each upload.

Uploads stream file parts and wait for asynchronous validation. Check completed
status and a non-null result before using the resource name. File size limit is
8 GiB. A successful transfer is not proof of successful dataset validation.
For resumable part-level operations use the SDK; the CLI upload command starts
a new session.

For another list page, pass both nextCursor and commitSha from the previous
response:

```bash
modaic resources list acme --cursor NEXT_CURSOR --commit-sha COMMIT_SHA
```

Downloads write to --output, or the resource name if omitted; choose an output
path that will not overwrite a file the user needs.
