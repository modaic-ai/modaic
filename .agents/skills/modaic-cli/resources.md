# resources

## upload

Upload a file and wait for validation.

```bash
modaic resources upload train.jsonl --kind dataset --name support-labels
```

| Flag / argument | What it does |
| --- | --- |
| `PATH` | Required local file path. |
| `--kind` | Required: `dataset` or `artifact`. |
| `--name` | Unique resource name. Generated when omitted. |
| `--workspace` | Workspace slug. Omit to use the authenticated account workspace. |
| `--timeout` | Validation wait in seconds. |
| `-h, --help` | Show usage and options. |

Check the returned status before using the resource. For other files, use `--kind artifact`. See [dataset format](fine-tunes.md#dataset-format).

## list

List uploaded resources.

```bash
modaic resources list --kind dataset
```

| Flag / argument | What it does |
| --- | --- |
| `--workspace` | Workspace slug. Omit to use the authenticated account workspace. |
| `--commit-sha` | Read a specific resource revision. |
| `--kind` | Filter by `dataset` or `artifact`. |
| `--cursor` | Next-page cursor. Pass the previous response commit with `--commit-sha`. |
| `-h, --help` | Show usage and options. |

## get

Read resource metadata.

```bash
modaic resources get support-labels
```

| Flag / argument | What it does |
| --- | --- |
| `NAME` | Required uploaded resource name. |
| `--workspace` | Workspace slug. Omit to use the authenticated account workspace. |
| `--commit-sha` | Read a specific resource revision. |
| `-h, --help` | Show usage and options. |

## download

Save resource bytes to a local file.

```bash
modaic resources download support-labels --output train-copy.jsonl
```

| Flag / argument | What it does |
| --- | --- |
| `NAME` | Required uploaded resource name. |
| `--workspace` | Workspace slug. Omit to use the authenticated account workspace. |
| `--commit-sha` | Read a specific resource revision. |
| `--output` | Output path. Defaults to the resource name. |
| `-h, --help` | Show usage and options. |

## Resource identity and pagination

Names are identifiers, not paths: 1–128 lowercase letters, digits, underscores,
or hyphens, starting with a letter or digit. Names are unique across datasets
and artifacts. Use a new name for each upload.

Uploads stream file parts and wait for asynchronous validation. Check completed
status and a non-null result before using the resource name. File size limit is
8 GiB. A successful transfer is not proof of successful dataset validation.
For resumable part-level operations use the SDK; the CLI upload command starts
a new session.

For another list page, pass both nextCursor and commitSha from the previous
response:

```bash
modaic resources list --workspace acme --cursor NEXT_CURSOR --commit-sha COMMIT_SHA
```

Downloads write to --output, or the resource name if omitted; choose an output
path that will not overwrite a file the user needs.
