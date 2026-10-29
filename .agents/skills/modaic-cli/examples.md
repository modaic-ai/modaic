# examples

## create

Upload examples from a file.

```bash
modaic examples create acme/support-priority --file examples.jsonl
```

| Flag / argument | What it does |
| --- | --- |
| `WORKSPACE/MODEL` | Required model path, such as `acme/support-priority`. |
| `--file` | Required JSONL, JSON array, or object with an `examples` array. |
| `-h, --help` | Show usage and options. |

Example JSONL row:

```json
{"state":"Checkout is unavailable","annotation":{"ground_truth":{"urgent":true}}}
```

## list

List a page of examples.

```bash
modaic examples list acme/support-priority --page 1 --page-size 20
```

| Flag / argument | What it does |
| --- | --- |
| `WORKSPACE/MODEL` | Required model path, such as `acme/support-priority`. |
| `--page` | Page number. Default: `1`. |
| `--page-size` | Examples per page. Default: `30`. |
| `-h, --help` | Show usage and options. |

## summary

Read example and annotation counts.

```bash
modaic examples summary acme/support-priority
```

| Flag / argument | What it does |
| --- | --- |
| `WORKSPACE/MODEL` | Required model path, such as `acme/support-priority`. |
| `-h, --help` | Show usage and options. |

## get

Read one example.

```bash
modaic examples get acme/support-priority EXAMPLE_ID
```

| Flag / argument | What it does |
| --- | --- |
| `WORKSPACE/MODEL` | Required model path, such as `acme/support-priority`. |
| `EXAMPLE_ID` | Required example ID. |
| `-h, --help` | Show usage and options. |

## annotate

Add or update an example label.

```bash
modaic examples annotate acme/support-priority EXAMPLE_ID --file annotation.json
```

| Flag / argument | What it does |
| --- | --- |
| `WORKSPACE/MODEL` | Required model path, such as `acme/support-priority`. |
| `EXAMPLE_ID` | Required example ID. |
| `--file` | Required JSON file with `ground_truth`. |
| `-h, --help` | Show usage and options. |

Example `annotation.json`:

```json
{"ground_truth":{"urgent":true}}
```

## decisions

Read decisions for an example.

```bash
modaic examples decisions acme/support-priority EXAMPLE_ID
```

| Flag / argument | What it does |
| --- | --- |
| `WORKSPACE/MODEL` | Required model path, such as `acme/support-priority`. |
| `EXAMPLE_ID` | Required example ID. |
| `-h, --help` | Show usage and options. |
