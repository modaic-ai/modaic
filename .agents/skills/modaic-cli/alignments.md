# alignments

## create

Start alignment. Collect and annotate examples before starting; see requirements below.

```bash
modaic alignments create acme/support-priority --branch main
```

| Flag / argument | What it does |
| --- | --- |
| `WORKSPACE/MODEL` | Required model path, such as `acme/support-priority`. |
| `--branch` | Branch to align. Default: `main`. |
| `--source-commit-sha` | Starting commit. Defaults to the commit returned by the model lookup. |
| `--max-metric-calls` | Evaluation budget. Omit to size automatically. |
| `--idempotency-key` | Retry key for this exact request. Generated when omitted. |
| `--reflection-model` | Model used to propose instruction changes. |
| `--reflection-minibatch-size` | Examples per reflection minibatch. |
| `--seed` | Random seed. Default: `0`. |
| `-h, --help` | Show usage and options. |

## list

List recent alignments for a model.

```bash
modaic alignments list acme/support-priority --limit 10
```

| Flag / argument | What it does |
| --- | --- |
| `WORKSPACE/MODEL` | Required model path, such as `acme/support-priority`. |
| `--limit` | Maximum number of jobs. Default: `20`. |
| `-h, --help` | Show usage and options. |

## get

Read alignment status and results.

```bash
modaic alignments get ALIGNMENT_ID
```

| Flag / argument | What it does |
| --- | --- |
| `ALIGNMENT_ID` | Required alignment job ID. |
| `-h, --help` | Show usage and options. |

## logs

Read available alignment output.

```bash
modaic alignments logs ALIGNMENT_ID
```

| Flag / argument | What it does |
| --- | --- |
| `ALIGNMENT_ID` | Required alignment job ID. |
| `-h, --help` | Show usage and options. |

## cancel

Request cancellation.

```bash
modaic alignments cancel ALIGNMENT_ID
```

| Flag / argument | What it does |
| --- | --- |
| `ALIGNMENT_ID` | Required alignment job ID. |
| `-h, --help` | Show usage and options. |

## Data requirements

The model must have questions with string instructions. Add representative
examples with ground-truth labels and enough data in the assigned train/test
splits. Noul labels are booleans, Choice labels are option keys, and Score labels
are zero-based criterion indices. Use [examples.md](examples.md) for ingestion
and annotation formats. Ingesting labels alone does not run inference.

An improving alignment saves a new model revision and checkpoint. Pin the
starting commit when comparing results. A completed run can leave the model
unchanged if no candidate improves it. Updating aligned questions can discard
learned instructions; see [models.md](models.md#update).
