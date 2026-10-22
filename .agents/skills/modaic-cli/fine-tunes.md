# fine-tunes

## create

Start training an existing model. See the dataset format below for the dataset format.

```bash
modaic fine-tunes create --model acme/support-priority \
  --training-path train.jsonl --wait
```

| Flag / argument | What it does |
| --- | --- |
| `--model` | Required existing model path using `mo`/`mo-1.3.1`. |
| `--training-file` | Uploaded dataset name. Use exactly one of this or `--training-path`. |
| `--training-path` | Local dataset to upload before training. |
| `--training-name` | Name for the dataset uploaded with `--training-path`. |
| `--workspace` | Workspace for uploading the local dataset. |
| `--training-type` | `head_only` (default) or `lora`. |
| `--destination` | Destination model path. |
| `--branch` | Destination branch. API default: `main`. |
| `--source-commit-sha` | Starting model revision. |
| `--validation-file` | Uploaded validation dataset; required for evaluations. |
| `--suffix` | Optional run label. |
| `--from-checkpoint` | Checkpoint reference for compatible continued training. |
| `--idempotency-key` | Retry key for this exact request. |
| `--lora-r` | LoRA rank. |
| `--lora-alpha` | LoRA scaling. |
| `--lora-dropout` | LoRA dropout. |
| `--max-seq-length` | Maximum tokens per example. API default: `262144`. |
| `--n-epochs` | Training epochs. API default: `1`. |
| `--n-checkpoints` | Checkpoints to save. API default: `1`. |
| `--n-evals` | Evaluation count. API default: `0`. |
| `--batch-size` | Batch size or `max` (API default). |
| `--gradient-accumulation-steps` | Accumulation steps. API default: `1`. |
| `--learning-rate` | Learning rate. API default: `0.00001`. |
| `--lr-scheduler-type` | `constant` (API default), `linear`, or `cosine`. |
| `--min-lr-ratio` | Scheduler minimum learning-rate ratio. API default: `0`. |
| `--num-cycles` | Scheduler cycles. API default: `0.5`. |
| `--warmup-ratio` | Fraction of steps used for warmup. API default: `0`. |
| `--max-grad-norm` | Gradient clipping threshold. API default: `1`. |
| `--weight-decay` | Weight decay. API default: `0`. |
| `--random-seed` | Training seed. API default: `131031`. |
| `--early-stopping` | Enable early stopping; requires validation and enough evaluations. |
| `--early-stopping-patience` | Evaluations without improvement before stopping. API default: `2`. |
| `--early-stopping-min-delta` | Minimum improvement. API default: `0`. |
| `--early-stopping-warmup-evals` | Evaluations before checking for early stopping. API default: `1`. |
| `--wandb-api-key` | Weights & Biases API key. Keep it secret. |
| `--wandb-base-url` | Weights & Biases service URL. |
| `--wandb-project-name` | Weights & Biases project. |
| `--wandb-name` | Weights & Biases run name. |
| `--wandb-entity` | Weights & Biases entity. |
| `--wait` | Wait for a terminal job status. |
| `--timeout` | Wait deadline in seconds; also applies to local dataset validation. |
| `-h, --help` | Show usage and options. |

`--wait` returns failed/cancelled jobs too; check `status` and `error`. A timeout does not cancel training.

## list

List recent fine-tuning jobs in a workspace.

```bash
modaic fine-tunes list acme --limit 10
```

| Flag / argument | What it does |
| --- | --- |
| `WORKSPACE` | Required workspace slug. |
| `--model-id` | Filter by destination model UUID. |
| `--limit` | Maximum jobs. Default: `20`. |
| `-h, --help` | Show usage and options. |

## get

Read job status and results.

```bash
modaic fine-tunes get JOB_ID
```

| Flag / argument | What it does |
| --- | --- |
| `JOB_ID` | Required fine-tuning job ID. |
| `-h, --help` | Show usage and options. |

## logs

Read available training and evaluation output.

```bash
modaic fine-tunes logs JOB_ID
```

| Flag / argument | What it does |
| --- | --- |
| `JOB_ID` | Required fine-tuning job ID. |
| `-h, --help` | Show usage and options. |

## wait

Wait until the job is done, failed, or cancelled.

```bash
modaic fine-tunes wait JOB_ID --timeout 600
```

| Flag / argument | What it does |
| --- | --- |
| `JOB_ID` | Required fine-tuning job ID. |
| `--timeout` | Wait deadline in seconds. Omit to wait indefinitely. |
| `-h, --help` | Show usage and options. |

Check `status` and `error`. A timeout stops waiting without cancelling the job.

## cancel

Request cancellation.

```bash
modaic fine-tunes cancel JOB_ID
```

| Flag / argument | What it does |
| --- | --- |
| `JOB_ID` | Required fine-tuning job ID. |
| `-h, --help` | Show usage and options. |

## Dataset format

Use UTF-8 JSONL with one state and one or more self-describing questions per
row. Question IDs and primitives may differ between and within rows.

```jsonl
{"state":"Checkout is unavailable","questions":{"urgent":{"question":"Urgent?","primitive":"noul","answer":1}},"split":"train"}
{"state":"Can I change my invoice address?","questions":{"urgent":{"question":"Urgent?","primitive":"noul","answer":0}},"split":"validation"}
{"state":"All payment attempts fail","questions":{"urgent":{"question":"Urgent?","primitive":"noul","answer":0.95}},"split":"test"}
```

This illustrates the format; use representative data for a real run. Noul
answers are probabilities from 0 to 1. Choice and Score answers are zero-based
integer indices or normalized probability arrays matching the criteria length.
Supply split on every row or omit it everywhere for
automatic assignment. Training data needs nonempty train, validation, and test
partitions. Optional groupId keeps related inputs together; identical visible
inputs must not cross partitions. Optional IDs must be unique.

No blank lines. Limits: 8 GiB per file, 100,000 rows, 1 MiB per line. Upload with
`resources upload --kind dataset` or let `--training-path` upload first. An
artifact is not a training dataset.

## Run and result handling

The CLI create command targets an existing model: it has no questions/schema
flag for creating a new model directly from a base model. First create the
model using [models.md](models.md#create), then train its workspace/model path.
Mo Fast is inference-only. Vision weights remain frozen.

For evaluations, pass `--validation-file` and positive `--n-evals`. Early
stopping also needs enough evaluations: patience + warmup evaluations + 1.
Only supported LoRA recipes are accepted; rank/scaling flags are not arbitrary
trainable-module configuration.

Treat queued/training/committing/publishing as in progress. Only done indicates
a successful terminal job; failed and cancelled do not. Check error details.
A published revision is resultCommitSha in the CLI's JSON output. Reuse an
idempotency key only for the same create request. Waiting timeouts do not
cancel the job; inspect it with get before starting a duplicate run.
