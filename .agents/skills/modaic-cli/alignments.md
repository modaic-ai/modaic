# Instruction optimization

`instruct-opts` is an alias for `instruction-optimizations`.

Upload a validated dataset in the model's workspace, using the same JSONL
format as fine-tuning. Every row shares one question set with nonempty text
instructions. Supported labels: Noul 0/1, keyed Choice option label, ordered
Score index. Soft labels are unsupported. Files total at most 64 MiB.

```bash
modaic resources upload train.jsonl --workspace acme --kind dataset --name triage-labels
modaic instruct-opts create acme/support-priority --training-file triage-labels
modaic instruct-opts create --model mo-latest --destination acme/support-priority --training-file triage-labels
modaic instruct-opts list acme/support-priority --limit 10
modaic instruct-opts get JOB_ID
modaic instruct-opts logs JOB_ID
modaic instruct-opts cancel JOB_ID
```

Create accepts either a positional model or `--model`, required
`--training-file`, optional `--destination` (required for a base model),
`--validation-file`, `--eval-only-dataset`, `--branch` (main),
`--source-commit-sha`, `--max-metric-calls`, `--idempotency-key`,
`--reflection-model`, `--reflection-minibatch-size`, and `--seed` (0).
Omit source commit to resolve the current head and omit budget for automatic sizing.
The CLI generates an idempotency key when omitted.

Training, validation, and held-out evaluation partitions must be nonempty.
Separate validation/evaluation files replace the corresponding automatic
partitions; inputs and groups cannot overlap across partitions. Saved model
question IDs, types, and criteria must match the dataset. Existing instructions
provide the seed. Stored examples and annotations are not optimization inputs.
Completion does not automatically rerun stored predictions.

Avoid writing to the branch until the job finishes. An improving run saves a
new revision. Use get/logs to inspect progress and errors; cancelled or failed
is not successful completion. Modaic Platform can monitor and cancel runs;
start them through the API, SDK, or CLI.
