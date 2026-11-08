# protege

Start Protégé against an existing Modaic repository:

```bash
modaic protege create acme/support-model \
  --prompt "Improve classification quality while preserving latency" \
  --eval-only-dataset support-heldout \
  --wait
```

`repo` and `--prompt` are required. `--eval-only-dataset` names an uploaded
`eval-only-dataset` in the repository's workspace. Without `--wait`, the
command returns the job immediately.

Inspect, wait for, or cancel by globally unique job ID:

```bash
modaic protege get JOB_ID
modaic protege wait JOB_ID --timeout 3600
modaic protege cancel JOB_ID
```
