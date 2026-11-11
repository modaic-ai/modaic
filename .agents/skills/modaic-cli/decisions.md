# Decisions

Run a System One request from a JSON file:

```bash
modaic decisions create --request request.json
```

The JSON object must contain `model` and `state`. It may also contain
`questions`, `assets`, `revision`, `example_id`, `capture`, and
`reasoning_effort`.

| Flag | Purpose |
| --- | --- |
| `--request PATH` | Required JSON request body. |
| `--idempotency-key KEY` | Reuse a prior decision safely. |

Number criteria accept only optional `min` and `max` exact-decimal strings.
Span answers expose `span`, `spans`, and `confidence`. `spans` contains at most
ten disjoint candidates ordered by descending confidence, and `span` is the
first candidate.
