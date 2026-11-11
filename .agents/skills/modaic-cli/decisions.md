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
| `--experimental-nullability` | Send `X-Modaic-Experimental-Nullability: true`; Span, Text, Number, and Datetime answers include `p_null`. The headline remains the best guess. |
| `--experimental-sampling` | Send `X-Modaic-Experimental-Sampling: true`; Number probabilities may include sampled alternatives. |

Number criteria accept only optional `min` and `max` exact-decimal strings.
Span answers expose `span`, `spans`, `confidence`, and—when experimental
nullability is enabled—`p_null`. `spans` contains at most ten disjoint
candidates ordered by descending confidence, and `span` is the first candidate.
