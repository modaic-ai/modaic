# Auto-Capping Batch Refactor Plan

Token-aware shard sizing and wave-based enqueued-scope concurrency for
`modaic.batch`. All config (limits, token counting, enqueued budgets) lives
directly on `BatchClient` subclasses as class-level attribute defaults,
overridable via constructor kwargs. `BatchProvider` is removed.

---

## 1. Resolved Decisions

**Override semantics for enqueued limits**: user-set constructor kwargs win
unconditionally over the per-model `enqueued_limits_fn` lookup. A user who
bothers to construct a custom client means the override to apply globally.

**Option A (flatten)**: `BatchProvider` is deleted. Every limit/config field
moves onto `BatchClient` as a class attribute. `JSONLShardWriter` and
`BatchJobRunner` read limits directly off the client instead of
`client.provider.*`.

---

## 2. File-Level Changes

| File | Change |
|---|---|
| `provider.py` | **Deleted** — all fields absorbed into client classes |
| `token_counting.py` | **New** — per-provider token counting functions |
| `enqueued_limits.py` | **New** — per-model enqueued limit lookup functions |
| `_experimental.py` | **New** — `@experimental` decorator |
| `clients/base.py` | Add all limit/config fields + cap properties to `BatchClient` base |
| `clients/openai.py` | Shadow class attrs with OpenAI defaults; add override kwargs |
| `clients/anthropic.py` | Shadow class attrs; add override kwargs; apply `@experimental` |
| `clients/azure.py` | Shadow class attrs; add override kwargs; apply `@experimental` |
| `clients/together.py` | Shadow class attrs; add override kwargs |
| `clients/fireworks.py` | Shadow class attrs; add override kwargs |
| `clients/vertex.py` | Shadow class attrs; add override kwargs; apply `@experimental` |
| `clients/vllm.py` | Remove inline `BatchProvider(...)` construction; set attrs directly |
| `writer.py` | Accept `BatchClient` instead of `BatchProvider`; add token tracking |
| `runner.py` | Read limits from `self.client.*`; replace `_execute_shards` with wave loop |
| `batch.py` | Remove `provider=` kwarg from `abatch()`; remove `get_batch_client()` provider threading |

---

## 3. Token Counting (`token_counting.py`)

```python
TokenCounter = Callable[[str, list[dict]], Optional[int]]
```

- **`count_tokens_tiktoken(model, messages) -> int`** — OpenAI / Azure. Uses
  `tiktoken.encoding_for_model()` with `cl100k_base` fallback on unknown models.
- **`count_tokens_hf(model, messages) -> int`** — Together / Fireworks / Vertex.
  Uses `tokenizers.Tokenizer.from_pretrained(model)`; applies chat template when
  available; sums token ids.
- **`count_tokens_anthropic(model, messages) -> int`** — Anthropic. Uses
  `anthropic.count_tokens()` which is local (tiktoken-based), not an API call.
- **`count_tokens_none(model, messages) -> None`** — no-op for vllm and any
  client where token limits are irrelevant.

All functions share the same `(str, list[dict]) -> Optional[int]` signature.
`None` return signals "skip token tracking" to the writer and runner.

---

## 4. Enqueued Limits (`enqueued_limits.py`)

```python
@dataclass(frozen=True)
class EnqueuedLimits:
    max_enqueued_reqs: Optional[int] = None
    max_enqueued_tokens: Optional[int] = None
    max_enqueued_jobs: Optional[int] = None

EnqueuedLimitsFn = Callable[[str], EnqueuedLimits]
```

Per-provider implementations:

- **`openai_enqueued_limits(model)`** — dict lookup against OpenAI TPD table
  (from `limits/openai.txt`). Key entries:

  | Model family | TPD |
  |---|---|
  | `gpt-5`, `gpt-5.1`, `gpt-5.2`, `gpt-5.3-*`, `gpt-5.4` | 900k–1.5M |
  | `gpt-5-mini` | 5M |
  | `gpt-5-nano`, `gpt-5.4-mini`, `gpt-5.4-nano` | 2M |
  | `gpt-5-pro`, `gpt-4o`, `gpt-4-turbo` | 90k |
  | `gpt-4.1` | 900k |
  | `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-4.1 (long context)` | 2M–4M |
  | `gpt-4o-mini` | 2M |
  | `gpt-4` | 100k |
  | `gpt-3.5-turbo` | 2M |

  Resolution: exact match → strip date suffix (`-YYYY-MM-DD`) → family prefix
  match → `90k` fallback. Populates only `max_enqueued_tokens`.

- **`azure_enqueued_limits(model)`** — Default-tier Azure TPD table:

  ```python
  AZURE_DEFAULT_ETPD = {
      "gpt-4.1": 200_000_000,
      "gpt-4.1-mini": 1_000_000_000,
      "gpt-4.1-nano": 1_000_000_000,
      "gpt-4o": 200_000_000,
      "gpt-4o-mini": 1_000_000_000,
      "gpt-4-turbo": 80_000_000,
      "gpt-4": 30_000_000,
      "o3-mini": 1_000_000_000,
      "o4-mini": 1_000_000_000,
      "gpt-5": 200_000_000,
      "gpt-5.1": 200_000_000,
  }
  ```

  Populates `max_enqueued_tokens`; `max_enqueued_jobs` comes from the client's
  class attribute (`500`).

- **`anthropic_enqueued_limits(_)`** — `EnqueuedLimits(max_enqueued_reqs=100_000)`.
- **`together_enqueued_limits(_)`** — `EnqueuedLimits(max_enqueued_tokens=30_000_000_000)`.
- **`fireworks_enqueued_limits(_)`** — `EnqueuedLimits()` (all `None`).
- **`vertex_enqueued_limits(_)`** — `EnqueuedLimits()` (constraints are per-file).
- **`none_enqueued_limits(_)`** — `EnqueuedLimits()`.

---

## 5. `BatchClient` Base + Subclasses (`clients/base.py` + each client)

`BatchClient` gains all config fields as class attributes with generic defaults.
Subclasses shadow only the fields they need to change. `__init__` on each class
accepts the overridable fields as optional kwargs and sets instance attributes
when provided — instance attributes shadow class attributes at lookup time, so
no explicit "apply override" logic is needed.

### Base class additions (`clients/base.py`)

```python
class BatchClient:
    # Config — subclasses shadow these with their own defaults
    name: str = "unknown"
    reqs_per_file: int = 50_000
    max_file_size: int = sys.maxsize
    tokens_per_file: Optional[int] = None
    default_enqueued_reqs: Optional[int] = None
    default_enqueued_tokens: Optional[int] = None
    default_enqueued_jobs: Optional[int] = None
    enable_concurrent_jobs: bool = True
    concurrency: Concurrency = "parallel"
    safety_margin: float = 0.95
    endpoint: Optional[str] = None
    requires_consistent_model: bool = False
    resumable: bool = True
    token_counter: Optional[TokenCounter] = None
    enqueued_limits_fn: Optional[EnqueuedLimitsFn] = None

    @property
    def request_cap(self) -> int:
        return max(1, int(self.reqs_per_file * self.safety_margin))

    @property
    def byte_cap(self) -> int:
        if self.max_file_size >= sys.maxsize // 2:
            return self.max_file_size
        return int(self.max_file_size * self.safety_margin)

    @property
    def token_cap(self) -> Optional[int]:
        if self.tokens_per_file is None:
            return None
        return int(self.tokens_per_file * self.safety_margin)
```

### Per-client class attributes and constructor

The pattern is identical for every client. Example — `OpenAIBatchClient`:

```python
class OpenAIBatchClient(RemoteBatchClient):
    # Class-level defaults
    name = "openai"
    reqs_per_file = 50_000
    max_file_size = 200 * 1024 * 1024
    endpoint = "/v1/chat/completions"
    requires_consistent_model = True
    token_counter = count_tokens_tiktoken
    enqueued_limits_fn = openai_enqueued_limits

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        poll_interval: float = 30.0,
        max_poll_time: str = "24h",
        # Override knobs — None means "use class default"
        reqs_per_file: Optional[int] = None,
        max_file_size: Optional[int] = None,
        tokens_per_file: Optional[int] = None,
        default_enqueued_reqs: Optional[int] = None,
        default_enqueued_tokens: Optional[int] = None,
        default_enqueued_jobs: Optional[int] = None,
        enable_concurrent_jobs: Optional[bool] = None,
    ):
        super().__init__(api_key=api_key, poll_interval=poll_interval, max_poll_time=max_poll_time)
        if reqs_per_file is not None:
            self.reqs_per_file = reqs_per_file
        if max_file_size is not None:
            self.max_file_size = max_file_size
        if tokens_per_file is not None:
            self.tokens_per_file = tokens_per_file
        if default_enqueued_reqs is not None:
            self.default_enqueued_reqs = default_enqueued_reqs
        if default_enqueued_tokens is not None:
            self.default_enqueued_tokens = default_enqueued_tokens
        if default_enqueued_jobs is not None:
            self.default_enqueued_jobs = default_enqueued_jobs
        if enable_concurrent_jobs is not None:
            self.enable_concurrent_jobs = enable_concurrent_jobs
```

Override usage:

```python
client = OpenAIBatchClient(api_key="...", reqs_per_file=10_000, default_enqueued_tokens=50_000_000)
await abatch(inputs, client=client)
```

### Per-client defaults

| Client | `name` | `reqs_per_file` | `max_file_size` | `default_enqueued_reqs` | `default_enqueued_tokens` | `default_enqueued_jobs` | `@experimental` |
|---|---|---|---|---|---|---|---|
| `OpenAIBatchClient` | `openai` | 50k | 200 MB | — | — | — | no |
| `AzureBatchClient` | `azure` | 100k | 200 MB | — | — | 500 | **yes** |
| `TogetherBatchClient` | `together_ai` | 50k | 100 MB | — | 30B | — | no |
| `AnthropicBatchClient` | `anthropic` | 100k | 256 MB | 100k | — | — | **yes** |
| `FireworksBatchClient` | `fireworks_ai` | 1M | 1 GB | — | — | — | no |
| `VertexBatchClient` | `vertex` | 200k | 1 GB | — | — | — | **yes** |
| `VLLMBatchClient` | `vllm` | `batch_size` or ∞ | ∞ | — | — | — | no |

`VLLMBatchClient` sets `self.reqs_per_file = batch_size` in its constructor
when `batch_size` is provided; drops the `BatchProvider(...)` construction
entirely.

---

## 6. `@experimental` Decorator (`_experimental.py`)

```python
def experimental(cls):
    cls._experimental = True
    orig_init = cls.__init__
    def _init(self, *args, **kwargs):
        import warnings
        warnings.warn(
            f"{cls.__name__} is experimental and not covered by tests. "
            "API may change; use with caution.",
            stacklevel=2,
        )
        orig_init(self, *args, **kwargs)
    cls.__init__ = _init
    return cls
```

Applied to `AnthropicBatchClient`, `AzureBatchClient`, `VertexBatchClient`.

---

## 7. `JSONLShardWriter` — Token-Aware (`writer.py`)

Constructor changes from `provider: BatchProvider` to `client: BatchClient`.
All `self.provider.*` references become `self.client.*`.

Add `_current_tokens: int = 0`. `add()` accepts `n_tokens: Optional[int] = None`.
`_would_exceed()` checks all three caps:

```python
def _would_exceed(self, size: int, n_tokens: Optional[int]) -> bool:
    if self._current_bytes + size > self.client.byte_cap:
        return True
    if self._current_n + 1 > self.client.request_cap:
        return True
    if n_tokens is not None and self.client.token_cap is not None:
        if self._current_tokens + n_tokens > self.client.token_cap:
            return True
    return False
```

`_roll()` resets `_current_tokens = 0`. After `finalize()`, expose
`shard_req_counts: list[int]` and `shard_token_counts: list[Optional[int]]`
(one entry per shard) for the runner's wave-selection logic.

---

## 8. `BatchJobRunner` — Wave Loop (`runner.py`)

All `self.client.provider.*` references become `self.client.*`.

Pre-compute token counts before writing shards:

```python
def _count_tokens(self, items: list[BatchRequestItem]) -> list[Optional[int]]:
    counter = self.client.token_counter
    if counter is None:
        return [None] * len(items)
    return [counter(item["model"], item["messages"]) for item in items]
```

Effective enqueued limits (override semantics — user-set class/instance attrs
win over per-model lookup):

```python
def _effective_enqueued_limits(self, model: str) -> EnqueuedLimits:
    fn = self.client.enqueued_limits_fn
    base = fn(model) if fn else EnqueuedLimits()
    return EnqueuedLimits(
        max_enqueued_reqs=self.client.default_enqueued_reqs or base.max_enqueued_reqs,
        max_enqueued_tokens=self.client.default_enqueued_tokens or base.max_enqueued_tokens,
        max_enqueued_jobs=self.client.default_enqueued_jobs or base.max_enqueued_jobs,
    )
```

Replace `_execute_shards` with `_execute_waves`:

```python
async def _execute_waves(
    self,
    shards: list[Path],
    shard_req_counts: list[int],
    shard_token_counts: list[Optional[int]],
) -> list[ShardOutcome]:
    outcomes: list[Optional[ShardOutcome]] = [None] * len(shards)
    pending: list[int] = list(range(len(shards)))

    while pending:
        wave = self._select_wave(pending, shard_req_counts, shard_token_counts)
        wave_outcomes = await asyncio.gather(
            *[self._run_one(i, shards[i]) for i in wave]
        )
        for idx, outcome in zip(wave, wave_outcomes):
            outcomes[idx] = outcome
        submitted = set(wave)
        pending = [i for i in pending if i not in submitted]

    return outcomes
```

`_select_wave()` greedily picks from `pending` until the next shard would push
the wave over `max_enqueued_reqs`, `max_enqueued_tokens`, or `max_enqueued_jobs`.
When a shard spans multiple models, the most-restrictive limit across those
models applies.

Behavior:
- All three limits `None` + `enable_concurrent_jobs=True` → one wave with all
  shards (current behavior, no change).
- `enable_concurrent_jobs=False` → one shard per wave (fully serial).
- Otherwise → bounded waves sized by the tightest budget.

---

## 9. `abatch()` Signature (`batch.py`)

`provider=` kwarg removed. `client=` is the sole override surface.

```python
async def abatch(
    inputs: GroupedBatchInputs,
    *,
    client: Optional[BatchClient] = None,
    show_progress: bool = True,
    poll_interval: float = 30.0,
    max_poll_time: str = "24h",
    return_messages: bool = False,
    mode: Mode = "parallel",
    max_concurrent: Optional[int] = None,
    cache: Any = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> list[tuple[dspy.Predict, ABatchResult]]: ...
```

When `client=None`, auto-construct from the predictor's LM as before —
`get_batch_client()` returns a default-constructed client instance (e.g.
`OpenAIBatchClient()`) with no overrides.

---

## 10. Non-Goals / Out of Scope

- No cross-process/cross-run tracking of enqueued budget. All enqueued-scope
  accounting is in-process per `abatch()` invocation.
- No deprecation of `max_concurrent` kwarg in this pass — keep as legacy alias
  mapping to `max_enqueued_reqs` wave logic. Revisit after migration.
- No changes to `adapters.py`, `progress.py`, `modal_job.py`, `storage.py`.

---

## 11. Testing Delta

- Unit tests for each `count_tokens_*` function (fixed-input expected counts).
- Unit tests for each `enqueued_limits_fn` (exact match, date-suffix strip,
  family fallback, unknown model).
- Writer tests: token-based roll with `tokens_per_file` set on client; no-op
  when `token_counter` returns `None`.
- Runner tests:
  - Single-shard batch with all limits `None` (current behavior unchanged).
  - Multi-shard batch with `default_enqueued_reqs` forcing two waves.
  - Multi-shard batch with `default_enqueued_tokens` forcing two waves.
  - `enable_concurrent_jobs=False` produces one shard per wave.
  - Constructor override: `OpenAIBatchClient(reqs_per_file=1)` splits into
    one request per shard.
  - `@experimental` warning fires on `AnthropicBatchClient()` and friends.
- Skip live integration tests for Anthropic / Azure / Vertex (experimental).
