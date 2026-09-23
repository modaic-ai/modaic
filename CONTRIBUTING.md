# Contributing

Install the development environment and run the checks:

```bash
uv sync --all-groups
uv run ruff format --check src/modaic tests
uv run ruff check src/modaic tests
uv run mypy src/modaic
uv run pytest
uv build
```

Keep the sync and async clients behaviorally identical. New endpoint methods
need mocked request-shape coverage and must remain HTTP-only at runtime.
