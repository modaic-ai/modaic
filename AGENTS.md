# Modaic Python SDK

This repository is an HTTP-only client for the public Modaic API.

- Runtime code lives in `src/modaic`.
- Keep the synchronous `Modaic` and asynchronous `AsyncModaic` surfaces in parity.
- Public Python names use `snake_case`; transport aliases map the API's wire casing.
- Runtime code must not invoke Git, subprocesses, or filesystem APIs.
- Update contract tests whenever an endpoint method or wire shape changes.
- Run `uv run ruff check`, `uv run mypy src/modaic`, and `uv run pytest` before shipping.
