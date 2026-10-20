# Modaic Python SDK

This repository is an HTTP-only client for the public Modaic API.

- Runtime code lives in `src/modaic`.
- Keep the synchronous `Modaic` and asynchronous `AsyncModaic` surfaces in parity.
- Public Python names use `snake_case`; transport aliases map the API's wire casing.
- Core HTTP runtime code must not invoke Git or subprocesses. Filesystem access
  is limited to the explicit path-upload helper and CLI credential/config/file
  commands; importing the SDK must never read or write local files.
- Update contract tests whenever an endpoint method or wire shape changes.
- Run `uv run ruff check`, `uv run mypy src/modaic`, and `uv run pytest` before shipping.
