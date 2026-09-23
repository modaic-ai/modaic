from __future__ import annotations

from typing import Any


class ModaicError(Exception):
    """Base class for every SDK error."""


class ModaicConnectionError(ModaicError):
    """The API could not be reached or returned an unreadable response."""


class ModaicAPIError(ModaicError):
    """A non-successful response from the Modaic API."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        code: str | None = None,
        request_id: str | None = None,
        details: Any = None,
        body: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
        self.details = details
        self.body = body


class ModaicTimeoutError(ModaicError):
    """A client-side request or polling deadline elapsed."""
