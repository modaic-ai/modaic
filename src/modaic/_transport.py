from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Mapping
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from .errors import ModaicAPIError, ModaicConnectionError, ModaicTimeoutError
from .types import DecisionResponse

DEFAULT_BASE_URL = "https://modaic.dev/api/v1"
_REPLAY_DELAYS = (0.1, 0.2, 0.4, 0.8, 1.6)


def _pending_replay(
    response: httpx.Response, method: str, path: str, headers: Mapping[str, str]
) -> bool:
    # Never retry unrelated conflicts or a request without a caller-stable key.
    return (
        method == "POST"
        and path.strip("/") == "systemone"
        and bool(headers.get("idempotency-key"))
        and response.status_code == 409
        and _api_error(response).code == "decision_in_progress"
    )


def resolve_api_key(api_key: str | None) -> str:
    value = api_key or os.getenv("MODAIC_API_KEY")
    if not value:
        raise ValueError("Set MODAIC_API_KEY or pass api_key to the client.")
    return value


def resolve_base_url(base_url: str | None, client_base_url: str | None = None) -> str:
    return (base_url or os.getenv("MODAIC_API_URL") or client_base_url or DEFAULT_BASE_URL).rstrip(
        "/"
    )


def _api_error(response: httpx.Response) -> ModaicAPIError:
    body: Any
    try:
        body = response.json()
    except ValueError:
        body = response.text or None
    request_id = response.headers.get("x-request-id")
    code = None
    details = None
    message = f"Modaic API returned HTTP {response.status_code}."
    if isinstance(body, Mapping):
        code_value = body.get("code")
        code = code_value if isinstance(code_value, str) else None
        details = body.get("details")
        detail = body.get("detail")
        if isinstance(detail, str):
            message = detail
        nested = body.get("error")
        if isinstance(nested, Mapping):
            nested_code = nested.get("code")
            nested_message = nested.get("message")
            if isinstance(nested_code, str):
                code = nested_code
            if isinstance(nested_message, str):
                message = nested_message
        body_request_id = body.get("requestId")
        if isinstance(body_request_id, str):
            request_id = body_request_id
    return ModaicAPIError(
        message,
        status_code=response.status_code,
        code=code,
        request_id=request_id,
        details=details,
        body=body,
    )


def _parse(response: httpx.Response, model: type[BaseModel] | None) -> Any:
    if response.status_code >= 400:
        raise _api_error(response)
    if response.status_code == 204 or not response.content:
        return None
    try:
        data = response.json()
    except ValueError as exc:
        raise ModaicConnectionError("Modaic API returned invalid JSON.") from exc
    if model is None:
        return data
    if issubclass(model, DecisionResponse) and isinstance(data, dict):
        data = {**data, "request_id": response.headers.get("x-request-id")}
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ModaicConnectionError("Modaic API returned an unexpected response shape.") from exc


class SyncTransport:
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        timeout: float,
        client: httpx.Client | None,
    ) -> None:
        self._api_key = resolve_api_key(api_key)
        self._owns_client = client is None
        self._base_url = resolve_base_url(base_url, str(client.base_url) if client else None)
        self._client = client or httpx.Client(base_url=self._base_url, timeout=timeout)
        self._timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        headers: Mapping[str, str] | None = None,
        model: type[BaseModel] | None = None,
    ) -> Any:
        request_headers = {"authorization": f"Bearer {self._api_key}", **(headers or {})}
        deadline = time.monotonic() + self._timeout
        try:
            response = self._client.request(
                method,
                f"{self._base_url}/{path.lstrip('/')}",
                params=params,
                json=json,
                headers=request_headers,
            )
            for delay in _REPLAY_DELAYS:
                if not _pending_replay(response, method, path, request_headers):
                    break
                if time.monotonic() + delay >= deadline:
                    raise ModaicTimeoutError("Modaic API replay timed out.")
                time.sleep(delay)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ModaicTimeoutError("Modaic API replay timed out.")
                response = self._client.request(
                    method,
                    f"{self._base_url}/{path.lstrip('/')}",
                    params=params,
                    json=json,
                    headers=request_headers,
                    timeout=remaining,
                )
        except httpx.TimeoutException as exc:
            raise ModaicTimeoutError("Modaic API request timed out.") from exc
        except httpx.HTTPError as exc:
            raise ModaicConnectionError("Could not reach the Modaic API.") from exc
        return _parse(response, model)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


class AsyncTransport:
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        timeout: float,
        client: httpx.AsyncClient | None,
    ) -> None:
        self._api_key = resolve_api_key(api_key)
        self._owns_client = client is None
        self._base_url = resolve_base_url(base_url, str(client.base_url) if client else None)
        self._client = client or httpx.AsyncClient(base_url=self._base_url, timeout=timeout)
        self._timeout = timeout

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        headers: Mapping[str, str] | None = None,
        model: type[BaseModel] | None = None,
    ) -> Any:
        request_headers = {"authorization": f"Bearer {self._api_key}", **(headers or {})}
        deadline = time.monotonic() + self._timeout
        try:
            response = await self._client.request(
                method,
                f"{self._base_url}/{path.lstrip('/')}",
                params=params,
                json=json,
                headers=request_headers,
            )
            for delay in _REPLAY_DELAYS:
                if not _pending_replay(response, method, path, request_headers):
                    break
                if time.monotonic() + delay >= deadline:
                    raise ModaicTimeoutError("Modaic API replay timed out.")
                await asyncio.sleep(delay)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ModaicTimeoutError("Modaic API replay timed out.")
                response = await self._client.request(
                    method,
                    f"{self._base_url}/{path.lstrip('/')}",
                    params=params,
                    json=json,
                    headers=request_headers,
                    timeout=remaining,
                )
        except httpx.TimeoutException as exc:
            raise ModaicTimeoutError("Modaic API request timed out.") from exc
        except httpx.HTTPError as exc:
            raise ModaicConnectionError("Could not reach the Modaic API.") from exc
        return _parse(response, model)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
