"""API URL precedence, tested through real request construction without network IO."""

from typing import Any

import httpx
import pytest

from modaic import AsyncModaic, Modaic


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("injected", [False, True], ids=["owned", "injected"])
@pytest.mark.parametrize(
    ("option", "env_url", "expected"),
    [
        (None, None, "https://modaic.dev/api/v1"),
        (None, "https://env.example/v1/", "https://env.example/v1"),
        ("https://option.example/api/v1///", None, "https://option.example/api/v1"),
        ("https://option.example/v1", "https://env.example/v1", "https://option.example/v1"),
        (None, "", "https://modaic.dev/api/v1"),
        ("", "https://env.example/v1", "https://env.example/v1"),
    ],
)
async def test_api_url(
    monkeypatch: pytest.MonkeyPatch,
    asynchronous: bool,
    injected: bool,
    option: str | None,
    env_url: str | None,
    expected: str,
) -> None:
    monkeypatch.delenv("MODAIC_API_URL", raising=False)
    if env_url is not None:
        monkeypatch.setenv("MODAIC_API_URL", env_url)
    requests: list[httpx.Request] = []

    def send(_self: httpx.Client, request: httpx.Request, **_kwargs: Any) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, request=request, json={"models": []})

    async def async_send(
        _self: httpx.AsyncClient, request: httpx.Request, **_kwargs: Any
    ) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, request=request, json={"models": []})

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "send", async_send)
    if asynchronous:
        async_http = httpx.AsyncClient() if injected else None
        async with AsyncModaic(api_key="test", base_url=option, http_client=async_http) as client:
            await client.models.list()
        if async_http is not None:
            assert not async_http.is_closed
            await async_http.aclose()
    else:
        http = httpx.Client() if injected else None
        with Modaic(api_key="test", base_url=option, http_client=http) as sync_client:
            sync_client.models.list()
        if http is not None:
            assert not http.is_closed
            http.close()
    assert str(requests[0].url) == f"{expected}/models"


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_overrides_injected_client_url(
    monkeypatch: pytest.MonkeyPatch,
    asynchronous: bool,
) -> None:
    monkeypatch.setenv("MODAIC_API_URL", "https://env.example/v1/")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"models": []})

    if asynchronous:
        async with httpx.AsyncClient(
            base_url="https://injected.example/v1", transport=httpx.MockTransport(handler)
        ) as http:
            async with AsyncModaic(api_key="test", http_client=http) as client:
                await client.models.list()
            assert str(http.base_url) == "https://injected.example/v1/"
    else:
        with httpx.Client(
            base_url="https://injected.example/v1", transport=httpx.MockTransport(handler)
        ) as sync_http:
            with Modaic(api_key="test", http_client=sync_http) as sync_client:
                sync_client.models.list()
            assert str(sync_http.base_url) == "https://injected.example/v1/"
    assert str(requests[0].url) == "https://env.example/v1/models"
