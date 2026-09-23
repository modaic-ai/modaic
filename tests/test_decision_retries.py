from typing import Any

import httpx
import pytest

import modaic._transport as transport_module
from modaic import DecisionResponse, ModaicAPIError, ModaicTimeoutError, NoulAnswer
from modaic._transport import AsyncTransport, SyncTransport


class ResponseModel(DecisionResponse):
    valid: NoulAnswer


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "scenario", ["pending", "conflict", "other_path", "no_key", "exhausted", "timeout", "success"]
)
async def test_replay_retries(mode: str, scenario: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transport_module, "_REPLAY_DELAYS", (0, 0, 0, 0, 0))
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if scenario == "success" or (scenario == "pending" and len(requests) == 3):
            return httpx.Response(
                200,
                json={
                    "model": "typesafe/jev-latest",
                    "answers": {"valid": {"type": "noul", "noul": 0.7}},
                    "usage": {"input_tokens": 5, "output_tokens": 0},
                },
                headers={"x-request-id": "last-request"},
            )
        return httpx.Response(
            409,
            json={
                "code": "idempotency_key_reused"
                if scenario == "conflict"
                else "decision_in_progress",
                "detail": "pending",
                "requestId": "pending-request",
            },
        )

    cls: Any = SyncTransport if mode == "sync" else AsyncTransport
    http = (httpx.Client if mode == "sync" else httpx.AsyncClient)(
        transport=httpx.MockTransport(handler)
    )
    transport = cls(
        api_key="test",
        base_url="https://example.test/v1",
        timeout=0 if scenario == "timeout" else 30,
        client=http,
    )

    async def run() -> Any:
        result = transport.request(
            "POST",
            "/models" if scenario == "other_path" else "/systemone",
            json={"state": False, "model": "typesafe/jev-latest"},
            headers={} if scenario == "no_key" else {"idempotency-key": "unchanged"},
            model=ResponseModel,
        )
        return await result if mode == "async" else result

    try:
        if scenario in ("pending", "success"):
            result = await run()
            assert isinstance(result, ResponseModel)
            assert result.valid == result.nouls["valid"]
            assert result.request_id == "last-request"
        elif scenario == "timeout":
            with pytest.raises(ModaicTimeoutError):
                await run()
        else:
            with pytest.raises(ModaicAPIError) as error:
                await run()
            assert error.value.status_code == 409
            assert error.value.request_id == "pending-request"
        expected = 3 if scenario == "pending" else 6 if scenario == "exhausted" else 1
        assert len(requests) == expected
        assert all(
            r.content == requests[0].content and r.headers == requests[0].headers for r in requests
        )
    finally:
        if mode == "sync":
            http.close()
        else:
            await http.aclose()
