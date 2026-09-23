from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from test_client import JOB_ID, alignment_json, batch_json
from test_regressions import Harness, call
from test_regressions import api as api

from modaic import ModaicAPIError, ModaicTimeoutError


@pytest.fixture
def bars(monkeypatch: pytest.MonkeyPatch) -> list[MagicMock]:
    created: list[MagicMock] = []

    def create(**kwargs: Any) -> MagicMock:
        bar = MagicMock()
        bar.n = 0
        bar.total = kwargs["total"]
        bar.options = kwargs

        def update(delta: int) -> None:
            bar.n += delta

        bar.update.side_effect = update
        created.append(bar)
        return bar

    monkeypatch.setattr("modaic._progress.tqdm", create)
    return created


def serve(api: Harness, jobs: list[dict[str, Any]]) -> None:
    responses: Iterator[dict[str, Any]] = iter(jobs)
    api.handler = lambda _: httpx.Response(200, json=next(responses))


@pytest.mark.parametrize("resource", ["batch_decisions", "alignments"])
@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
async def test_progress_terminal_states_and_cleanup(
    api: Harness,
    bars: list[MagicMock],
    resource: str,
    status: str,
) -> None:
    factory = batch_json if resource == "batch_decisions" else alignment_json
    serve(api, [factory("queued"), factory("running"), factory(status)])
    result = await call(getattr(api.client, resource).wait, JOB_ID, progress=True, poll_interval=0)
    assert result.status == status
    assert len(bars) == 1
    assert status in bars[0].set_postfix_str.call_args.args[0]
    bars[0].close.assert_called_once()
    assert len(api.requests) == 3
    assert all(not r.url.query and not r.content for r in api.requests)


async def test_batch_counts_failures_and_deduplicates_snapshots(
    api: Harness,
    bars: list[MagicMock],
) -> None:
    running = {**batch_json("running"), "progress": {"total": 5, "completed": 2, "failed": 1}}
    finished = {**batch_json("completed"), "progress": {"total": 5, "completed": 4, "failed": 1}}
    serve(api, [running, running, finished])
    await call(api.client.batch_decisions.wait, JOB_ID, progress=True, poll_interval=0)
    assert bars[0].n == 5
    assert bars[0].total == 5
    assert [c.args[0] for c in bars[0].update.call_args_list] == [3, 2]
    assert "1 failed" in bars[0].set_postfix_str.call_args.args[0]


async def test_alignment_unknown_total_and_early_completion(
    api: Harness,
    bars: list[MagicMock],
) -> None:
    serve(
        api,
        [
            {**alignment_json("queued"), "progress": None},
            {**alignment_json("running"), "progress": {"stage": "optimizing", "metricCalls": 2}},
            {
                **alignment_json("completed"),
                "progress": {
                    "stage": "done",
                    "metricCalls": 3,
                    "maxMetricCalls": 10,
                },
            },
        ],
    )
    await call(api.client.alignments.wait, JOB_ID, progress=True, poll_interval=0)
    assert bars[0].options["total"] is None
    assert bars[0].total == 10
    assert bars[0].n == 3  # Completion must not inflate usage to the budget.
    assert bars[0].options["unit"] == "metric calls"


@pytest.mark.parametrize("resource", ["batch_decisions", "alignments"])
async def test_default_is_silent(api: Harness, bars: list[MagicMock], resource: str) -> None:
    await call(getattr(api.client, resource).wait, JOB_ID)
    assert bars == []


@pytest.mark.parametrize("resource", ["batch_decisions", "alignments"])
@pytest.mark.parametrize("failure", ["timeout", "http", "cancel"])
async def test_progress_closes_without_hiding_errors_or_cancelling_remote_job(
    api: Harness,
    bars: list[MagicMock],
    resource: str,
    failure: str,
) -> None:
    factory = batch_json if resource == "batch_decisions" else alignment_json

    def handler(_: httpx.Request) -> httpx.Response:
        if len(api.requests) == 1:
            return httpx.Response(200, json=factory("running"))
        if failure == "cancel":
            raise asyncio.CancelledError
        return httpx.Response(503, json={"detail": "Unavailable"})

    api.handler = handler
    expected = {
        "timeout": ModaicTimeoutError,
        "http": ModaicAPIError,
        "cancel": asyncio.CancelledError,
    }[failure]
    with pytest.raises(expected):
        await call(
            getattr(api.client, resource).wait,
            JOB_ID,
            progress=True,
            timeout=0 if failure == "timeout" else 1,
            poll_interval=0,
        )
    bars[0].close.assert_called_once()
    assert bars[0].set_postfix_str.call_args.args[0] == (
        "wait timed out" if failure == "timeout" else "wait interrupted"
    )
    assert all(r.method == "GET" for r in api.requests)


async def test_real_tqdm_writes_to_stderr_only(
    api: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    await call(api.client.batch_decisions.wait, JOB_ID, progress=True)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Batch decisions" in captured.err
    assert "completed" in captured.err
    assert captured.err.endswith("\n")
