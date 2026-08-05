# ruff: noqa: ANN001, ANN201
import json
import os
import time
from unittest.mock import MagicMock

import httpx
import pytest
from modaic_client.client import (
    Arbiter,
    BatchExampleResult,
    BatchJob,
    ModaicClient,
    configure_modaic_client,
    get_modaic_client,
)
from modaic_client.exceptions import AuthenticationError, RepositoryExistsError
from modaic_client.schemas import (
    AnnotateExampleResponse,
    ExamplesPage,
    FieldSchema,
    IngestExamplesResponse,
    PredictedExample,
)
from pydantic import ValidationError

from tests.utils import delete_program_repo

# ── Mock helpers (for unit tests) ────────────────────────────────────


def _make_mock_client(handler):
    transport = httpx.MockTransport(handler)
    mock_httpx = httpx.Client(transport=transport, base_url="http://test")
    return ModaicClient(modaic_token="test-token", base_url="http://test", client=mock_httpx)


# ── Integration test setup ───────────────────────────────────────────

MODAIC_TOKEN = os.getenv("MODAIC_TOKEN")
USERNAME = None
TEST_PROGRAM = "modaic-client-test"

if MODAIC_TOKEN:
    try:
        USERNAME = ModaicClient().get_user_info()["login"]
    except Exception:
        pass

requires_token = pytest.mark.skipif(
    not MODAIC_TOKEN or not USERNAME, reason="MODAIC_TOKEN not set or user info unavailable"
)


# ── Integration fixtures (module-scoped) ─────────────────────────────


@pytest.fixture(scope="module")
def client():
    return ModaicClient()


@pytest.fixture(scope="module")
def test_repo(client):
    repo = f"{USERNAME}/{TEST_PROGRAM}"
    delete_program_repo(username=USERNAME, program_name=TEST_PROGRAM, ignore_errors=True)
    created = client.create_repo(repo)
    assert created is True
    yield repo
    delete_program_repo(username=USERNAME, program_name=TEST_PROGRAM, ignore_errors=True)


@pytest.fixture(scope="module")
def arbiter(client, test_repo):
    return client.create_arbiter(
        test_repo,
        inputs=[FieldSchema(name="question", type="string")],
        outputs=[FieldSchema(name="answer", type="string")],
        instructions="Answer the question concisely.",
        model="together_ai/openai/gpt-oss-120b",
    )


@pytest.fixture(scope="module")
def ingest_response(client, arbiter):
    return client.ingest_examples(
        [
            {
                "arbiter_repo": arbiter.repo,
                "input": {"question": "What is 1+1?"},
                "serialized_output": "2",
                "ground_truth": "2",
            },
            {
                "arbiter_repo": arbiter.repo,
                "input": {"question": "What is 2+2?"},
                "serialized_output": "4",
                "ground_truth": "4",
            },
        ]
    )


@pytest.fixture(scope="module")
def ingested_example_ids(client, ingest_response):
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            client.get_example(ingest_response.example_ids[-1])
            break
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                raise
            time.sleep(2)
    else:
        raise TimeoutError(f"Example {ingest_response.example_ids[-1]} not available after 90s")
    return ingest_response.example_ids


# ══════════════════════════════════════════════════════════════════════
#  INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════


@requires_token
class TestGetUserInfoIntegration:
    def test_returns_user_info(self, client):
        info = client.get_user_info()
        assert info["login"] == USERNAME
        assert "email" in info
        assert "avatar_url" in info
        assert "name" in info


@requires_token
class TestCreateDeleteRepoIntegration:
    def test_create_repo(self, test_repo):
        # test_repo fixture already asserts create_repo returned True
        assert test_repo == f"{USERNAME}/{TEST_PROGRAM}"

    def test_create_repo_exist_ok_returns_false(self, client, test_repo):
        assert client.create_repo(test_repo, exist_ok=True) is False

    def test_create_repo_exist_ok_false_raises(self, client, test_repo):
        with pytest.raises(RepositoryExistsError):
            client.create_repo(test_repo, exist_ok=False)

    def test_delete_and_recreate_repo(self, client):
        repo = f"{USERNAME}/{TEST_PROGRAM}-tmp"
        delete_program_repo(username=USERNAME, program_name=f"{TEST_PROGRAM}-tmp", ignore_errors=True)
        try:
            assert client.create_repo(repo) is True
            assert client.delete_repo(repo) is True
        finally:
            delete_program_repo(username=USERNAME, program_name=f"{TEST_PROGRAM}-tmp", ignore_errors=True)


@requires_token
class TestCreateArbiterIntegration:
    def test_returns_arbiter(self, arbiter):
        assert isinstance(arbiter, Arbiter)

    def test_arbiter_has_correct_repo(self, arbiter, test_repo):
        assert arbiter.repo == test_repo


@requires_token
class TestIngestExamplesIntegration:
    def test_response_shape(self, ingest_response):
        assert isinstance(ingest_response, IngestExamplesResponse)
        assert ingest_response.queued is True
        assert len(ingest_response.example_ids) == 2
        assert all(isinstance(eid, str) and len(eid) > 0 for eid in ingest_response.example_ids)

    def test_returns_example_ids(self, ingested_example_ids):
        assert len(ingested_example_ids) == 2
        assert all(isinstance(eid, str) for eid in ingested_example_ids)


@requires_token
class TestListExamplesIntegration:
    def test_returns_examples_page(self, client, arbiter, ingested_example_ids):
        page = client.list_examples(user=arbiter._repo_user, program=arbiter._repo_name)
        assert isinstance(page, ExamplesPage)
        assert page.total >= 1
        assert len(page.items) >= 1

    def test_pagination(self, client, arbiter, ingested_example_ids):
        page = client.list_examples(user=arbiter._repo_user, program=arbiter._repo_name, page=1, page_size=1)
        assert page.page_size == 1
        assert len(page.items) == 1
        assert page.total >= 2


@requires_token
class TestGetExampleIntegration:
    def test_returns_predicted_example(self, client, ingested_example_ids):
        ex = client.get_example(ingested_example_ids[0])
        assert isinstance(ex, PredictedExample)
        assert ex.id == ingested_example_ids[0]


@requires_token
class TestAnnotateExampleIntegration:
    def test_annotate_returns_success(self, client, arbiter, ingested_example_ids):
        # v2: dict-based ground_truth (output field name -> value).
        resp = client.annotate_example(
            ingested_example_ids[1],
            [{"arbiter_repo": arbiter.repo, "ground_truth": {"answer": "4"}, "ground_reasoning": "simple math"}],
        )
        assert isinstance(resp, AnnotateExampleResponse)

    def test_annotate_string_ground_truth_is_deprecated(self, client, arbiter, ingested_example_ids):
        # Legacy string ground_truth still works via the deprecated v1 endpoint.
        with pytest.warns(DeprecationWarning):
            resp = client.annotate_example(
                ingested_example_ids[1],
                [{"arbiter_repo": arbiter.repo, "ground_truth": "4", "ground_reasoning": "simple math"}],
            )
        assert isinstance(resp, AnnotateExampleResponse)


@requires_token
@pytest.mark.slow
class TestPredictIntegration:
    def test_predict_returns_prediction(self, client, arbiter):
        prediction = client.predict({"question": "What is 1+1?"}, arbiter)
        assert isinstance(prediction.example_id, str)
        assert prediction.arbiter_repo == arbiter.repo
        assert prediction.output is not None

    def test_predict_all_returns_results(self, client, arbiter):
        results = client.predict_all(
            examples=[{"input": {"question": "What is 1+1?"}}],
            arbiters=[arbiter],
            wait_for_results=True,
            poll_interval=10.0,
        )
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], BatchExampleResult)
        assert len(results[0].predictions) == 1


# ══════════════════════════════════════════════════════════════════════
#  UNIT TESTS (no token required, mocked HTTP)
# ══════════════════════════════════════════════════════════════════════


class TestModaicClientInit:
    def test_init_with_explicit_params(self):
        c = ModaicClient(modaic_token="tok", base_url="http://x")
        assert c.modaic_token == "tok"
        assert c.base_url == "http://x"

    def test_init_defaults_from_settings(self, monkeypatch):
        monkeypatch.setattr("modaic_client.client.settings.modaic_token", "env-tok")
        monkeypatch.setattr("modaic_client.client.settings.modaic_api_url", "http://env")
        c = ModaicClient()
        assert c.modaic_token == "env-tok"
        assert c.base_url == "http://env"

    def test_init_accepts_injected_client(self):
        mock = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
        c = ModaicClient(client=mock)
        assert c._client is mock

    def test_init_timeout_default(self):
        c = ModaicClient()
        assert c._timeout == 30.0

    def test_init_custom_timeout(self):
        c = ModaicClient(timeout=60.0)
        assert c._timeout == 60.0


class TestGetClient:
    def test_uses_injected_client(self):
        mock = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
        c = ModaicClient(modaic_token="tok", client=mock)
        with c.get_client() as cl:
            assert cl is mock

    def test_sets_auth_header(self):
        mock = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
        c = ModaicClient(modaic_token="tok", client=mock)
        with c.get_client() as cl:
            assert cl.headers["Authorization"] == "Bearer tok"

    def test_restores_auth_header(self):
        mock = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
        c = ModaicClient(modaic_token="tok", client=mock)
        with c.get_client():
            pass
        assert "Authorization" not in mock.headers

    def test_access_token_override(self):
        mock = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
        c = ModaicClient(modaic_token="tok", client=mock)
        with c.get_client(access_token="override") as cl:
            assert cl.headers["Authorization"] == "Bearer override"


class TestGetArbiterUnit:
    def test_returns_arbiter_with_correct_repo(self):
        c = _make_mock_client(lambda r: httpx.Response(200))
        a = c.get_arbiter("user/repo")
        assert isinstance(a, Arbiter)
        assert a.repo == "user/repo"

    def test_sets_client_on_arbiter(self):
        c = _make_mock_client(lambda r: httpx.Response(200))
        a = c.get_arbiter("user/repo")
        assert a.client is c

    def test_custom_revision(self):
        c = _make_mock_client(lambda r: httpx.Response(200))
        a = c.get_arbiter("user/repo", revision="v2")
        assert a.revision == "v2"


class TestPredictAllUnit:
    @staticmethod
    def _finish_sse_body(job_id: str = "job-123", total: int = 2) -> bytes:
        return (
            "event: finish\n"
            f'data: {{"event":"finish","status":"done",'
            f'"predictions_progress":{{"current":{total},"total":{total},'
            f'"completed":{total},"failed":0}},"scores_progress":null,'
            f'"results":{{"total":{total},"examples":{total},"arbiters":1}},'
            f'"error":null,"job_id":"{job_id}","ts":1.0}}\n\n'
        ).encode()

    def test_sends_full_example_schema(self):
        captured = {}

        def handler(request):
            path = request.url.path
            if path.endswith("/jobs/batch/predictions") and request.method == "POST":
                captured["url"] = str(request.url)
                captured["body"] = json.loads(request.content)
                return httpx.Response(
                    200,
                    json={
                        "job_id": "job-123",
                        "event": "start",
                        "status": "predicting",
                        "arbiters": ["user/repo"],
                        "total": 2,
                    },
                )
            if path.endswith("/events"):
                return httpx.Response(
                    200,
                    content=self._finish_sse_body("job-123", 2),
                    headers={"content-type": "text/event-stream"},
                )
            if path.endswith("/results"):
                return httpx.Response(200, content=b"", headers={"content-type": "application/x-ndjson"})
            return httpx.Response(404)

        c = _make_mock_client(handler)
        arbiter = c.get_arbiter("user/repo")
        out = c.predict_all(
            examples=[
                {
                    "input": {"question": "hi"},
                    "alt_id": "ex-1",
                    "ground_truth": {"verdict": "yes"},
                    "ground_reasoning": "obvious",
                    "split": "train",
                },
                {"input": {"question": "bye"}},
            ],
            arbiters=[arbiter],
            show_progress=False,
        )

        assert out == []  # empty results body → empty list
        assert "/api/v1/jobs/batch/predictions" in captured["url"]
        assert captured["body"]["arbiters"] == [{"arbiter_repo": "user/repo", "arbiter_revision": "main"}]
        assert captured["body"]["compute_confidence"] is False
        examples = captured["body"]["examples"]
        assert examples[0] == {
            "input": {"question": "hi"},
            "alt_id": "ex-1",
            "ground_truth": {"verdict": "yes"},
            "ground_reasoning": "obvious",
            "split": "train",
        }
        assert examples[1] == {
            "input": {"question": "bye"},
            "alt_id": None,
            "ground_truth": None,
            "ground_reasoning": "",
        }

    def test_sends_example_ids_payload(self):
        captured = {}

        def handler(request):
            path = request.url.path
            if path.endswith("/jobs/batch/predictions") and request.method == "POST":
                captured["body"] = json.loads(request.content)
                return httpx.Response(
                    200,
                    json={
                        "job_id": "job-456",
                        "event": "start",
                        "status": "predicting",
                        "arbiters": ["user/repo"],
                        "total": 2,
                    },
                )
            if path.endswith("/events"):
                return httpx.Response(
                    200,
                    content=self._finish_sse_body("job-456", 2),
                    headers={"content-type": "text/event-stream"},
                )
            if path.endswith("/results"):
                return httpx.Response(200, content=b"", headers={"content-type": "application/x-ndjson"})
            return httpx.Response(404)

        c = _make_mock_client(handler)
        arbiter = c.get_arbiter("user/repo")
        c.predict_all(
            example_ids=["ex-A", "ex-B"],
            arbiters=[arbiter],
            compute_confidence=True,
            show_progress=False,
        )
        assert captured["body"]["example_ids"] == ["ex-A", "ex-B"]
        assert "examples" not in captured["body"]
        assert captured["body"]["compute_confidence"] is True

    def test_wait_for_none_returns_handle_immediately(self):
        captured = {"events_called": 0, "results_called": 0}

        def handler(request):
            path = request.url.path
            if path.endswith("/jobs/batch/predictions") and request.method == "POST":
                return httpx.Response(
                    200,
                    json={
                        "job_id": "job-1",
                        "event": "start",
                        "status": "predicting",
                        "arbiters": ["user/repo"],
                        "total": 1,
                    },
                )
            if path.endswith("/events"):
                captured["events_called"] += 1
            if path.endswith("/results"):
                captured["results_called"] += 1
            return httpx.Response(404)

        c = _make_mock_client(handler)
        arbiter = c.get_arbiter("user/repo")
        out = c.predict_all(
            examples=[{"input": {"q": "hi"}}],
            arbiters=[arbiter],
            wait_for=None,
        )
        # No /events or /results call happened — we got a handle, not results.
        assert captured["events_called"] == 0
        assert captured["results_called"] == 0
        assert isinstance(out, BatchJob)
        assert out.job_id == "job-1"

    def test_validates_example_constraints(self):
        c = _make_mock_client(lambda r: httpx.Response(200))
        arbiter = c.get_arbiter("user/repo")
        with pytest.raises(ValueError, match="exactly one of `examples` or `example_ids`"):
            c.predict_all(arbiters=[arbiter])
        with pytest.raises(ValueError, match="exactly one of `examples` or `example_ids`"):
            c.predict_all(
                examples=[{"input": {"q": "x"}}],
                example_ids=["ex-1"],
                arbiters=[arbiter],
            )
        with pytest.raises(ValueError, match="at least one arbiter"):
            c.predict_all(examples=[{"input": {"q": "x"}}], arbiters=[])
        with pytest.raises(ValueError, match="missing required 'input'"):
            c.predict_all(examples=[{"ground_truth": {"verdict": "y"}}], arbiters=[arbiter])

    def test_wait_for_scores_requires_compute_confidence(self):
        c = _make_mock_client(lambda r: httpx.Response(200))
        arbiter = c.get_arbiter("user/repo")
        with pytest.raises(ValueError, match="wait_for='scores' requires compute_confidence=True"):
            c.predict_all(
                examples=[{"input": {"q": "x"}}],
                arbiters=[arbiter],
                wait_for="scores",
            )


class TestBatchJobStreamingUnit:
    """Unit coverage for BatchJob.events() and the SSE-driven path inside
    BatchJob.wait(). SSE responses are constructed via httpx.MockTransport —
    the client's iter_lines() works on any byte payload, so we don't need a
    real network socket to exercise the parser, the event/status dispatch,
    or the polling fallback."""

    @staticmethod
    def _snapshot(
        *,
        event: str,
        status: str,
        predictions_current: int = 0,
        predictions_total: int = 0,
        scores_current: int | None = None,
        scores_total: int | None = None,
        results: dict | None = None,
        error: str | None = None,
        job_id: str = "j1",
    ) -> dict:
        """Build a `BatchPredictionsEvent`-shaped dict for SSE bodies."""
        return {
            "event": event,
            "status": status,
            "job_id": job_id,
            "ts": 1.0,
            "predictions_progress": {
                "current": predictions_current,
                "total": predictions_total,
                "completed": predictions_current,
                "failed": 0,
            },
            "scores_progress": (
                None
                if scores_current is None
                else {
                    "current": scores_current,
                    "total": scores_total or 0,
                    "completed": scores_current,
                    "failed": 0,
                }
            ),
            "results": results,
            "error": error,
        }

    @classmethod
    def _sse_payload(cls, *snapshots: dict, include_heartbeat: bool = False) -> bytes:
        """Encode a sequence of snapshot dicts as SSE wire format."""
        chunks: list[str] = []
        if include_heartbeat:
            chunks.append(": heartbeat\n\n")
        for snap in snapshots:
            chunks.append(f"event: {snap.get('event', 'message')}\ndata: {json.dumps(snap)}\n\n")
        return "".join(chunks).encode("utf-8")

    def _events_handler(self, *snapshots: dict, include_heartbeat: bool = False):
        body = self._sse_payload(*snapshots, include_heartbeat=include_heartbeat)

        def handler(request):
            assert request.url.path.endswith("/events")
            return httpx.Response(
                200,
                content=body,
                headers={"content-type": "text/event-stream"},
            )

        return handler

    def test_events_yields_parsed_progress_events(self):
        c = _make_mock_client(
            self._events_handler(
                self._snapshot(event="start", status="predicting", predictions_total=4),
                self._snapshot(
                    event="prediction",
                    status="predicting",
                    predictions_current=1,
                    predictions_total=4,
                ),
                self._snapshot(
                    event="finish",
                    status="done",
                    predictions_current=4,
                    predictions_total=4,
                    results={"total": 4, "examples": 2, "arbiters": 2},
                ),
                include_heartbeat=True,
            )
        )

        job = BatchJob(client=c, job_id="j1", total=4, arbiters=["a/b"])
        evts = list(job.events())

        # Heartbeat (`: heartbeat`) is silently dropped by the parser.
        assert [e.event for e in evts] == ["start", "prediction", "finish"]
        assert evts[0].predictions_progress.total == 4
        assert evts[1].predictions_progress.current == 1
        assert evts[2].status == "done"
        assert evts[2].results.total == 4

    def test_events_terminates_on_finish_even_with_trailing_events(self):
        c = _make_mock_client(
            self._events_handler(
                self._snapshot(event="finish", status="done"),
                # Never reached because finish terminates the iterator.
                self._snapshot(
                    event="prediction",
                    status="predicting",
                    predictions_current=99,
                    predictions_total=99,
                ),
            )
        )
        job = BatchJob(client=c, job_id="j1", total=1, arbiters=[])
        events = [e.event for e in job.events()]
        assert events == ["finish"]

    def test_events_404_raises_streaming_not_available(self):
        from modaic_client.client import _StreamingNotAvailable

        def handler(request):
            return httpx.Response(404, json={"detail": "no /events on this build"})

        c = _make_mock_client(handler)
        job = BatchJob(client=c, job_id="j1", total=0, arbiters=[])
        with pytest.raises(_StreamingNotAvailable):
            list(job.events())

    def test_wait_invokes_on_event_and_fetches_results(self):
        events_body = self._sse_payload(
            self._snapshot(event="start", status="predicting", predictions_total=1),
            self._snapshot(
                event="prediction",
                status="predicting",
                predictions_current=1,
                predictions_total=1,
            ),
            self._snapshot(
                event="finish",
                status="done",
                predictions_current=1,
                predictions_total=1,
            ),
        )
        results_body = (
            json.dumps(
                {
                    "example_id": "ex-1",
                    "input": {"q": "hi"},
                    "predictions": [
                        {
                            "prediction_id": "p-1",
                            "output": {"answer": "ok"},
                            "reasoning": "",
                            "messages": [{"role": "user", "content": "hi"}],
                            "confidence": 0.7,
                        }
                    ],
                }
            )
            + "\n"
        ).encode("utf-8")

        def handler(request):
            if request.url.path.endswith("/events"):
                return httpx.Response(
                    200,
                    content=events_body,
                    headers={"content-type": "text/event-stream"},
                )
            if request.url.path.endswith("/results"):
                return httpx.Response(
                    200,
                    content=results_body,
                    headers={"content-type": "application/x-ndjson"},
                )
            return httpx.Response(404)

        c = _make_mock_client(handler)
        job = BatchJob(client=c, job_id="j1", total=1, arbiters=["a/b"])

        seen_events: list[str] = []
        out = job.wait(
            show_progress=False,
            on_event=lambda e: seen_events.append(e.event),
        )

        assert seen_events[0] == "start"
        assert seen_events[-1] == "finish"
        assert isinstance(out, list)
        assert len(out) == 1
        assert isinstance(out[0], BatchExampleResult)
        assert out[0].example_id == "ex-1"
        assert out[0].predictions[0].arbiter_repo == "a/b"
        # Confidence flows through and pre-populates the prediction so
        # `.confidence` doesn't block on the wait_for_confidence_score path.
        assert out[0].predictions[0]._confidence == 0.7

    def test_results_splits_only_on_newline_not_unicode_line_separators(self):
        """Regression: ``httpx.Response.iter_lines()`` uses ``str.splitlines()``
        under the hood, which also breaks on ``U+2028`` and ``U+2029``.
        Pydantic's ``model_dump_json`` leaves those characters raw, so a single
        NDJSON record containing scraped prose (e.g. LitBench rationales) was
        being split mid-string and raised ``JSONDecodeError``."""
        body = (
            json.dumps(
                {
                    "example_id": "ex-1",
                    "input": {"story": "line one line two line three"},
                    "predictions": [
                        {
                            "prediction_id": "p-1",
                            "output": {"answer": "ok"},
                            "reasoning": "rationale with a separator inside",
                            "messages": [],
                            "confidence": None,
                        }
                    ],
                },
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")

        def handler(request):
            assert request.url.path.endswith("/results")
            return httpx.Response(
                200,
                content=body,
                headers={"content-type": "application/x-ndjson"},
            )

        c = _make_mock_client(handler)
        job = BatchJob(client=c, job_id="j1", total=1, arbiters=["a/b"])

        out = job.results()
        assert len(out) == 1
        assert out[0].example_id == "ex-1"
        assert out[0].input["story"] == "line one line two line three"
        assert out[0].predictions[0].reasoning == "rationale with a separator inside"

    def test_wait_raises_on_failed_status(self):
        events_body = self._sse_payload(
            self._snapshot(event="finish", status="failed", error="boom"),
        )

        def handler(request):
            assert request.url.path.endswith("/events")
            return httpx.Response(
                200,
                content=events_body,
                headers={"content-type": "text/event-stream"},
            )

        c = _make_mock_client(handler)
        job = BatchJob(client=c, job_id="j1", total=1, arbiters=[])
        with pytest.raises(RuntimeError, match="boom"):
            job.wait(show_progress=False)

    def test_show_progress_advances_tqdm_bar(self, monkeypatch):
        """When ``show_progress=True``, the bar is driven by
        ``predictions_progress`` counters (no per-event ticks). We swap in a
        fake bar and assert it lands at total."""
        from modaic_client import client as cm

        bars: list[MagicMock] = []

        def fake_bar(total):
            bar = MagicMock(spec=["update", "close", "n", "total"])
            bar.n = 0
            bar.total = total

            def _update(delta):
                bar.n += delta

            bar.update.side_effect = _update
            bars.append(bar)
            return bar

        monkeypatch.setattr(cm, "_make_progress_bar", fake_bar)

        events_body = self._sse_payload(
            self._snapshot(event="start", status="predicting", predictions_total=2),
            self._snapshot(
                event="prediction",
                status="predicting",
                predictions_current=1,
                predictions_total=2,
            ),
            self._snapshot(
                event="prediction",
                status="predicting",
                predictions_current=2,
                predictions_total=2,
            ),
            self._snapshot(
                event="finish",
                status="done",
                predictions_current=2,
                predictions_total=2,
            ),
        )

        def handler(request):
            if request.url.path.endswith("/events"):
                return httpx.Response(
                    200,
                    content=events_body,
                    headers={"content-type": "text/event-stream"},
                )
            if request.url.path.endswith("/results"):
                return httpx.Response(200, content=b"", headers={"content-type": "application/x-ndjson"})
            return httpx.Response(404)

        c = _make_mock_client(handler)
        job = BatchJob(client=c, job_id="j1", total=2, arbiters=["a/b"])
        job.wait(show_progress=True)

        assert len(bars) == 1
        bar = bars[0]
        # Bar advances monotonically and lands at total.
        assert bar.n == 2
        bar.close.assert_called_once()

    def test_wait_falls_back_to_polling_on_404(self, monkeypatch):
        """A 404 on /events drops wait() into the polling branch (status +
        results) so older servers / unrouted endpoints still work."""

        def handler(request):
            path = request.url.path
            if path.endswith("/events"):
                return httpx.Response(404, json={"detail": "no events"})
            if path.endswith("/jobs/batch/predictions/j1"):
                return httpx.Response(
                    200,
                    json={
                        "job_id": "j1",
                        "event": "finish",
                        "status": "done",
                        "predictions_progress": {
                            "current": 1,
                            "total": 1,
                            "completed": 1,
                            "failed": 0,
                        },
                        "scores_progress": None,
                        "results": {"total": 1, "examples": 1, "arbiters": 1},
                        "error": None,
                        "ts": 1.0,
                    },
                )
            if path.endswith("/results"):
                return httpx.Response(200, content=b"", headers={"content-type": "application/x-ndjson"})
            return httpx.Response(404)

        monkeypatch.setattr(time, "sleep", lambda _s: None)

        c = _make_mock_client(handler)
        job = BatchJob(client=c, job_id="j1", total=1, arbiters=[])
        out = job.wait(show_progress=False, poll_interval=0.0, timeout=5.0)
        assert out == []


class TestCreateRepoErrors:
    def test_empty_path_raises_value_error(self):
        c = _make_mock_client(lambda r: httpx.Response(200))
        with pytest.raises(ValueError):
            c.create_repo("")

    def test_long_name_raises_value_error(self):
        c = _make_mock_client(lambda r: httpx.Response(200))
        with pytest.raises(ValueError, match="too long"):
            c.create_repo("user/" + "x" * 101)

    def test_raises_auth_error_on_401(self):
        c = _make_mock_client(lambda r: httpx.Response(401, json={}))
        with pytest.raises(AuthenticationError):
            c.create_repo("user/repo")

    def test_raises_auth_error_on_403(self):
        c = _make_mock_client(lambda r: httpx.Response(403, json={}))
        with pytest.raises(AuthenticationError):
            c.create_repo("user/repo")

    def test_raises_on_server_error(self):
        c = _make_mock_client(lambda r: httpx.Response(500, json={"message": "server error"}))
        with pytest.raises(Exception, match="Failed to create repository"):
            c.create_repo("user/repo")


class TestDeleteRepoErrors:
    def test_empty_path_raises_value_error(self):
        c = _make_mock_client(lambda r: httpx.Response(200))
        with pytest.raises(ValueError):
            c.delete_repo("")

    def test_raises_auth_error_on_401(self):
        c = _make_mock_client(lambda r: httpx.Response(401, json={}))
        with pytest.raises(AuthenticationError):
            c.delete_repo("user/repo")

    def test_raises_auth_error_on_403(self):
        c = _make_mock_client(lambda r: httpx.Response(403, json={}))
        with pytest.raises(AuthenticationError):
            c.delete_repo("user/repo")

    def test_raises_on_404(self):
        c = _make_mock_client(lambda r: httpx.Response(404, json={}))
        with pytest.raises(Exception, match="not found"):
            c.delete_repo("user/repo")


class TestGetUserInfoErrors:
    def test_raises_when_no_token(self, monkeypatch):
        monkeypatch.setattr("modaic_client.client.settings.modaic_token", None)
        transport = httpx.MockTransport(lambda r: httpx.Response(200))
        mock_httpx = httpx.Client(transport=transport, base_url="http://test")
        c = ModaicClient(modaic_token=None, client=mock_httpx)
        with pytest.raises(AuthenticationError, match="No access token"):
            c.get_user_info()

    def test_raises_auth_error_on_401(self, monkeypatch):
        monkeypatch.setattr("modaic_client.client.settings.modaic_git_url", "http://test")
        c = _make_mock_client(lambda r: httpx.Response(401))
        with pytest.raises(AuthenticationError):
            c.get_user_info()


class TestSingletonFunctions:
    @pytest.fixture(autouse=True)
    def _reset_singleton(self, monkeypatch):
        monkeypatch.setattr("modaic_client.client._modaic_client", None)
        yield
        monkeypatch.setattr("modaic_client.client._modaic_client", None)

    def test_get_modaic_client_creates_singleton(self, monkeypatch):
        monkeypatch.setattr("modaic_client.client.settings.modaic_token", "tok")
        first = get_modaic_client()
        second = get_modaic_client()
        assert first is second

    def test_configure_replaces_singleton(self):
        old = get_modaic_client()
        new = configure_modaic_client(modaic_token="new-tok")
        assert new is not old
        assert get_modaic_client() is new

    def test_configure_returns_new_client(self):
        c = configure_modaic_client(modaic_token="tok", timeout=99.0)
        assert isinstance(c, ModaicClient)
        assert c.modaic_token == "tok"
        assert c._timeout == 99.0


class TestFieldSchema:
    """Unit coverage for the create_arbiter FieldSchema request model."""

    def test_options_canonical(self):
        assert FieldSchema(name="q", type="string", options=["a", "b"]).options == ["a", "b"]

    def test_allowed_values_alias(self):
        # The legacy `allowed_values` key still populates `options`.
        assert FieldSchema(name="q", type="string", allowed_values=["a", "b"]).options == ["a", "b"]

    def test_range_stored(self):
        assert FieldSchema(name="r", type="number", range=[1, 5]).range == [1, 5]

    def test_range_requires_number_type(self):
        with pytest.raises(ValidationError):
            FieldSchema(name="x", type="string", range=[1, 5])

    def test_range_values_must_be_integers(self):
        with pytest.raises(ValidationError):
            FieldSchema(name="x", type="number", range=[1.3, 2.7])

    def test_range_must_have_two_elements(self):
        with pytest.raises(ValidationError):
            FieldSchema(name="x", type="number", range=[1, 2, 3])

    def test_range_lo_must_not_exceed_hi(self):
        with pytest.raises(ValidationError):
            FieldSchema(name="x", type="number", range=[5, 1])

    def test_options_and_range_mutually_exclusive(self):
        with pytest.raises(ValidationError):
            FieldSchema(name="x", type="number", options=[1, 2], range=[1, 5])

    def test_object_with_options_raises(self):
        with pytest.raises(ValidationError):
            FieldSchema(name="x", type="object", allowed_values=["x"])
