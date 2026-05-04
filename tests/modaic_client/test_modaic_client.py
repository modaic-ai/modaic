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
        raise TimeoutError(
            f"Example {ingest_response.example_ids[-1]} not available after 90s"
        )
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
    def test_sends_full_example_schema(self):
        # predict_all always waits now, so we have to satisfy the
        # POST + /events + /results triplet rather than just the POST.
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
                        "status": "queued",
                        "arbiters": ["user/repo"],
                        "total": 2,
                    },
                )
            if path.endswith("/events"):
                # Single-frame ``done`` so wait() returns immediately.
                body = (
                    "event: done\n"
                    'data: {"kind":"done","status":"success"}\n\n'
                ).encode()
                return httpx.Response(
                    200, content=body, headers={"content-type": "text/event-stream"}
                )
            if path.endswith("/results"):
                return httpx.Response(
                    200, content=b"", headers={"content-type": "application/x-ndjson"}
                )
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
            "split": "none",
        }

    def test_validates_example_constraints(self):
        c = _make_mock_client(lambda r: httpx.Response(200))
        arbiter = c.get_arbiter("user/repo")
        with pytest.raises(ValueError, match="at least one example"):
            c.predict_all(examples=[], arbiters=[arbiter])
        with pytest.raises(ValueError, match="at least one arbiter"):
            c.predict_all(examples=[{"input": {"q": "x"}}], arbiters=[])
        with pytest.raises(ValueError, match="missing required 'input'"):
            c.predict_all(examples=[{"ground_truth": {"verdict": "y"}}], arbiters=[arbiter])


class TestBatchJobStreamingUnit:
    """Unit coverage for BatchJob.events() and the SSE-driven path inside
    BatchJob.wait(). SSE responses are constructed via httpx.MockTransport —
    the client's iter_lines() works on any byte payload, so we don't need a
    real network socket to exercise the parser, the kind dispatch, or the
    polling fallback."""

    @staticmethod
    def _sse_payload(*events: dict, include_heartbeat: bool = False) -> bytes:
        """Encode a sequence of event dicts as SSE wire format."""
        chunks: list[str] = []
        if include_heartbeat:
            chunks.append(": heartbeat\n\n")
        for evt in events:
            chunks.append(
                f"event: {evt.get('kind', 'message')}\n"
                f"data: {json.dumps(evt)}\n\n"
            )
        return "".join(chunks).encode("utf-8")

    def _events_handler(self, *events: dict, include_heartbeat: bool = False):
        body = self._sse_payload(*events, include_heartbeat=include_heartbeat)

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
                {
                    "kind": "started",
                    "job_id": "j1",
                    "total": 4,
                    "arbiters": ["a/b"],
                    "ts": 1.0,
                },
                {
                    "kind": "prediction_completed",
                    "job_id": "j1",
                    "example_index": 0,
                    "arbiter_index": 0,
                    "arbiter_repo": "a/b",
                    "latency_ms": 123,
                    "ts": 1.1,
                },
                {"kind": "done", "job_id": "j1", "status": "success", "ts": 2.0},
                include_heartbeat=True,
            )
        )

        job = BatchJob(client=c, job_id="j1", total=4, arbiters=["a/b"])
        evts = list(job.events())

        # Heartbeat (`: heartbeat`) is silently dropped by the parser.
        assert [e.kind for e in evts] == [
            "started",
            "prediction_completed",
            "done",
        ]
        assert evts[0].total == 4
        assert evts[0].arbiters == ["a/b"]
        assert evts[1].latency_ms == 123
        assert evts[1].arbiter_repo == "a/b"
        assert evts[2].status == "success"

    def test_events_terminates_on_done_even_with_trailing_events(self):
        c = _make_mock_client(
            self._events_handler(
                {"kind": "done", "job_id": "j1", "status": "success"},
                {"kind": "prediction_completed", "example_index": 99},  # never seen
            )
        )
        job = BatchJob(client=c, job_id="j1", total=1, arbiters=[])
        kinds = [e.kind for e in job.events()]
        assert kinds == ["done"]

    def test_events_404_raises_streaming_not_available(self):
        from modaic_client.client import _StreamingNotAvailable

        def handler(request):
            return httpx.Response(404, json={"detail": "no /events on this build"})

        c = _make_mock_client(handler)
        job = BatchJob(client=c, job_id="j1", total=0, arbiters=[])
        with pytest.raises(_StreamingNotAvailable):
            list(job.events())

    def test_wait_invokes_on_event_and_fetches_results(self):
        # Two endpoints in play: SSE /events to drive the wait(), and
        # /results to satisfy the post-success fetch.
        events_body = self._sse_payload(
            {"kind": "started", "total": 1, "arbiters": ["a/b"]},
            {
                "kind": "prediction_completed",
                "example_index": 0,
                "arbiter_index": 0,
                "arbiter_repo": "a/b",
                "latency_ms": 5,
            },
            {"kind": "done", "status": "success"},
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

        seen_kinds: list[str] = []
        out = job.wait(
            show_progress=False,
            on_event=lambda e: seen_kinds.append(e.kind),
        )

        assert seen_kinds == ["started", "prediction_completed", "done"]
        assert isinstance(out, list)
        assert len(out) == 1
        assert isinstance(out[0], BatchExampleResult)
        assert out[0].example_id == "ex-1"
        assert out[0].predictions[0].arbiter_repo == "a/b"

    def test_wait_raises_on_done_failure(self):
        events_body = self._sse_payload(
            {"kind": "done", "status": "failure", "error": "boom"},
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
        with pytest.raises(RuntimeError, match="status=failure: boom"):
            job.wait(show_progress=False)

    def test_show_progress_advances_tqdm_bar(self, monkeypatch):
        """When ``show_progress=True``, each prediction_* event bumps the
        bar and ``done`` closes it. We swap in a fake bar object and assert
        the call sequence rather than depending on tqdm's terminal output."""
        from modaic_client import client as cm

        bars: list[MagicMock] = []

        def fake_bar(total):
            bar = MagicMock(spec=["update", "close", "n"])
            bar.n = 0
            bars.append(bar)
            return bar

        monkeypatch.setattr(cm, "_make_progress_bar", fake_bar)

        events_body = self._sse_payload(
            {"kind": "started", "total": 2, "arbiters": ["a/b"]},
            {"kind": "prediction_completed", "example_index": 0, "arbiter_index": 0},
            {"kind": "prediction_failed", "example_index": 1, "arbiter_index": 0,
             "error": "x"},
            {"kind": "done", "status": "success"},
        )

        def handler(request):
            if request.url.path.endswith("/events"):
                return httpx.Response(
                    200,
                    content=events_body,
                    headers={"content-type": "text/event-stream"},
                )
            if request.url.path.endswith("/results"):
                return httpx.Response(
                    200, content=b"", headers={"content-type": "application/x-ndjson"}
                )
            return httpx.Response(404)

        c = _make_mock_client(handler)
        job = BatchJob(client=c, job_id="j1", total=2, arbiters=["a/b"])
        job.wait(show_progress=True)

        assert len(bars) == 1
        bar = bars[0]
        # One ``update(1)`` per prediction_{completed,failed}; ``started``
        # and ``done`` don't tick the bar.
        update_calls = [c.args for c in bar.update.call_args_list]
        assert update_calls == [(1,), (1,)]
        bar.close.assert_called_once()

    def test_wait_falls_back_to_polling_on_404(self, monkeypatch):
        """A 404 on /events drops wait() into the polling branch (status +
        results) so older servers / unrouted endpoints still work."""

        def handler(request):
            path = request.url.path
            if path.endswith("/events"):
                return httpx.Response(404, json={"detail": "no events"})
            if path.endswith("/jobs/batch/predictions/j1"):
                return httpx.Response(200, json={"job_id": "j1", "status": "SUCCESS"})
            if path.endswith("/results"):
                return httpx.Response(
                    200, content=b"", headers={"content-type": "application/x-ndjson"}
                )
            return httpx.Response(404)

        monkeypatch.setattr(time, "sleep", lambda _s: None)

        c = _make_mock_client(handler)
        job = BatchJob(client=c, job_id="j1", total=0, arbiters=[])
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
