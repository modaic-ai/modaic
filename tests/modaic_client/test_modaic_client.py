import json
import os

import httpx
import pytest
from modaic_client.client import (
    Arbiter,
    ModaicClient,
    configure_modaic_client,
    get_modaic_client,
)
from modaic_client.exceptions import AuthenticationError, RepositoryExistsError
from modaic_client.schemas import (
    AnnotateExampleResponse,
    ArbiterPredictResponse,
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
        output=FieldSchema(name="answer", type="string"),
        instructions="Answer the question concisely.",
        model="together_ai/openai/gpt-oss-120b",
    )


@pytest.fixture(scope="module")
def ingested_example_ids(client, arbiter):
    resp = client.ingest_examples(
        [
            {"arbiter_repo": arbiter.repo, "input": {"question": "What is 1+1?"}, "ground_truth": "2"},
            {"arbiter_repo": arbiter.repo, "input": {"question": "What is 2+2?"}, "ground_truth": "4"},
        ]
    )
    return resp.example_ids


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
            ingested_example_ids[0],
            [{"arbiter_repo": arbiter.repo, "ground_truth": "2", "ground_reasoning": "simple math"}],
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

    def test_predict_all_returns_response(self, client, arbiter):
        resp = client.predict_all({"question": "What is 1+1?"}, [arbiter])
        assert isinstance(resp, ArbiterPredictResponse)
        assert isinstance(resp.example_id, str)
        assert len(resp.predictions) == 1


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
    def test_sends_ground_data_in_arbiters(self):
        captured = {}

        def handler(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "example_id": "ex-001",
                    "predictions": [
                        {
                            "arbiter_repo": "user/repo",
                            "commit_hash": "abc",
                            "output": {"output": "A"},
                            "prediction_id": "pred-001",
                            "reasoning": "r",
                            "messages": [],
                        }
                    ],
                },
            )

        c = _make_mock_client(handler)
        arbiter = c.get_arbiter("user/repo")
        c.predict_all(
            {"question": "hi"},
            [arbiter],
            ground_data=[{"ground_truth": "yes", "ground_reasoning": "obvious"}],
        )

        arb_data = captured["body"]["arbiters"][0]
        assert arb_data["ground_truth"] == "yes"
        assert arb_data["ground_reasoning"] == "obvious"


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
