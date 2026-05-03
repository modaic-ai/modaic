import json
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, Literal, Optional, Tuple

import httpx
from pydantic import BaseModel, PrivateAttr
from typing_extensions import TypedDict

import modaic_client.exceptions as exceptions

from .config import settings
from .exceptions import AuthenticationError, RepositoryExistsError
from .schemas import (
    AnnotateExampleResponse,
    ConfidenceStatusResponse,
    ExamplesPage,
    FieldSchema,
    IngestExamplesResponse,
    InitArbiterRequest,
    Output,
    PredictedExample,
    PredictionAnnotation,
)

_modaic_client = None
_client_lock = threading.Lock()


def raise_errors(response: httpx.Response):
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        try:
            detail = e.response.json().get("detail", e.response.text)
        except json.decoder.JSONDecodeError:
            detail = e.response.text
        if (
            isinstance(detail, dict)
            and (modaic_error := getattr(exceptions, detail.get("modaic_error"), None))
            and (message := detail.get("message"))
        ):
            raise modaic_error(message) from e
        raise httpx.HTTPStatusError(str(detail), request=e.request, response=e.response) from None


class ArbiterPrediction(BaseModel):
    arbiter_repo: str
    commit_hash: Optional[str] = None
    output: Output
    reasoning: str
    messages: list[dict]
    example_id: Optional[str] = None
    prediction_id: Optional[str] = None
    _client: "ModaicClient" = PrivateAttr()
    _confidence: float | None = PrivateAttr(default=None)

    def request_confidence_score(
        self,
        access_token: Optional[str] = None,
    ) -> ConfidenceStatusResponse:
        if not self.prediction_id:
            raise ValueError("prediction_id is required to request a confidence score")
        return self._client.request_confidence_score(
            prediction_id=self.prediction_id,
            access_token=access_token,
        )

    def get_confidence_score(
        self,
        access_token: Optional[str] = None,
    ) -> ConfidenceStatusResponse:
        if not self.prediction_id:
            raise ValueError("prediction_id is required to read a confidence score")
        return self._client.get_confidence_score(
            prediction_id=self.prediction_id,
            access_token=access_token,
        )

    @property
    def confidence(self) -> float:
        if self._confidence is not None:
            return self._confidence
        if not self.prediction_id:
            raise ValueError("prediction_id is required to fetch confidence")
        result = self._client.wait_for_confidence_score(
            prediction_id=self.prediction_id,
        )
        if result.status != "completed" or result.score is None:
            raise RuntimeError(
                f"Confidence scoring did not complete: status={result.status} "
                f"error={result.error}"
            )
        self._confidence = result.score
        return result.score


class BatchExampleResult(BaseModel):
    """One example's results from a batch predictions job, with per-arbiter predictions."""

    example_id: str
    input: Optional[dict] = None
    predictions: list[ArbiterPrediction]


_BATCH_TERMINAL_STATES = {"SUCCESS", "FAILURE", "REVOKED"}


class BatchJob:
    """Handle for an in-flight batch predictions job."""

    def __init__(self, client: "ModaicClient", job_id: str, total: int, arbiters: Optional[list[str]] = None):
        self.client = client
        self.job_id = job_id
        self.total = total
        self.arbiters = arbiters or []

    def status(self) -> dict:
        with self.client.get_client() as http:
            response = http.get(
                f"/api/v1/jobs/batch/predictions/{self.job_id}",
                timeout=30.0,
            )
            raise_errors(response)
            return response.json()

    def results(self) -> list[BatchExampleResult]:
        with self.client.get_client() as http:
            with http.stream(
                "GET",
                f"/api/v1/jobs/batch/predictions/{self.job_id}/results",
                timeout=300.0,
            ) as response:
                raise_errors(response)
                rows: list[BatchExampleResult] = []
                for line in response.iter_lines():
                    if not line:
                        continue
                    payload = json.loads(line)
                    predictions = [
                        self.client._build_arbiter_prediction(
                            arbiter_repo=self._arbiter_for_index(i),
                            example_id=payload["example_id"],
                            prediction_id=p.get("prediction_id"),
                            output=p.get("output") or {},
                            reasoning=p.get("reasoning") or "",
                            messages=p.get("messages") or [],
                        )
                        for i, p in enumerate(payload.get("predictions", []))
                    ]
                    rows.append(
                        BatchExampleResult(
                            example_id=payload["example_id"],
                            input=payload.get("input"),
                            predictions=predictions,
                        )
                    )
                return rows

    def wait(self, poll_interval: float = 30.0, timeout: float = 3600.0) -> list[BatchExampleResult]:
        deadline = time.monotonic() + timeout
        while True:
            state = self.status()
            status = state.get("status")
            if status == "SUCCESS":
                return self.results()
            if status in {"FAILURE", "REVOKED"}:
                raise RuntimeError(f"Batch job {self.job_id} ended with status {status}: {state.get('error')}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Batch job {self.job_id} did not complete within {timeout}s (last status: {status})")
            time.sleep(poll_interval)

    def cancel(self) -> dict:
        with self.client.get_client() as http:
            response = http.delete(
                f"/api/v1/jobs/batch/predictions/{self.job_id}",
                timeout=30.0,
            )
            raise_errors(response)
            return response.json()

    def _arbiter_for_index(self, index: int) -> str:
        if 0 <= index < len(self.arbiters):
            return self.arbiters[index]
        return ""


class Arbiter:
    client: "ModaicClient"
    repo: str
    revision: str

    def __init__(self, repo: str, revision: str = "main"):
        self.repo = repo
        self.revision = revision
        self.client = get_modaic_client()

    @property
    def _repo_user(self) -> str:
        return self.repo.split("/")[0]

    @property
    def _repo_name(self) -> str:
        return self.repo.split("/")[1]

    def __call__(
        self,
        ground_truth: Optional[dict] = None,
        ground_reasoning: str = "",
        compute_confidence: bool = False,
        **inputs,
    ) -> ArbiterPrediction:
        return self.predict(
            ground_truth=ground_truth,
            ground_reasoning=ground_reasoning,
            compute_confidence=compute_confidence,
            **inputs,
        )

    def predict(
        self,
        ground_truth: Optional[dict] = None,
        ground_reasoning: str = "",
        compute_confidence: bool = False,
        **inputs,
    ) -> ArbiterPrediction:
        return self.client.predict(
            inputs, self, ground_truth, ground_reasoning, compute_confidence=compute_confidence
        )

    def predict_all(
        self,
        examples: "list[BatchExample]",
        *,
        wait_for_results: bool = True,
        poll_interval: float = 30.0,
        timeout: float = 3600.0,
    ) -> "list[BatchExampleResult] | BatchJob":
        return self.client.predict_all(
            examples,
            [self],
            wait_for_results=wait_for_results,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    def ingest_examples(self, examples: list[dict]) -> "IngestExamplesResponse":
        for ex in examples:
            ex.setdefault("arbiter_repo", self.repo)
        return self.client.ingest_examples(examples)

    def list_examples(
        self,
        page: int = 1,
        page_size: int = 50,
        version: Optional[int] = None,
        commit_hash: Optional[str] = None,
        search: Optional[str] = None,
    ) -> "ExamplesPage":
        return self.client.list_examples(
            user=self._repo_user,
            program=self._repo_name,
            page=page,
            page_size=page_size,
            version=version,
            commit_hash=commit_hash,
            search=search,
        )

    def get_example(self, example_id: str) -> "PredictedExample":
        return self.client.get_example(example_id)

    def annotate_example(
        self, example_id: str, ground_truth: Optional[str] = None, ground_reasoning: Optional[str] = None
    ) -> "AnnotateExampleResponse":
        annotation: PredictionAnnotation = {"arbiter_repo": self.repo}
        if ground_truth is not None:
            annotation["ground_truth"] = ground_truth
        if ground_reasoning is not None:
            annotation["ground_reasoning"] = ground_reasoning
        return self.client.annotate_example(example_id, [annotation])

    def set_client(self, client: "ModaicClient"):
        self.client = client

    def to_dict(self) -> dict:
        return {
            "arbiter_repo": self.repo,
            "arbiter_revision": self.revision,
        }


class BatchExample(TypedDict, total=False):
    """One row in a batch predictions request. ``input`` is required; the rest are optional."""

    input: dict
    alt_id: Optional[str]
    ground_truth: Optional[dict]
    ground_reasoning: str
    split: Literal["train", "test", "none"]


class ModaicClient:
    def __init__(
        self,
        modaic_token: Optional[str] = None,
        base_url: Optional[str] = None,
        *,
        client: Optional[httpx.Client] = None,
        timeout: float = 30.0,
    ):
        self.modaic_token = modaic_token or settings.modaic_token
        self.base_url = base_url or settings.modaic_api_url
        self._client = client
        self._timeout = timeout

    def _resolve_token(self, access_token: Optional[str] = None) -> str:
        return access_token if access_token is not None else self.modaic_token

    @contextmanager
    def get_client(self, access_token: Optional[str] = None) -> Iterator[httpx.Client]:
        token = self._resolve_token(access_token)

        # If we were given a client (TestClient or httpx.Client), reuse it.
        if self._client is not None:
            # set auth header for the duration of the context
            old = self._client.headers.get("Authorization")
            self._client.headers["Authorization"] = f"Bearer {token}"
            try:
                yield self._client
            finally:
                if old is None:
                    self._client.headers.pop("Authorization", None)
                else:
                    self._client.headers["Authorization"] = old
            return

        # Production/default path: real network client
        client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=self._timeout,
        )
        try:
            yield client
        finally:
            client.close()

    def get_arbiter(self, repo: str, revision: str = "main") -> Arbiter:
        arbiter = Arbiter(repo, revision)
        arbiter.set_client(self)
        return arbiter

    def create_arbiter(
        self,
        repo: str,
        inputs: list[FieldSchema],
        outputs: list[FieldSchema],
        instructions: Optional[str] = None,
        model: str = "qwen3-vl-32b-instruct",
        base_url: Optional[str] = None,
    ) -> Arbiter:
        request = InitArbiterRequest(
            repo=repo, inputs=inputs, outputs=outputs, instructions=instructions, model=model, base_url=base_url
        )
        with self.get_client() as client:
            response = client.post(
                "/api/v1/arbiters",
                json=request.model_dump(),
            )
            raise_errors(response)
        arbiter = Arbiter(repo)
        arbiter.set_client(self)
        return arbiter

    def predict_all(
        self,
        examples: list[BatchExample],
        arbiters: list[Arbiter],
        *,
        wait_for_results: bool = True,
        poll_interval: float = 30.0,
        timeout: float = 3600.0,
    ) -> list[BatchExampleResult] | BatchJob:
        if not examples:
            raise ValueError("predict_all requires at least one example")
        if not arbiters:
            raise ValueError("predict_all requires at least one arbiter")
        if len(arbiters) > 5:
            raise ValueError("predict_all accepts at most 5 arbiters per call")
        if len(examples) > 1000:
            raise ValueError("predict_all accepts at most 1000 examples per call")
        for i, ex in enumerate(examples):
            if "input" not in ex:
                raise ValueError(f"examples[{i}] is missing required 'input' key")

        examples_payload = [
            {
                "input": ex["input"],
                "alt_id": ex.get("alt_id"),
                "ground_truth": ex.get("ground_truth"),
                "ground_reasoning": ex.get("ground_reasoning", ""),
                "split": ex.get("split", "none"),
            }
            for ex in examples
        ]
        arbiters_payload = [arb.to_dict() for arb in arbiters]

        with self.get_client() as client:
            response = client.post(
                "/api/v1/jobs/batch/predictions",
                json={"arbiters": arbiters_payload, "examples": examples_payload},
                timeout=60.0,
            )
            raise_errors(response)
            data = response.json()

        job = BatchJob(
            client=self,
            job_id=data["job_id"],
            total=data.get("total", len(arbiters) * len(examples)),
            arbiters=[arb.repo for arb in arbiters],
        )
        if wait_for_results:
            return job.wait(poll_interval=poll_interval, timeout=timeout)
        return job

    def predict(
        self,
        input: dict,
        arbiter: Arbiter,
        ground_truth: Optional[dict] = None,
        ground_reasoning: str = "",
        compute_confidence: bool = False,
    ) -> ArbiterPrediction:
        with self.get_client() as client:
            response = client.post(
                "/api/v2/arbiters/predictions",
                json={
                    "input": input,
                    "arbiter_repo": arbiter.repo,
                    "arbiter_revision": arbiter.revision,
                    "ground_truth": ground_truth,
                    "ground_reasoning": ground_reasoning,
                    "compute_confidence": compute_confidence,
                },
                timeout=300.0,
            )
            raise_errors(response)
            data = response.json()

        return self._build_arbiter_prediction(
            arbiter_repo=arbiter.repo,
            example_id=data.get("example_id"),
            prediction_id=data.get("prediction_id"),
            output=data.get("output") or {},
            reasoning=data.get("reasoning") or "",
            messages=data.get("messages") or [],
        )

    def _build_arbiter_prediction(
        self,
        *,
        arbiter_repo: str,
        example_id: Optional[str],
        prediction_id: Optional[str],
        output: dict,
        reasoning: str,
        messages: list[dict],
        commit_hash: Optional[str] = None,
    ) -> ArbiterPrediction:
        prediction = ArbiterPrediction(
            arbiter_repo=arbiter_repo,
            commit_hash=commit_hash,
            output=Output.model_validate(output),
            reasoning=reasoning,
            messages=messages,
            example_id=example_id,
            prediction_id=prediction_id,
        )
        prediction._client = self
        return prediction

    def ingest_examples(self, examples: list[dict]) -> IngestExamplesResponse:
        body = "\n".join(json.dumps(ex) for ex in examples)
        with self.get_client() as client:
            response = client.post(
                "/api/v1/examples",
                content=body,
                headers={"Content-Type": "text/plain"},
            )
            raise_errors(response)
            return IngestExamplesResponse.model_validate(response.json())

    def list_examples(
        self,
        user: str,
        program: str,
        page: int = 1,
        page_size: int = 50,
        version: Optional[int] = None,
        commit_hash: Optional[str] = None,
        search: Optional[str] = None,
    ) -> ExamplesPage:
        params: dict[str, Any] = {"user": user, "program": program, "page": page, "page_size": page_size}
        if version is not None:
            params["version"] = version
        if commit_hash is not None:
            params["commit_hash"] = commit_hash
        if search is not None:
            params["search"] = search

        with self.get_client() as client:
            response = client.get("/api/v1/examples", params=params)
            raise_errors(response)
            print(response.json())
            return ExamplesPage.model_validate(response.json())

    def get_example(self, example_id: str) -> PredictedExample:
        with self.get_client() as client:
            response = client.get(f"/api/v1/examples/{example_id}")
            raise_errors(response)
            return PredictedExample.model_validate(response.json())

    def annotate_example(self, example_id: str, annotations: list[PredictionAnnotation]) -> AnnotateExampleResponse:
        with self.get_client() as client:
            response = client.patch(
                f"/api/v1/examples/{example_id}/annotation",
                json={"annotations": annotations},
            )
            raise_errors(response)
            return AnnotateExampleResponse.model_validate(response.json())

    def _get_git_headers(self, access_token: Optional[str] = None) -> dict[str, str]:
        token = self._resolve_token(access_token)
        return {
            "Authorization": f"token {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "ModaicClient/1.0",
        }

    def create_repo(
        self, repo_path: str, exist_ok: bool = False, private: bool = False, access_token: Optional[str] = None
    ) -> bool:
        """
        Creates a remote repository in modaic hub on the given repo_path. e.g. "user/repo"

        Args:
            repo_path: The path on Modaic hub to create the remote repository.
            exist_ok: If True, don't raise an error if the repository already exists.
            private: Whether the repository should be private.

        Raises:
            RepositoryExistsError: If the repository already exists on the hub.
            AuthenticationError: If authentication fails or access is denied.
            ValueError: If inputs are invalid.

        Returns:
            True if a new repository was created, False if it already existed.
        """
        if not repo_path or not repo_path.strip():
            raise ValueError("Repository ID cannot be empty")

        repo_user, repo_name = repo_path.strip().split("/", 1)
        if len(repo_name) > 100:
            raise ValueError("Repository name too long (max 100 characters)")

        payload = {
            "username": repo_user,
            "name": repo_name,
            "description": "",
            "private": private,
            "auto_init": True,
            "default_branch": "main",
            "trust_model": "default",
        }

        try:
            with self.get_client(access_token=access_token) as client:
                response = client.post(
                    "/api/v2/repos",
                    json=payload,
                    headers=self._get_git_headers(access_token=access_token),
                )

                if response.is_success:
                    return True

                error_data = {}
                try:
                    error_data = response.json()
                except Exception:
                    pass

                error_message = error_data.get("message", f"HTTP {response.status_code}")

                if response.status_code in (409, 422) or "already exists" in error_message.lower():
                    if exist_ok:
                        return False
                    else:
                        raise RepositoryExistsError(f"Repository '{repo_path}' already exists")
                elif response.status_code == 401:
                    raise AuthenticationError("Invalid access token or authentication failed")
                elif response.status_code == 403:
                    raise AuthenticationError("Access denied - insufficient permissions")
                else:
                    raise Exception(f"Failed to create repository: {error_message}")

        except httpx.HTTPError as e:
            raise Exception(f"Request failed: {str(e)}") from e

    def delete_repo(self, repo_path: str, access_token: Optional[str] = None) -> bool:
        """
        Deletes a remote repository from modaic hub.

        Args:
            repo_path: The path on Modaic hub of the repository to delete. e.g. "user/repo"

        Raises:
            AuthenticationError: If authentication fails or access is denied.
            ValueError: If inputs are invalid.

        Returns:
            True if the repository was deleted successfully.
        """
        if not repo_path or not repo_path.strip():
            raise ValueError("Repository ID cannot be empty")

        repo_user, repo_name = repo_path.strip().split("/", 1)

        try:
            with self.get_client(access_token=access_token) as client:
                response = client.delete(
                    f"/api/v2/repos/{repo_user}/{repo_name}",
                    headers=self._get_git_headers(access_token=access_token),
                )

                if response.is_success or response.status_code == 204:
                    return True

                if response.status_code == 401:
                    raise AuthenticationError("Invalid access token or authentication failed")
                elif response.status_code == 403:
                    raise AuthenticationError("Access denied - insufficient permissions")
                elif response.status_code == 404:
                    raise Exception(f"Repository '{repo_path}' not found")
                else:
                    error_data = {}
                    try:
                        error_data = response.json()
                    except Exception:
                        pass
                    error_message = error_data.get("message", f"HTTP {response.status_code}")
                    raise Exception(f"Failed to delete repository: {error_message}")

        except httpx.HTTPError as e:
            raise Exception(f"Request failed: {str(e)}") from e

    def get_user_info(self, access_token: Optional[str] = None) -> dict[str, Any]:
        """
        Returns the user info for the configured modaic token.

        Returns:
            Dict with keys: login, email, avatar_url, name
        """
        token = self._resolve_token(access_token)
        if token is None:
            raise AuthenticationError("No access token provided")

        protocol = "https://" if settings.modaic_git_url.startswith("https://") else "http://"
        url = f"{protocol}{settings.modaic_git_url.replace('https://', '').replace('http://', '')}/api/v1/user"

        with self.get_client(access_token=access_token) as client:
            response = client.get(url, headers=self._get_git_headers(access_token=access_token))
            if response.status_code == 401:
                raise AuthenticationError("Invalid access token or authentication failed")
            raise_errors(response)
            data = response.json()
            return {
                "login": data["login"],
                "email": data["email"],
                "avatar_url": data["avatar_url"],
                "name": data["full_name"],
            }

    def request_confidence_score(
        self,
        prediction_id: str,
        access_token: Optional[str] = None,
    ) -> ConfidenceStatusResponse:
        """Idempotently enqueue confidence scoring for ``prediction_id``."""
        with self.get_client(access_token=access_token) as client:
            response = client.post(
                f"/api/v1/arbiters/predictions/{prediction_id}/confidence",
                timeout=10.0,
            )
            raise_errors(response)
            return ConfidenceStatusResponse.model_validate(response.json())

    def get_confidence_score(
        self,
        prediction_id: str,
        access_token: Optional[str] = None,
    ) -> ConfidenceStatusResponse:
        """Read current state of the confidence resource."""
        with self.get_client(access_token=access_token) as client:
            response = client.get(
                f"/api/v1/arbiters/predictions/{prediction_id}/confidence",
                timeout=10.0,
            )
            raise_errors(response)
            return ConfidenceStatusResponse.model_validate(response.json())

    def wait_for_confidence_score(
        self,
        prediction_id: str,
        access_token: Optional[str] = None,
        timeout: float = 300.0,
        poll_interval: float = 1.0,
    ) -> ConfidenceStatusResponse:
        """POST then poll GET until ``completed`` / ``failed`` / timeout."""
        result = self.request_confidence_score(
            prediction_id=prediction_id, access_token=access_token
        )
        if result.status in {"completed", "failed"}:
            return result

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(poll_interval)
            result = self.get_confidence_score(
                prediction_id=prediction_id, access_token=access_token
            )
            if result.status in {"completed", "failed"}:
                return result
        return result


def get_modaic_client() -> ModaicClient:
    global _modaic_client
    if _modaic_client is None:
        with _client_lock:
            if _modaic_client is None:
                _modaic_client = ModaicClient()
    return _modaic_client


def configure_modaic_client(
    modaic_token: Optional[str] = None,
    base_url: Optional[str] = None,
    *,
    client: Optional[httpx.Client] = None,
    timeout: float = 30.0,
) -> ModaicClient:
    global _modaic_client
    with _client_lock:
        _modaic_client = ModaicClient(modaic_token=modaic_token, base_url=base_url, client=client, timeout=timeout)
    return _modaic_client
