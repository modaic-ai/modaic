import json
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Tuple

import httpx
from pydantic import BaseModel, PrivateAttr
from typing_extensions import TypedDict

import modaic_client.exceptions as exceptions

from .config import settings
from .exceptions import AuthenticationError, RepositoryExistsError
from .schemas import (
    AnnotateExampleResponse,
    ArbiterPredictResponse,
    ConfidenceScoreResponse,
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
    commit_hash: str
    output: Output
    reasoning: str
    messages: list[dict]
    example_id: Optional[str] = None
    prediction_id: Optional[str] = None
    _client: "ModaicClient" = PrivateAttr()
    _confidence: float | None = PrivateAttr(default=None)

    def create_confidence_score(
        self,
        access_token: Optional[str] = None,
    ) -> ConfidenceScoreResponse:
        return self._client.create_confidence_score(
            arbiter_repo=self.arbiter_repo,
            messages=self.messages,
            access_token=access_token,
            prediction_id=self.prediction_id,
        )

    @property
    def confidence(self) -> float:
        if self._confidence is not None:
            return self._confidence
        confidence = self.create_confidence_score().confidence
        self._confidence = confidence
        return confidence


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

    def __call__(self, ground_truth: Optional[str] = None, ground_reasoning: str = "", **inputs) -> ArbiterPrediction:
        return self.predict(ground_truth=ground_truth, ground_reasoning=ground_reasoning, **inputs)

    def predict(self, ground_truth: Optional[str] = None, ground_reasoning: str = "", **inputs) -> ArbiterPrediction:
        return self.client.predict(inputs, self, ground_truth, ground_reasoning)

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


class GroundData(TypedDict):
    ground_truth: Optional[str]
    ground_reasoning: str


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
        self, input: dict, arbiters: list[Arbiter], ground_data: Optional[list[GroundData]] = None
    ) -> ArbiterPredictResponse:
        arbiters_data = [arbiter.to_dict() for arbiter in arbiters]

        if ground_data is not None:
            for arbiter_dict, ground in zip(arbiters_data, ground_data, strict=True):
                arbiter_dict["ground_truth"] = ground.get("ground_truth")
                arbiter_dict["ground_reasoning"] = ground.get("ground_reasoning", "")

        with self.get_client() as client:
            response = client.post(
                "/api/v1/arbiters/predictions",
                json={
                    "input": input,
                    "arbiters": arbiters_data,
                },
            )
            raise_errors(response)
            return ArbiterPredictResponse.model_validate(response.json())

    def predict(
        self, input: dict, arbiter: Arbiter, ground_truth: Optional[str] = None, ground_reasoning: str = ""
    ) -> ArbiterPrediction:
        response = self.predict_all(
            input, [arbiter], [{"ground_truth": ground_truth, "ground_reasoning": ground_reasoning}]
        )
        example_id = response.example_id
        prediction = response.predictions[0]
        arbiter_prediction = ArbiterPrediction(
            example_id=example_id,
            arbiter_repo=prediction.arbiter_repo,
            commit_hash=prediction.commit_hash,
            output=prediction.output,
            reasoning=prediction.reasoning,
            messages=prediction.messages,
            prediction_id=prediction.prediction_id,
        )
        arbiter_prediction._client = self
        return arbiter_prediction

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

    def create_confidence_score(
        self,
        arbiter_repo: str,
        messages: list[dict[str, Any]],
        access_token: Optional[str] = None,
        prediction_id: str | None = None,
    ) -> ConfidenceScoreResponse:
        with self.get_client(access_token=access_token) as client:
            response = client.post(
                "/api/v1/arbiters/predictions/confidence",
                json={"arbiter_repo": arbiter_repo, "prediction_id": prediction_id, "messages": messages},
                timeout=300.0,
            )
            raise_errors(response)
            return ConfidenceScoreResponse.model_validate(response.json())


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
