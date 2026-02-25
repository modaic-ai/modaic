import json
from datetime import datetime
from typing import Any, Optional, Tuple

import requests
from pydantic import BaseModel
from typing_extensions import TypedDict

from .config import settings
from .schemas import (
    AnnotateExampleResponse,
    ArbiterPrediction,
    ArbiterPredictResponse,
    ExamplesPage,
    FieldSchema,
    IngestExamplesResponse,
    InitArbiterRequest,
    PredictedExample,
    PredictionAnnotation,
)

_modaic_client = None


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

    def predict(
        self, input: dict, ground_truth: Optional[str] = None, ground_reasoning: str = ""
    ) -> Tuple[str, ArbiterPrediction]:
        return self.client.predict(input, self, ground_truth, ground_reasoning)

    def ingest_examples(self, examples: list[dict]) -> "IngestExamplesResponse":
        for ex in examples:
            ex.setdefault("arbiter_repo", self.repo)
        return self.client.ingest_examples(examples)

    def list_examples(
        self,
        limit: int = 50,
        cursor: Optional[str] = None,
        version: Optional[int] = None,
        commit_hash: Optional[str] = None,
        search: Optional[str] = None,
    ) -> "ExamplesPage":
        return self.client.list_examples(
            user=self._repo_user,
            program=self._repo_name,
            limit=limit,
            cursor=cursor,
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
    def __init__(self, modaic_token: Optional[str] = None):
        self.modaic_token = modaic_token or settings.modaic_token

    def get_arbiter(self, repo: str, revision: str = "main") -> Arbiter:
        arbiter = Arbiter(repo, revision)
        arbiter.set_client(self)
        return arbiter

    def create_arbiter(
        self, repo: str, inputs: list[FieldSchema], output: FieldSchema, instructions: Optional[str] = None
    ) -> Arbiter:
        request = InitArbiterRequest(repo=repo, inputs=inputs, output=output, instructions=instructions)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.modaic_token}",
        }
        response = requests.post(
            f"{settings.modaic_api_url}/api/v1/arbiters",
            json=request.model_dump(),
            headers=headers,
        )
        response.raise_for_status()
        arbiter = Arbiter(repo)
        arbiter.set_client(self)
        return arbiter

    def predict_all(
        self, input: dict, arbiters: list[Arbiter], ground_data: Optional[list[GroundData]] = None
    ) -> ArbiterPredictResponse:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.modaic_token}",
        }

        arbiters_data = [arbiter.to_dict() for arbiter in arbiters]

        if ground_data is not None:
            for arbiter, ground in zip(arbiters, ground_data, strict=True):
                arbiter.to_dict().update(
                    {
                        "ground_truth": ground.get("ground_truth"),
                        "ground_reasoning": ground.get("ground_reasoning", ""),
                    }
                )

        response = requests.post(
            f"{settings.modaic_api_url}/api/v1/arbiters/predictions",
            json={
                "input": input,
                "arbiters": arbiters_data,
            },
            headers=headers,
        )
        response.raise_for_status()
        return ArbiterPredictResponse.model_validate(response.json())

    def predict(
        self, input: dict, arbiter: Arbiter, ground_truth: Optional[str] = None, ground_reasoning: str = ""
    ) -> Tuple[str, ArbiterPrediction]:
        response = self.predict_all(
            input, [arbiter], [{"ground_truth": ground_truth, "ground_reasoning": ground_reasoning}]
        )
        example_id = response.example_id
        prediction = response.predictions[0]
        return example_id, prediction

    def ingest_examples(self, examples: list[dict]) -> IngestExamplesResponse:
        headers = {
            "Content-Type": "text/plain",
            "Authorization": f"Bearer {self.modaic_token}",
        }
        body = "\n".join(json.dumps(ex) for ex in examples)
        response = requests.post(
            f"{settings.modaic_api_url}/api/v1/examples",
            data=body,
            headers=headers,
        )
        response.raise_for_status()
        return IngestExamplesResponse.model_validate(response.json())

    def list_examples(
        self,
        user: str,
        program: str,
        limit: int = 50,
        cursor: Optional[str] = None,
        version: Optional[int] = None,
        commit_hash: Optional[str] = None,
        search: Optional[str] = None,
    ) -> ExamplesPage:
        headers = {
            "Authorization": f"Bearer {self.modaic_token}",
        }
        params: dict[str, Any] = {"user": user, "program": program, "limit": limit}
        if cursor is not None:
            params["cursor"] = cursor
        if version is not None:
            params["version"] = version
        if commit_hash is not None:
            params["commit_hash"] = commit_hash
        if search is not None:
            params["search"] = search

        response = requests.get(
            f"{settings.modaic_api_url}/api/v1/examples",
            params=params,
            headers=headers,
        )
        response.raise_for_status()
        return ExamplesPage.model_validate(response.json())

    def get_example(self, example_id: str) -> PredictedExample:
        headers = {
            "Authorization": f"Bearer {self.modaic_token}",
        }
        response = requests.get(
            f"{settings.modaic_api_url}/api/v1/examples/{example_id}",
            headers=headers,
        )
        response.raise_for_status()
        return PredictedExample.model_validate(response.json())

    def annotate_example(self, example_id: str, annotations: list[PredictionAnnotation]) -> AnnotateExampleResponse:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.modaic_token}",
        }
        response = requests.patch(
            f"{settings.modaic_api_url}/api/v1/examples/{example_id}/annotation",
            json={"annotations": annotations},
            headers=headers,
        )
        response.raise_for_status()
        return AnnotateExampleResponse.model_validate(response.json())


def get_modaic_client() -> ModaicClient:
    global _modaic_client
    if _modaic_client is None:
        _modaic_client = ModaicClient()
    return _modaic_client
