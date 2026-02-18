from typing import Any, Optional, Tuple

import requests
from pydantic import BaseModel
from typing_extensions import TypedDict

from .config import settings


class ArbiterPrediction(BaseModel):
    arbiter_repo: str
    commit_hash: str
    output: Any
    reasoning: str
    messages: list[dict]


class ArbiterPredictResponse(BaseModel):
    example_id: str
    predictions: list[ArbiterPrediction]


_modaic_client = None


class Arbiter:
    client: "ModaicClient"
    repo: str
    revision: str

    def __init__(self, repo: str, revision: str = "main"):
        self.repo = repo
        self.revision = revision
        self.client = get_modaic_client()

    def predict(
        self, input: dict, ground_truth: Optional[str] = None, ground_reasoning: str = ""
    ) -> Tuple[str, ArbiterPrediction]:
        return self.client.predict(input, self, ground_truth, ground_reasoning)

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


def get_modaic_client() -> ModaicClient:
    global _modaic_client
    if _modaic_client is None:
        _modaic_client = ModaicClient()
    return _modaic_client
