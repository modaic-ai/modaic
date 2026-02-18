from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional

import httpx
from dspy import Prediction
from dspy.utils.callback import BaseCallback
from modaic_client import settings
from pydantic import BaseModel

from .exceptions import ModaicError

if TYPE_CHECKING:
    from modaic.hub import Commit
    from modaic.precompiled import PrecompiledProgram
from concurrent.futures import ThreadPoolExecutor


class ModaicTrackCallback(BaseCallback):
    def __init__(self):
        self.inputs_cache = {}

    def on_module_start(self, call_id: str, instance: PrecompiledProgram, inputs: dict):
        kwargs = inputs.get("kwargs", {})
        source = instance._source_commit
        if source and settings.track:
            self.inputs_cache[call_id] = {"kwargs": kwargs, "source": source}

    def on_module_end(self, call_id: str, outputs: Prediction, exception: Optional[Exception]):
        if call_id in self.inputs_cache:
            entry = self.inputs_cache.pop(call_id)
            log_prediction(entry["source"], entry["kwargs"], outputs)


def extract_output(prediction: Prediction) -> tuple[Any, str, str]:
    """
    Extracts the predicted output from the prediction

    Returns:
    - output: The predicted output
    - serialized_output: The serialized output
    - output_field: The field name of the output
    """

    if len(prediction._store.keys()) != 2 or "reasoning" not in prediction._store:
        raise ModaicError("Arbiter must return a Prediction with 2 fields with one of them named 'reasoning'.")

    # extract the field that is not named "reasoning" as output
    output_field, output = next((k, v) for k, v in prediction._store.items() if k != "reasoning")
    # Convert the output to a string so we can store it in the database
    if isinstance(output, BaseModel):
        serialized_output = output.model_dump_json()
    else:
        serialized_output = json.dumps(output)
    return output, serialized_output, output_field


executor = ThreadPoolExecutor(max_workers=2)


def log_prediction(commit: Commit, inputs: dict, prediction: Prediction) -> None:
    output, serialized_output, output_field = extract_output(prediction)

    if hasattr(prediction, "reasoning"):
        reasoning = prediction.reasoning
    else:
        reasoning = None

    example = {
        "arbiter_repo": commit.repo,
        "arbiter_hash": commit.sha,
        "input": inputs,
        "output": serialized_output,
        "reasoning": reasoning,
    }
    body = json.dumps(example)
    executor.submit(
        httpx.post(
            f"{settings.modaic_api_url}/api/v1/examples",
            content=body,
            headers={
                "Content-Type": "application/x-ndjson",
                "Authorization": f"Bearer {settings.modaic_token}",
            },
        )
    )
