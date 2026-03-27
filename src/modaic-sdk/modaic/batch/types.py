from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import dspy
from pydantic import BaseModel
from typing_extensions import TypedDict

from .storage import get_batch_duckdb_path

try:
    import duckdb
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'modaic.batch requires `duckdb`. Install a batch client extra with `uv add "modaic[openai]"` '
        "(or `anthropic`, `azure`, `together`, `fireworks`, `modal`, or `vllm`)."
    ) from exc


class Message(TypedDict):
    role: str
    content: str | list[dict]


class BatchRequestItem(TypedDict):
    id: str
    model: str
    messages: list[Message]
    lm_kwargs: dict


@dataclass
class FailedPrediction:
    """Represents a failed prediction that can be retried with a different adapter."""

    error: str
    index: int


class Outputs(TypedDict):
    text: str | None
    reasoning_content: str | None


class ABatchRow(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    prediction: dspy.Prediction | FailedPrediction
    messages: list[Message]
    outputs: Outputs
    example: dict[str, Any]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=str)


def _serialize_prediction(prediction: dspy.Prediction | FailedPrediction) -> tuple[str, str]:
    if isinstance(prediction, FailedPrediction):
        return (
            "failed_prediction",
            _json_dumps(
                {
                    "error": prediction.error,
                    "index": prediction.index,
                    "_messages": getattr(prediction, "_messages", None),
                    "_outputs": getattr(prediction, "_outputs", None),
                }
            ),
        )
    payload = {
        "store": prediction.toDict() if hasattr(prediction, "toDict") else dict(prediction),
        "_messages": getattr(prediction, "_messages", None),
        "_outputs": getattr(prediction, "_outputs", None),
    }
    return "prediction", _json_dumps(payload)


def _deserialize_prediction(kind: str, payload: str) -> dspy.Prediction | FailedPrediction:
    data = json.loads(payload)
    if kind == "failed_prediction":
        prediction = FailedPrediction(error=data["error"], index=data["index"])
    else:
        prediction = dspy.Prediction(**data["store"])

    if data.get("_messages") is not None:
        prediction._messages = data["_messages"]
    if data.get("_outputs") is not None:
        prediction._outputs = data["_outputs"]
    return prediction


class ABatchResult:
    def __init__(self, batch_id: str, path: Path):
        self.batch_id = batch_id
        self.path = Path(path)

    @classmethod
    def from_rows(cls, batch_id: str, rows: list[ABatchRow], predict_index: int = 0) -> ABatchResult:
        path = get_batch_duckdb_path(batch_id, predict_index=predict_index)
        conn = duckdb.connect(str(path))
        try:
            conn.execute("DROP TABLE IF EXISTS batch_metadata")
            conn.execute("DROP TABLE IF EXISTS abatch_rows")
            conn.execute("CREATE TABLE batch_metadata (batch_id VARCHAR NOT NULL)")
            conn.execute(
                """
                CREATE TABLE abatch_rows (
                    row_index BIGINT NOT NULL,
                    prediction_kind VARCHAR NOT NULL,
                    prediction_payload VARCHAR NOT NULL,
                    messages_payload VARCHAR NOT NULL,
                    outputs_payload VARCHAR NOT NULL,
                    example_payload VARCHAR NOT NULL
                )
                """
            )
            conn.execute("INSERT INTO batch_metadata VALUES (?)", [batch_id])
            conn.executemany(
                "INSERT INTO abatch_rows VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        row_index,
                        *_serialize_prediction(row.prediction),
                        _json_dumps(row.messages),
                        _json_dumps(row.outputs),
                        _json_dumps(row.example),
                    )
                    for row_index, row in enumerate(rows)
                ],
            )
        finally:
            conn.close()
        return cls(batch_id=batch_id, path=path)

    def __iter__(self) -> Iterator[ABatchRow]:
        yield from self.iter()

    def iter(self) -> Iterator[ABatchRow]:
        conn = duckdb.connect(str(self.path), read_only=True)
        try:
            result = conn.execute(
                """
                SELECT prediction_kind, prediction_payload, messages_payload, outputs_payload, example_payload
                FROM abatch_rows
                ORDER BY row_index
                """
            )
            while True:
                rows = result.fetchmany(128)
                if not rows:
                    break
                for prediction_kind, prediction_payload, messages_payload, outputs_payload, example_payload in rows:
                    yield ABatchRow(
                        prediction=_deserialize_prediction(prediction_kind, prediction_payload),
                        messages=json.loads(messages_payload),
                        outputs=json.loads(outputs_payload),
                        example=json.loads(example_payload),
                    )
        finally:
            conn.close()

    def __len__(self) -> int:
        conn = duckdb.connect(str(self.path), read_only=True)
        try:
            row_count = conn.execute("SELECT COUNT(*) FROM abatch_rows").fetchone()
        finally:
            conn.close()
        return 0 if row_count is None else int(row_count[0])

    def __getitem__(self, index: int) -> ABatchRow:
        if index < 0:
            index += len(self)
        if index < 0:
            raise IndexError("ABatchResult index out of range")

        conn = duckdb.connect(str(self.path), read_only=True)
        try:
            row = conn.execute(
                """
                SELECT prediction_kind, prediction_payload, messages_payload, outputs_payload, example_payload
                FROM abatch_rows
                WHERE row_index = ?
                """,
                [index],
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            raise IndexError("ABatchResult index out of range")

        prediction_kind, prediction_payload, messages_payload, outputs_payload, example_payload = row
        return ABatchRow(
            prediction=_deserialize_prediction(prediction_kind, prediction_payload),
            messages=json.loads(messages_payload),
            outputs=json.loads(outputs_payload),
            example=json.loads(example_payload),
        )


class BatchRequest(TypedDict):
    """
    Common batch request format for all providers.

    Args:
        requests: List of batch request items.
        model: Model to use for the batch request if all requests have the same model. None means all requests have different models.
        lm_kwargs: LM kwargs to use for the batch request if all requests have the same LM kwargs. None means all requests have different LM kwargs.
    """

    requests: list[BatchRequestItem]
    model: Optional[str]
    lm_kwargs: Optional[dict]


class ResultItem(TypedDict, total=False):
    """
    Result item from a batch completion.

    The `text` field is required. Optional fields include `reasoning_content`,
    `logprobs` and `tool_calls`
    which may be present depending on the request configuration.
    """

    text: str
    reasoning_content: Optional[str]
    logprobs: Optional[dict]
    tool_calls: Optional[list[dict]]


ResultItem.__required_keys__ = frozenset({"text"})
ResultItem.__optional_keys__ = frozenset({"reasoning_content", "logprobs", "tool_calls"})


@dataclass
class BatchReponse:
    """Result from a completed batch job (raw API response)."""

    batch_id: str
    status: str
    results: list[dict[str, Any]]
    errors: Optional[list[dict]] = None
    raw_response: Optional[str | dict] = None
