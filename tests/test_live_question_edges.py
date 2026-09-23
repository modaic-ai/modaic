"""Opt-in real inference and validation tests, without repository capture.

Set MODAIC_E2E=1, MODAIC_API_URL (local server), and MODAIC_API_KEY.
These requests may incur inference charges. Never run in ordinary CI.
"""

import inspect
import math
import os
from typing import Any

import pytest
from pydantic import Field, create_model

from modaic import (
    AsyncModaic,
    Choice,
    ChoiceAnswer,
    DecisionResponse,
    Modaic,
    ModaicAPIError,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
)

pytestmark = pytest.mark.skipif(os.getenv("MODAIC_E2E") != "1", reason="Live API opt-in required")

CASES = [
    (
        "minimal_singletons",
        {"ticket": "Duplicate payment"},
        Noul(instructions="Was there a duplicate payment?"),
        Choice(criteria={"only": "Only option"}),
        Score(criteria=["Only level"]),
    ),
    (
        "null_instructions",
        {"ticket": "Duplicate payment"},
        Noul(instructions=None, criteria={"true": "Duplicate payment", "false": "No duplicate"}),
        Choice(instructions=None, criteria={"yes": None, "no": None}),
        Score(instructions=None, criteria=["False", "True"]),
    ),
    (
        "structured",
        {"ticket": "Duplicate payment", "amount": 0, "resolved": False},
        Noul(
            instructions={"task": "Is a refund needed?"}, criteria={"true": {"reason": "Duplicate"}}
        ),
        Choice(
            instructions=["Choose a category"],
            criteria={"billing": {"examples": ["Duplicate payment"]}, "other": ["Everything else"]},
        ),
        Score(
            instructions={"rubric": "Urgency"},
            criteria=[{"description": "Routine"}, ["Urgent", "Financial impact"]],
        ),
    ),
    (
        "unicode_ids",
        "Paiement en double. 二重請求。💳",
        Noul(criteria={"false": "No duplicate"}),
        Choice(
            criteria={"0": "Routine", "false": "Unknown", "needs review 🚨": "Duplicate charge"}
        ),
        Score(criteria=["問題なし", "要確認", "緊急 🚨"]),
    ),
    (
        "many_levels",
        {"value": 0},
        Noul(instructions="Is the value nonzero?"),
        Choice(criteria={str(i): f"Category {i}" for i in range(12)}),
        Score(criteria=[f"Severity {i}" for i in range(7)]),
    ),
    (
        "empty_duplicate_descriptions",
        "",
        Noul(criteria={"true": "", "false": ""}),
        Choice(criteria={"a": "", "b": ""}),
        Score(criteria=["Same", "Same"]),
    ),
    (
        "array_state",
        [],
        Noul(instructions="Does the input contain information?", criteria={}),
        Choice(criteria={"empty": "No information", "present": "Some information"}),
        Score(criteria=["Nothing", "Partial", "Complete"]),
    ),
]

INVALID = [
    ("choice_without_options", {"q": {"type": "choice", "criteria": {}}}),
    ("empty_questions", {}),
    ("empty_score", {"q": {"type": "score", "criteria": []}}),
    ("null_score_level", {"q": {"type": "score", "criteria": [None]}}),
    ("numeric_choice_description", {"q": {"type": "choice", "criteria": {"a": 0}}}),
    ("empty_choice_label", {"q": {"type": "choice", "criteria": {"": "Empty"}}}),
    ("invalid_noul_criteria", {"q": {"type": "noul", "criteria": "yes"}}),
    ("unknown_question_type", {"q": {"type": "unknown"}}),
]


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
async def test_live_question_edges(mode: str, case: tuple[Any, ...]) -> None:
    name, state, noul, choice, score = case
    ids = (
        ("billing.check", "triage / priorité", "score.結果")
        if name == "unicode_ids"
        else ("n", "c", "s")
    )
    questions = dict(zip(ids, (noul, choice, score), strict=True))
    response_type = create_model(
        "EdgeResponse",
        __base__=DecisionResponse,
        binary=(NoulAnswer, Field(alias=ids[0])),
        category=(ChoiceAnswer, Field(alias=ids[1])),
        rating=(ScoreAnswer, Field(alias=ids[2])),
    )
    # No base_url: exercise MODAIC_API_URL through both clients.
    assert os.environ["MODAIC_API_URL"]
    client = (Modaic if mode == "sync" else AsyncModaic)(timeout=60)
    try:
        result = client.decisions.create(
            model="typesafe/jev-latest",
            state=state,
            questions=questions,
            response_model=response_type,
        )
        if inspect.isawaitable(result):
            result = await result
        assert isinstance(result, response_type)
        assert set(result.answers) == set(ids)
        assert result.binary == result.nouls[ids[0]]
        assert result.category == result.choices[ids[1]]
        assert result.rating == result.scores[ids[2]]
        assert math.isfinite(result.binary.noul) and 0 <= result.binary.noul <= 1
        assert result.category.choice in choice.criteria
        assert set(result.category.probabilities) == set(choice.criteria)
        assert math.isfinite(result.rating.score)
        assert 0 <= result.rating.score <= len(score.criteria) - 1
        assert result.rating.legend == {str(i): value for i, value in enumerate(score.criteria)}
        assert set(result.rating.probabilities) == set(result.rating.legend)
        for answer in (result.category, result.rating):
            assert 0 <= answer.confidence <= 1
            assert all(math.isfinite(p) and 0 <= p <= 1 for p in answer.probabilities.values())
            assert sum(answer.probabilities.values()) == pytest.approx(1, abs=0.01)
        assert result.usage.input_tokens >= 0 and result.usage.output_tokens >= 0
        assert result.request_id
        print(
            f"PASS {mode}/{name}: all three answer types, distributions, typed aliases", flush=True
        )
    finally:
        closing = client.close()
        if inspect.isawaitable(closing):
            await closing


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("case", INVALID, ids=[case[0] for case in INVALID])
async def test_live_invalid_questions(mode: str, case: tuple[str, Any]) -> None:
    assert os.environ["MODAIC_API_URL"]
    client = (Modaic if mode == "sync" else AsyncModaic)(timeout=60)
    try:
        with pytest.raises(ModaicAPIError) as caught:
            result = client.decisions.create(
                model="typesafe/jev-latest", state=None, questions=case[1]
            )
            if inspect.isawaitable(result):
                await result
        assert caught.value.status_code == 422
        assert caught.value.request_id
        assert caught.value.code == "validation_error"
    finally:
        closing = client.close()
        if inspect.isawaitable(closing):
            await closing


# These inputs must remain supported by the public schema and live inference.
CONTRACT_EDGES = [
    ("null_state", None, {"q": Noul(instructions="Has content?")}),
    ("false_state", False, {"q": Noul(instructions="Has content?")}),
    ("zero_state", 0, {"q": Noul(instructions="Has content?")}),
    ("number_instructions", {"value": 0}, {"q": Noul(instructions=0)}),
    (
        "false_instructions",
        {"value": 0},
        {"q": Choice(instructions=False, criteria={"a": "A", "b": "B"})},
    ),
    ("true_instructions", {"value": 0}, {"q": Score(instructions=True, criteria=["Low", "High"])}),
    ("noul_without_context", {"ticket": "Duplicate payment"}, {"q": Noul()}),
    (
        "empty_question_id",
        {"ticket": "Duplicate payment"},
        {"": Noul(instructions="Duplicate payment?")},
    ),
]


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("case", CONTRACT_EDGES, ids=[case[0] for case in CONTRACT_EDGES])
async def test_live_schema_contract_edges(mode: str, case: tuple[Any, ...]) -> None:
    assert os.environ["MODAIC_API_URL"]
    client = (Modaic if mode == "sync" else AsyncModaic)(timeout=60)
    try:
        response_type = (
            create_model(
                "EmptyIdResponse", __base__=DecisionResponse, empty=(NoulAnswer, Field(alias=""))
            )
            if case[0] == "empty_question_id"
            else DecisionResponse
        )
        result = client.decisions.create(
            model="typesafe/jev-latest",
            state=case[1],
            questions=case[2],
            response_model=response_type,
        )
        if inspect.isawaitable(result):
            result = await result
        assert set(result.answers) == set(case[2])
        if case[0] == "empty_question_id":
            assert result.empty == result.nouls[""]
    finally:
        closing = client.close()
        if inspect.isawaitable(closing):
            await closing
