"""Offline validation of every question/label in the retained live workload."""

import json
from pathlib import Path
from typing import Any

import pytest

from modaic import Choice, Noul, Score

DOMAINS = json.loads(
    Path(__file__).with_name("fixtures").joinpath("domain-stress.json").read_text()
)["domains"]


@pytest.mark.parametrize("domain", DOMAINS, ids=[d["slug"] for d in DOMAINS])
def test_domain_fixture(domain: dict[str, Any]) -> None:
    constructors = {"noul": Noul, "choice": Choice, "score": Score}
    questions = domain["questions"]
    for question in questions.values():
        parsed = constructors[question["type"]](**question)
        assert parsed.model_dump(exclude_unset=True) == question
    assert len(domain["examples"]) == 12
    assert len({e["name"] for e in domain["examples"]}) == 12
    binary, choice, score = domain["keys"]
    for example in domain["examples"]:
        truth = example["groundTruth"]
        assert set(truth) == set(questions)
        assert isinstance(truth[binary], bool)
        assert truth[choice] in questions[choice]["criteria"]
        assert 0 <= truth[score] < len(questions[score]["criteria"])
        assert example["rationale"] and example["state"] is not None
