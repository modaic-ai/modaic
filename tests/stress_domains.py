"""Opt-in, billable domain stress workload. Retains every model and example.

MODAIC_E2E=1 MODAIC_API_URL=... MODAIC_API_KEY=... MODAIC_E2E_WORKSPACE=...
  python tests/stress_domains.py RUN_ID OUTPUT.json
Filesystem access belongs to this QA runner, never to the SDK runtime.
"""

import asyncio
import inspect
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import Field, create_model

from modaic import (
    AsyncModaic,
    Choice,
    ChoiceAnswer,
    DecisionResponse,
    Modaic,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
)


async def call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    value = fn(*args, **kwargs)
    return await value if inspect.isawaitable(value) else value


async def wait_history(model: Any, example_id: str) -> Any:
    deadline = time.monotonic() + 30
    while True:
        history = await call(model.examples.list_decisions, example_id)
        if history.decisions:
            return history
        if time.monotonic() >= deadline:
            raise AssertionError("Decision was not persisted within 30 seconds")
        await asyncio.sleep(0.5)


def validate(result: Any, questions: dict[str, Any]) -> None:
    assert set(result.answers) == set(questions), "Answer keys differ"
    assert result.request_id, "Missing request ID"
    assert result.usage.input_tokens >= 0 and result.usage.output_tokens >= 0
    for name, answer in result.answers.items():
        question = questions[name]
        assert answer.type == question.type
        if isinstance(answer, NoulAnswer):
            assert 0 <= answer.noul <= 1
        else:
            assert 0 <= answer.confidence <= 1
            assert all(0 <= p <= 1 for p in answer.probabilities.values())
            assert abs(sum(answer.probabilities.values()) - 1) < 0.01
            if isinstance(answer, ChoiceAnswer):
                assert answer.choice in question.criteria
                assert set(answer.probabilities) == set(question.criteria)
            else:
                assert 0 <= answer.score <= len(question.criteria) - 1
                assert len(answer.legend) == len(question.criteria)


async def run_domain(domain: dict[str, Any], run_id: str, report: dict[str, Any]) -> None:
    mode = domain["mode"]
    client = (Modaic if mode == "sync" else AsyncModaic)(timeout=60)
    workspace = os.environ["MODAIC_E2E_WORKSPACE"]
    slug = f"sdk-stress-{domain['slug']}-{run_id}"
    entry = {
        "domain": domain["slug"],
        "client": f"python-{mode}",
        "model": f"{workspace}/{slug}",
        "examples": [],
        "errors": [],
    }
    report["domains"].append(entry)
    constructors = {"noul": Noul, "choice": Choice, "score": Score}
    questions = {
        name: constructors[q["type"]](**{k: v for k, v in q.items() if k != "type"})
        for name, q in domain["questions"].items()
    }
    binary, choice, score = domain["keys"]
    response_class = create_model(
        "DomainResponse",
        __base__=DecisionResponse,
        binary_answer=(NoulAnswer, Field(alias=binary)),
        choice_answer=(ChoiceAnswer, Field(alias=choice)),
        score_answer=(ScoreAnswer, Field(alias=score)),
    )
    started = time.perf_counter()
    try:
        model = await call(
            client.models.create,
            workspace=workspace,
            slug=slug,
            description=f"Synthetic SDK stress test: {domain['description']}",
            model="typesafe/jev-latest",
            questions=questions,
        )
        assert model.workspace == workspace
        entry["model_id"] = model.id
        fetched = await call(client.models.get, workspace=workspace, model=slug)
        assert fetched.id == model.id and fetched.workspace.slug == workspace
        examples = [{**example, "id": str(uuid4())} for example in domain["examples"]]
        await call(
            model.examples.ingest,
            examples=[
                {
                    "id": e["id"],
                    "state": e["state"],
                    "annotation": {
                        "ground_truth": e["groundTruth"],
                        "ground_reasoning": e["rationale"],
                    },
                }
                for e in examples
            ],
        )
        entry["setup_ms"] = round((time.perf_counter() - started) * 1000)
        print(f"READY {entry['model']} ({mode})", flush=True)
        semaphore = asyncio.Semaphore(1 if mode == "sync" else 3)

        async def decide(e: dict[str, Any], index: int) -> None:
            async with semaphore:
                row = {"name": e["name"], "id": e["id"], "expected": e["groundTruth"]}
                entry["examples"].append(row)
                begin = time.perf_counter()
                request = {
                    "state": e["state"],
                    "example_id": e["id"],
                    "idempotency_key": f"stress-{run_id}-{e['id']}",
                    "response_model": response_class,
                }
                try:
                    if index % 2:
                        result = await call(
                            client.decisions.create, model=entry["model"], **request
                        )
                    else:
                        result = await call(model.decisions.create, **request)
                    row["latency_ms"] = round((time.perf_counter() - begin) * 1000)
                    row["response"] = result.model_dump(mode="json")
                    row["request_id"] = result.request_id
                    validate(result, questions)
                    assert result.binary_answer == result.nouls[binary]
                    assert result.choice_answer == result.choices[choice]
                    assert result.score_answer == result.scores[score]
                    assert result.captured and result.example_id == e["id"]
                    assert result.decision_id and result.revision and result.checkpoint is not None
                    row["quality"] = {
                        "noul_matches": (result.binary_answer.noul >= 0.5)
                        == e["groundTruth"][binary],
                        "choice_matches": result.choice_answer.choice == e["groundTruth"][choice],
                        "score_absolute_error": abs(
                            result.score_answer.score - e["groundTruth"][score]
                        ),
                    }
                    if index == 0:
                        replay_begin = time.perf_counter()
                        replay = await call(model.decisions.create, **request)
                        assert replay.model_dump() == result.model_dump(), "Replay changed response"
                        row["replay_verified"] = True
                        row["immediate_replay_ms"] = round(
                            (time.perf_counter() - replay_begin) * 1000
                        )
                    history = await wait_history(model, e["id"])
                    row["persistence_verified_ms"] = round((time.perf_counter() - begin) * 1000)
                    assert len(history.decisions) == 1, "Expected one captured decision"
                    saved = history.decisions[0]
                    assert saved.id == result.decision_id and saved.error is None
                    assert saved.request["state"] == e["state"]
                    assert saved.request["questions"] == domain["questions"]
                    assert saved.answers == result.model_dump()["answers"]
                    stored = await call(model.examples.get, e["id"])
                    assert stored.state == e["state"]
                    assert stored.annotation.ground_truth == e["groundTruth"]
                    assert stored.annotation.ground_reasoning == e["rationale"]
                    assert stored.latest_decision.id == result.decision_id
                    assert stored.decision_count == 1
                    row["passed"] = True
                except Exception as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                    row["elapsed_ms"] = round((time.perf_counter() - begin) * 1000)
                print(
                    f"{'PASS' if row.get('passed') else 'FAIL'} {e['name']} "
                    f"{row.get('latency_ms', row.get('elapsed_ms'))}ms {row.get('error', '')}",
                    flush=True,
                )

        await asyncio.gather(*(decide(e, i) for i, e in enumerate(examples)))
        pages = [await call(model.examples.list, page=p, page_size=5) for p in range(1, 4)]
        assert all(p.total == 12 and p.total_pages == 3 for p in pages)
        items = [item for page in pages for item in page.items]
        assert len(items) == 12 and {e.id for e in items} == {e["id"] for e in examples}
        entry["pagination_verified"] = True
        entry["stored_outputs"] = sum(e.latest_decision is not None for e in items)
    except Exception as error:
        entry["errors"].append(f"{type(error).__name__}: {error}")
        print(f"DOMAIN FAIL {entry['model']}: {entry['errors'][-1]}", flush=True)
    finally:
        await call(client.close)


async def main() -> None:
    if os.getenv("MODAIC_E2E") != "1":
        raise SystemExit("Set MODAIC_E2E=1; this makes billable requests and retains data")
    for name in ("MODAIC_API_KEY", "MODAIC_API_URL", "MODAIC_E2E_WORKSPACE"):
        if not os.getenv(name):
            raise SystemExit(f"Missing {name}")
    run_id, output = sys.argv[1:]
    domains = json.loads(
        Path(__file__).with_name("fixtures").joinpath("domain-stress.json").read_text()
    )["domains"]
    report = {"run_id": run_id, "api_url": os.environ["MODAIC_API_URL"], "domains": []}
    for domain in domains:
        if domain["mode"] != "typescript":
            await run_domain(domain, run_id, report)
            Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    failures = sum(
        len(d["errors"]) + sum(not e.get("passed") for e in d["examples"])
        for d in report["domains"]
    )
    print(f"Finished: {failures} transport/contract/persistence failures; report {output}")
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    asyncio.run(main())
