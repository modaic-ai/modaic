"""Recheck retained workload artifacts without repeating inference.

Usage: python tests/verify_stress_report.py INPUT.json VERIFIED.json
Requires the same explicit live configuration as stress_domains.py.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from stress_domains import wait_history

from modaic import AsyncModaic


async def main() -> None:
    if os.getenv("MODAIC_E2E") != "1":
        raise SystemExit("Set MODAIC_E2E=1")
    for key in ("MODAIC_API_KEY", "MODAIC_API_URL"):
        if not os.getenv(key):
            raise SystemExit(f"Missing {key}")
    source, target = sys.argv[1:]
    report = json.loads(Path(source).read_text())
    fixture = json.loads(
        Path(__file__).with_name("fixtures").joinpath("domain-stress.json").read_text()
    )
    domains = {d["slug"]: d for d in fixture["domains"]}
    failures = 0
    async with AsyncModaic(timeout=60) as client:
        for domain in report["domains"]:
            workspace, slug = domain["model"].split("/")
            model = await client.models.get(workspace=workspace, model=slug)
            examples = {e["name"]: e for e in domains[domain["domain"]]["examples"]}
            for index, row in enumerate(domain["examples"]):
                try:
                    e = examples[row["name"]]
                    history = await wait_history(model, row["id"])
                    assert len(history.decisions) == 1
                    saved = history.decisions[0]
                    assert saved.error is None and saved.source == "live"
                    assert saved.request["state"] == e["state"]
                    assert saved.request["questions"] == domains[domain["domain"]]["questions"]
                    assert saved.answers == row["response"]["answers"]
                    expected_id = row["response"].get("decision_id") or row["response"].get(
                        "decisionId"
                    )
                    assert saved.id == expected_id
                    stored = await model.examples.get(row["id"])
                    assert stored.state == e["state"]
                    assert stored.annotation is not None
                    assert stored.annotation.ground_truth == e["groundTruth"]
                    assert stored.annotation.ground_reasoning == e["rationale"]
                    assert stored.latest_decision is not None
                    assert stored.latest_decision.id == saved.id and stored.decision_count == 1
                    if index == 0:
                        replay = await model.decisions.create(
                            state=e["state"],
                            example_id=row["id"],
                            idempotency_key=f"stress-{report['run_id']}-{row['id']}",
                        )
                        assert replay.decision_id == saved.id
                        assert replay.model_dump()["answers"] == saved.answers
                        assert len((await model.examples.list_decisions(row["id"])).decisions) == 1
                        row["replay_verified"] = True
                    row["persistence_recheck_passed"] = True
                except Exception as error:
                    row["persistence_recheck_error"] = f"{type(error).__name__}: {error}"
                    failures += 1
            try:
                items: list[Any] = []
                for page in range(1, 4):
                    result = await model.examples.list(page=page, page_size=5)
                    assert result.total == 12 and result.total_pages == 3
                    items.extend(result.items)
                assert len(items) == 12
                assert {e.id for e in items} == {r["id"] for r in domain["examples"]}
                assert all(e.latest_decision is not None for e in items)
                domain["pagination_recheck_passed"] = True
                domain["stored_outputs"] = len(items)
            except Exception as error:
                domain["recheck_error"] = f"{type(error).__name__}: {error}"
                failures += 1
            print(
                f"VERIFIED {domain['model']}: "
                f"{sum(r.get('persistence_recheck_passed', False) for r in domain['examples'])}/12",
                flush=True,
            )
            Path(target).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Verification failures: {failures}")
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    asyncio.run(main())
