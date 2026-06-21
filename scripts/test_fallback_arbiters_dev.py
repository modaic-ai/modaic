#!/usr/bin/env python
"""Live QA against the DEV Modaic server for the gpt-5.5 / claude-opus-4-8 arbiters.

For each model this script verifies, end to end against the dev stack:

  1. The arbiter is created and pushed to the dev hub.
  2. Inference runs on the LIVE provider (OpenAI / Anthropic): the server
     executes the arbiter LM and returns a real classification + reasoning.
  3. Confidence scoring completes and returns a calibrated score in [0, 1].

gpt-5.5 and claude-opus-4-8 have no probe of their own, so step 3 exercises the
server's probe resolution path for these models — i.e. that confidence scoring
still produces a usable score for them.

Configuration (env vars; dev defaults mirror qa/envs.json):

  DEV_MODAIC_TOKEN   (or MODAIC_TOKEN)  required — dev hub token
  MODAIC_API_URL     default http://localhost:8000        (dev REST API)
  MODAIC_GIT_URL     default https://dev.git.modaic.dev   (dev hub git)
  MODAIC_NAMESPACE   default "modaic"  — hub org/user the test repos push under

The OPENAI_API_KEY / ANTHROPIC_API_KEY the server needs to execute the arbiter
LM must be configured on the dev server / Modaic Hub. This script does not send
provider keys; it only drives the SDK.

Run:
  uv run python scripts/test_fallback_arbiters_dev.py
  uv run python scripts/test_fallback_arbiters_dev.py --only gpt-5.5
  MODAIC_API_URL=https://dev.api.modaic.dev uv run python scripts/test_fallback_arbiters_dev.py
"""
# ruff: noqa: T201  (this is a CLI script — prints are the interface)
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass

import dspy
import modaic
from modaic_client import ModaicClient, configure_modaic_client

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_GIT_URL = "https://dev.git.modaic.dev"
DEFAULT_NAMESPACE = "modaic"

# Per-request HTTP timeout. Must exceed the server's ~120s SSE connection cap
# so the confidence-stream reconnect logic kicks in instead of read-timing-out.
CLIENT_TIMEOUT = 300.0
# Overall deadline for confidence scoring — the probe runs on Modal and the
# gpt-oss fallback path can cold-boot, so give it room.
CONFIDENCE_TIMEOUT = 600.0


class SpamClassifier(dspy.Signature):
    """Classify whether an email is spam or not."""

    subject: str = dspy.InputField(desc="The email subject line")
    body: str = dspy.InputField(desc="The email body")
    is_spam: bool = dspy.OutputField(desc="Whether the email is spam")


# An unambiguous spam example so the live model's output is easy to eyeball.
SPAM_INPUT = {
    "subject": "You won a million dollars!",
    "body": "Click here to claim your prize now! Limited time offer, act fast!!!",
}


@dataclass(frozen=True)
class Case:
    label: str  # what the user calls it
    model: str  # litellm model string the arbiter LM uses
    repo_suffix: str  # hub repo name (under the namespace)


CASES = [
    Case("gpt-5.5", "openai/gpt-5.5", "qa-fallback-gpt-5-5"),
    Case("opus-4.8", "anthropic/claude-opus-4-8", "qa-fallback-opus-4-8"),
]


def _mask(token: str) -> str:
    if len(token) <= 8:
        return "****"
    return f"{token[:4]}…{token[-4:]}"


def _extract_is_spam(output: object) -> object:
    """Best-effort read of the is_spam field off the Output model for a soft
    sanity print — never raises, structure varies by serialization."""
    for getter in (
        lambda: output.is_spam,
        lambda: output["is_spam"],  # type: ignore[index]
        lambda: output.model_dump().get("is_spam"),  # type: ignore[attr-defined]
    ):
        try:
            return getter()
        except Exception:
            continue
    return "<unknown>"


def run_case(client: ModaicClient, case: Case, namespace: str, confidence_timeout: float) -> bool:
    repo = f"{namespace}/{case.repo_suffix}"
    print(f"\n{'=' * 72}\n▶ {case.label}  ({case.model})\n  repo: {repo}\n{'=' * 72}")

    # 1) Create the repo (idempotent) and push the arbiter.
    created = client.create_repo(repo, exist_ok=True, private=True)
    print(f"  create_repo: {'created' if created else 'already existed'}")

    arbiter_program = modaic.Predict(
        SpamClassifier, lm=dspy.LM(model=case.model)
    ).as_arbiter()
    print(f"  arbiter metadata: {arbiter_program.metadata}")

    commit = arbiter_program.push_to_hub(
        repo, commit_message=f"qa: fallback arbiter smoke test ({case.label})"
    )
    print(f"  push_to_hub: {commit}")

    arbiter = client.get_arbiter(repo)

    # 2) Predict — this runs inference on the live provider server-side and,
    #    with compute_confidence=True, kicks off confidence scoring.
    print("  running prediction (live provider inference)…")
    t0 = time.monotonic()
    result = client.predict(input=SPAM_INPUT, arbiter=arbiter, compute_confidence=True)
    infer_ms = (time.monotonic() - t0) * 1000

    if result.output is None:
        raise AssertionError("prediction returned no output — inference did not run")
    if not result.reasoning or not result.reasoning.strip():
        raise AssertionError("prediction returned empty reasoning — inference did not run")

    print(f"  ✓ inference ran ({infer_ms:.0f} ms)")
    print(f"      prediction_id: {result.prediction_id}")
    print(f"      output:        {result.output}  (is_spam={_extract_is_spam(result.output)})")
    print(f"      reasoning:     {result.reasoning.strip()[:240]}")

    # 3) Confidence — probe scoring runs on Modal; the gpt-oss fallback path
    #    can cold-boot, so wait with a generous deadline (longer than the
    #    .confidence property's fixed 300s).
    print("  waiting for confidence score (probe scoring; may cold-start)…")
    t1 = time.monotonic()
    status = client.wait_for_confidence_score(result.prediction_id, timeout=confidence_timeout)
    conf_ms = (time.monotonic() - t1) * 1000

    if status.status != "completed" or status.score is None:
        raise AssertionError(
            f"confidence scoring did not complete: status={status.status} error={status.error}"
        )
    if not (0.0 <= status.score <= 1.0):
        raise AssertionError(f"confidence not in [0, 1]: {status.score!r}")

    print(f"  ✓ confidence scored ({conf_ms:.0f} ms): {status.score:.4f}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--only",
        choices=[c.label for c in CASES],
        help="Run a single model instead of all.",
    )
    parser.add_argument("--api-url", default=os.getenv("MODAIC_API_URL", DEFAULT_API_URL))
    parser.add_argument("--git-url", default=os.getenv("MODAIC_GIT_URL", DEFAULT_GIT_URL))
    parser.add_argument("--namespace", default=os.getenv("MODAIC_NAMESPACE", DEFAULT_NAMESPACE))
    parser.add_argument(
        "--confidence-timeout",
        type=float,
        default=CONFIDENCE_TIMEOUT,
        help="Seconds to wait for confidence scoring before failing the step.",
    )
    args = parser.parse_args()

    token = os.getenv("DEV_MODAIC_TOKEN") or os.getenv("MODAIC_TOKEN")
    if not token:
        print("ERROR: set DEV_MODAIC_TOKEN (or MODAIC_TOKEN) to your dev hub token.", file=sys.stderr)
        return 2

    print("Modaic dev fallback-arbiter QA")
    print(f"  api_url:   {args.api_url}")
    print(f"  git_url:   {args.git_url}")
    print(f"  namespace: {args.namespace}")
    print(f"  token:     {_mask(token)}")
    if not (os.getenv("OPENAI_API_KEY") and os.getenv("ANTHROPIC_API_KEY")):
        print(
            "  note: OPENAI_API_KEY / ANTHROPIC_API_KEY not both set locally — that's "
            "fine if they're configured on the dev server / hub (where inference runs)."
        )

    # Point the SDK + client at dev. push/pull use git_url; predict uses api_url.
    modaic.configure(modaic_token=token, modaic_api_url=args.api_url, modaic_git_url=args.git_url)
    client = configure_modaic_client(modaic_token=token, base_url=args.api_url, timeout=CLIENT_TIMEOUT)

    cases = [c for c in CASES if (args.only is None or c.label == args.only)]
    results: dict[str, bool] = {}
    for case in cases:
        try:
            results[case.label] = run_case(client, case, args.namespace, args.confidence_timeout)
        except Exception as exc:
            results[case.label] = False
            print(f"  ✗ FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    print(f"\n{'=' * 72}\nSummary\n{'=' * 72}")
    for label, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
