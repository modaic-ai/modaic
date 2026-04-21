"""Modal app used by batch integration tests — runs VLLMBatchClient on a Modal H100.

Image config follows the /modal skill's vllm.md recipe (CUDA 12.8 + Python 3.11 +
flashinfer + hf_transfer + hf-xet) and adds the local modaic workspace so
VLLMBatchClient and its deps (dspy, duckdb, datasets) are importable inside the
container.

Tests import `app` and `run_vllm_batch` from this module.
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Any

import modal

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

app = modal.App("modaic-batch-integration-tests")

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
torch_cache_vol = modal.Volume.from_name("torch-cache", create_if_missing=True)

# Canonical vLLM image from the /modal skill (vllm.md), with modaic workspace
# source added so VLLMBatchClient is available inside the container.
# `copy=True` on add_local_dir so the subsequent pip install layer sees the files.
# SETUPTOOLS_SCM_PRETEND_VERSION bypasses hatch-vcs git detection (no .git in container).
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .uv_pip_install(
        "vllm>=0.18.0",
        "flashinfer-python",
        "hf_transfer",
        "hf-xet>=1.1.7",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TQDM_DISABLE": "1",
        "TRANSFORMERS_VERBOSITY": "warning",
        "SETUPTOOLS_SCM_PRETEND_VERSION": "0.0.0",
    })
    .add_local_dir(str(_PROJECT_ROOT / "src" / "modaic-client"), "/pkg/modaic-client", copy=True)
    .add_local_dir(str(_PROJECT_ROOT / "src" / "modaic-sdk"), "/pkg/modaic-sdk", copy=True)
    .run_commands(
        "pip install hatchling hatch-vcs && "
        "pip install --no-build-isolation /pkg/modaic-client && "
        "pip install --no-build-isolation '/pkg/modaic-sdk[vllm]'"
    )
)

VOLUMES = {
    "/root/.cache/huggingface": hf_cache_vol,
    "/root/.cache/torch": torch_cache_vol,
}

VALID_LABELS = {"1 star", "2 stars", "3 stars", "4 stars", "5 stars"}


@app.function(
    gpu="H100",
    image=vllm_image,
    timeout=60 * 60 * 24,
    cpu=8,
    volumes=VOLUMES,
)
async def run_vllm_batch(
    model_id: str,
    batch_size: int,
    n_rows: int,
    enforce_eager: bool = False,
) -> dict[str, Any]:
    """Run VLLMBatchClient on `n_rows` Yelp reviews, sharded into `batch_size` chunks.

    Loads the dataset from HuggingFace inside the container. Runs with
    return_messages=True and validates per-row that each prediction has a
    valid star_rating, non-empty messages list, non-empty outputs["text"],
    and non-empty outputs["reasoning_content"].

    Returns counts the caller asserts on: total, successes, shards,
    valid_labels, nonempty_messages, nonempty_text, nonempty_reasoning.
    """
    import dspy
    from datasets import load_dataset

    from modaic import Predict
    from modaic.batch import abatch
    from modaic.batch.clients.vllm import VLLMBatchClient
    from modaic.batch.types import FailedPrediction

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    ds = load_dataset("Yelp/yelp_review_full", split="train")
    inputs = [
        {"review_text": r["text"]}
        for r in itertools.islice(itertools.cycle(ds), n_rows)
    ]

    predictor = Predict(
        "review_text -> star_rating: Literal['1 star', '2 stars', '3 stars', '4 stars', '5 stars']"
    )
    lm = dspy.LM(f"huggingface/{model_id}")
    client = VLLMBatchClient(lm=lm, batch_size=batch_size, enforce_eager=enforce_eager)

    async with client.session():
        with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
            grouped = await abatch(
                [(predictor, inputs)],
                client=client,
                mode="sequential",
                show_progress=False,
                return_messages=True,
            )

    _, result = grouped[0]
    successes = 0
    valid_labels = 0
    nonempty_messages = 0
    nonempty_text = 0
    nonempty_reasoning = 0
    fully_successful = 0
    for row in result:
        if isinstance(row.prediction, FailedPrediction):
            continue
        successes += 1
        has_valid_label = getattr(row.prediction, "star_rating", None) in VALID_LABELS
        has_messages = isinstance(row.messages, list) and len(row.messages) > 0
        outputs = row.outputs or {}
        text = outputs.get("text") or ""
        reasoning = outputs.get("reasoning_content") or ""
        has_text = len(text) > 0
        has_reasoning = len(reasoning) > 0
        if has_valid_label:
            valid_labels += 1
        if has_messages:
            nonempty_messages += 1
        if has_text:
            nonempty_text += 1
        if has_reasoning:
            nonempty_reasoning += 1
        if has_valid_label and has_messages and has_text and has_reasoning:
            fully_successful += 1

    shards = (n_rows + batch_size - 1) // batch_size
    return {
        "total": len(result),
        "successes": successes,
        "shards": shards,
        "valid_labels": valid_labels,
        "nonempty_messages": nonempty_messages,
        "nonempty_text": nonempty_text,
        "nonempty_reasoning": nonempty_reasoning,
        "fully_successful": fully_successful,
    }
