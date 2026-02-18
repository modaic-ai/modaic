"""
Simple example of batch inference with Anthropic using AnthropicBatchClient.

This example demonstrates two approaches:
1. Using AnthropicBatchClient directly with raw requests
2. Using the high-level abatch() function with dspy

Requires:
- ANTHROPIC_API_KEY environment variable set
"""

import asyncio
from logging import getLogger

from dotenv import load_dotenv

logger = getLogger(__name__)

load_dotenv()


async def direct_client_example():
    """Example using AnthropicBatchClient directly with raw batch requests."""
    from modaic.batch.clients import AnthropicBatchClient
    from modaic.batch.types import BatchRequest, BatchRequestItem

    # Create the client
    client = AnthropicBatchClient(
        poll_interval=10.0,  # Check status every 10 seconds
        max_poll_time="1h",  # Wait up to 1 hour
    )

    # Create a batch request with raw messages
    batch_request = BatchRequest(
        requests=[
            BatchRequestItem(
                id="request-0",
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "What is 2 + 2? Answer briefly."}],
                lm_kwargs={"max_tokens": 100},
            ),
            BatchRequestItem(
                id="request-1",
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "What is the capital of France? Answer briefly."}],
                lm_kwargs={"max_tokens": 100},
            ),
            BatchRequestItem(
                id="request-2",
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "Name a primary color. Answer with one word."}],
                lm_kwargs={"max_tokens": 100},
            ),
        ],
        model="claude-sonnet-4-20250514",
        lm_kwargs={"max_tokens": 100},
    )

    logger.info("Submitting batch to Anthropic...")
    result = await client.submit_and_wait(batch_request, show_progress=True)

    logger.info(f"\nBatch {result.batch_id} completed with status: {result.status}")
    logger.info(f"Got {len(result.results)} results\n")

    # Parse and display results
    for raw_result in result.results:
        custom_id = raw_result.get("custom_id", "unknown")
        parsed = client.parse(raw_result)
        logger.info(f"[{custom_id}] {parsed['text']}")


async def dspy_example():
    """Example using the high-level abatch() function with dspy predictor."""
    import dspy
    from modaic.batch import abatch

    # Configure dspy with Anthropic
    dspy.configure(lm=dspy.LM("anthropic/claude-sonnet-4-20250514"))

    # Create a simple predictor
    predictor = dspy.Predict("question -> answer")

    # Prepare inputs
    inputs = [
        {"question": "What is 2 + 2?"},
        {"question": "What is the capital of France?"},
        {"question": "Name a primary color."},
    ]

    logger.info("Submitting batch via dspy...")
    predictions = await abatch(
        predictor,
        inputs,
        show_progress=True,
        poll_interval=10.0,
        max_poll_time="1h",
    )

    logger.info(f"\nGot {len(predictions)} predictions\n")
    for i, pred in enumerate(predictions):
        logger.info(f"[{i}] Q: {inputs[i]['question']}")
        logger.info(f"    A: {pred.answer}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "dspy":
        asyncio.run(dspy_example())
    else:
        asyncio.run(direct_client_example())
