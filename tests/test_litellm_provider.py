"""Regression tests for ``modaic/_litellm_provider.py``.

These are hermetic — they assert the litellm registry state that ``import modaic``
sets up, without making any network calls.
"""

import litellm
import modaic  # noqa: F401  (import side effect registers providers + model caps)


def test_modaic_provider_registered() -> None:
    """The ``modaic/`` provider is registered with litellm on import."""
    assert "modaic" in litellm.provider_list


def test_gateway_gpt_oss_supports_response_schema() -> None:
    """gpt-oss-120b on the Vercel AI Gateway is marked structured-output capable.

    Without this, dspy's ``JSONAdapter`` sees ``supports_response_schema is False``
    and falls back to ``response_format={"type": "json_object"}``, which the Vercel
    AI Gateway rejects with ``400 "Invalid input"`` for this model. Keeping the flag
    True holds dspy on the ``json_schema`` path the gateway accepts.

    Asserted at the litellm layer (where the fix lives and where dspy reads it),
    so it is robust across dspy versions. ``custom_llm_provider`` mirrors how dspy
    resolves the gateway provider before the litellm lookup.
    """
    assert (
        litellm.supports_response_schema(
            model="vercel_ai_gateway/openai/gpt-oss-120b",
            custom_llm_provider="vercel_ai_gateway",
        )
        is True
    )
