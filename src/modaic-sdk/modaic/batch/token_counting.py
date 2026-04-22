"""Per-provider token counters.

Each counter has the signature ``(model: str, messages: list[dict]) -> Optional[int]``.
Return ``None`` to signal "this provider does not care about token limits" — callers
skip token tracking entirely in that case.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

TokenCounter = Callable[[str, list[dict[str, Any]]], Optional[int]]


def _stringify_message(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif block.get("type") == "text" and isinstance(block.get("content"), str):
                    parts.append(block["content"])
        return "".join(parts)
    return "" if content is None else str(content)


def _flatten_messages(messages: list[dict[str, Any]]) -> str:
    return "\n".join(_stringify_message(m) for m in messages)


@lru_cache(maxsize=64)
def _tiktoken_encoding(model: str):
    try:
        import tiktoken
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tiktoken is required for OpenAI/Azure token counting. "
            'Install it with `uv add "modaic[openai]"`.'
        ) from exc

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens_tiktoken(model: str, messages: list[dict[str, Any]]) -> int:
    """Count tokens for OpenAI/Azure models using tiktoken."""
    encoding = _tiktoken_encoding(model)
    total = 0
    for message in messages:
        total += len(encoding.encode(_stringify_message(message)))
        role = message.get("role")
        if isinstance(role, str):
            total += len(encoding.encode(role))
        # Per-message overhead per OpenAI's documented estimator.
        total += 4
    total += 2  # priming tokens for assistant reply
    return total


@lru_cache(maxsize=16)
def _hf_tokenizer(model: str):
    try:
        from tokenizers import Tokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tokenizers is required for HF-based token counting. "
            'Install it with `uv add "modaic[together]"` (or any HF-backed extra).'
        ) from exc

    repo = model.split("/", 1)[-1] if "/" in model else model
    try:
        return Tokenizer.from_pretrained(repo)
    except Exception:
        logger.warning("tokenizers.from_pretrained(%s) failed; falling back to full model string", repo)
        return Tokenizer.from_pretrained(model)


def count_tokens_hf(model: str, messages: list[dict[str, Any]]) -> int:
    """Count tokens for OSS/HF models using the ``tokenizers`` library."""
    tokenizer = _hf_tokenizer(model)
    encoded = tokenizer.encode(_flatten_messages(messages))
    return len(encoded.ids)


def count_tokens_anthropic(model: str, messages: list[dict[str, Any]]) -> int:
    """Approximate token count for Anthropic models using the SDK's local counter."""
    try:
        import anthropic
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "anthropic is required for Anthropic token counting. "
            'Install it with `uv add "modaic[anthropic]"`.'
        ) from exc

    local_count = getattr(anthropic, "count_tokens", None)
    text = _flatten_messages(messages)
    if callable(local_count):
        try:
            return int(local_count(text))
        except Exception:
            logger.debug("anthropic.count_tokens failed; falling back to tiktoken estimate", exc_info=True)

    return count_tokens_tiktoken("gpt-4", messages)


def count_tokens_none(model: str, messages: list[dict[str, Any]]) -> None:
    """No-op counter for providers that don't track token limits (e.g. vllm)."""
    return None
