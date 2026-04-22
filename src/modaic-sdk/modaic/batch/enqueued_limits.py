"""Per-provider enqueued-scope limit lookup.

Each lookup fn takes a model string and returns an :class:`EnqueuedLimits`
describing how much a user may have in flight at one time across active
batch jobs. ``None`` fields mean "no limit known".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional


def _strip_provider_prefix(model: str) -> str:
    """Return the provider-less portion of a litellm model string.

    ``"openai/gpt-4o-mini"`` → ``"gpt-4o-mini"``. Falls back to ``model`` if
    litellm cannot parse it (e.g. a bare model name).
    """
    try:
        from litellm import get_llm_provider

        return get_llm_provider(model)[0]
    except Exception:
        return model.split("/", 1)[-1] if "/" in model else model


@dataclass(frozen=True)
class EnqueuedLimits:
    max_enqueued_reqs: Optional[int] = None
    max_enqueued_tokens: Optional[int] = None
    max_enqueued_jobs: Optional[int] = None


EnqueuedLimitsFn = Callable[[str], EnqueuedLimits]


_OPENAI_TPD: dict[str, int] = {
    # gpt-5 family
    "gpt-5": 1_500_000,
    "gpt-5-chat-latest": 900_000,
    "gpt-5-codex": 900_000,
    "gpt-5-mini": 5_000_000,
    "gpt-5-nano": 2_000_000,
    "gpt-5-pro": 90_000,
    "gpt-5.1": 900_000,
    "gpt-5.1-chat-latest": 900_000,
    "gpt-5.1-codex": 900_000,
    "gpt-5.1-codex-max": 900_000,
    "gpt-5.1-codex-mini": 2_000_000,
    "gpt-5.2": 900_000,
    "gpt-5.2-chat-latest": 900_000,
    "gpt-5.2-codex": 900_000,
    "gpt-5.2-pro": 900_000,
    "gpt-5.3-chat-latest": 900_000,
    "gpt-5.3-codex": 900_000,
    "gpt-5.4": 900_000,
    "gpt-5.4-long-context": 2_000_000,
    "gpt-5.4-mini": 2_000_000,
    "gpt-5.4-nano": 2_000_000,
    "gpt-5.4-pro": 900_000,
    "gpt-5.4-pro-long-context": 2_000_000,
    # gpt-4.1 family
    "gpt-4.1": 900_000,
    "gpt-4.1-long-context": 2_000_000,
    "gpt-4.1-mini": 2_000_000,
    "gpt-4.1-mini-long-context": 4_000_000,
    "gpt-4.1-nano": 2_000_000,
    "gpt-4.1-nano-long-context": 4_000_000,
    # gpt-4o family
    "gpt-4o": 90_000,
    "gpt-4o-mini": 2_000_000,
    # legacy
    "gpt-4": 100_000,
    "gpt-4-turbo": 90_000,
    "gpt-3.5-turbo": 2_000_000,
    "gpt-3.5-turbo-16k": 2_000_000,
    "gpt-3.5-turbo-instruct": 200_000,
}

_DATE_SUFFIX_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def _openai_tpd_lookup(model: str) -> int:
    model = _strip_provider_prefix(model)
    if model in _OPENAI_TPD:
        return _OPENAI_TPD[model]
    stripped = _DATE_SUFFIX_RE.sub("", model)
    if stripped in _OPENAI_TPD:
        return _OPENAI_TPD[stripped]
    # family prefix: longest known key that is a prefix of the model name
    best: Optional[str] = None
    for key in _OPENAI_TPD:
        if model.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    if best is not None:
        return _OPENAI_TPD[best]
    return 90_000


def openai_enqueued_limits(model: str) -> EnqueuedLimits:
    return EnqueuedLimits(max_enqueued_tokens=_openai_tpd_lookup(model))


# Azure Default-tier Enqueued Tokens Per Day; see
# https://learn.microsoft.com/azure/ai-services/openai/quotas-limits
_AZURE_DEFAULT_ETPD: dict[str, int] = {
    "gpt-4.1": 200_000_000,
    "gpt-4.1-mini": 1_000_000_000,
    "gpt-4.1-nano": 1_000_000_000,
    "gpt-4o": 200_000_000,
    "gpt-4o-mini": 1_000_000_000,
    "gpt-4-turbo": 80_000_000,
    "gpt-4": 30_000_000,
    "o3-mini": 1_000_000_000,
    "o4-mini": 1_000_000_000,
    "gpt-5": 200_000_000,
    "gpt-5.1": 200_000_000,
}


def _azure_etpd_lookup(model: str) -> Optional[int]:
    model = _strip_provider_prefix(model)
    if model in _AZURE_DEFAULT_ETPD:
        return _AZURE_DEFAULT_ETPD[model]
    stripped = _DATE_SUFFIX_RE.sub("", model)
    if stripped in _AZURE_DEFAULT_ETPD:
        return _AZURE_DEFAULT_ETPD[stripped]
    best: Optional[str] = None
    for key in _AZURE_DEFAULT_ETPD:
        if model.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    return _AZURE_DEFAULT_ETPD[best] if best is not None else None


def azure_enqueued_limits(model: str) -> EnqueuedLimits:
    return EnqueuedLimits(max_enqueued_tokens=_azure_etpd_lookup(model))


def anthropic_enqueued_limits(model: str) -> EnqueuedLimits:
    del model
    return EnqueuedLimits(max_enqueued_reqs=100_000)


def together_enqueued_limits(model: str) -> EnqueuedLimits:
    del model
    return EnqueuedLimits(max_enqueued_tokens=30_000_000_000)


def fireworks_enqueued_limits(model: str) -> EnqueuedLimits:
    del model
    return EnqueuedLimits()


def vertex_enqueued_limits(model: str) -> EnqueuedLimits:
    del model
    return EnqueuedLimits()


def none_enqueued_limits(model: str) -> EnqueuedLimits:
    del model
    return EnqueuedLimits()
