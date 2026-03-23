"""LMDB-backed request cache for deduplicating vLLM requests."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import lmdb

logger = logging.getLogger(__name__)

DEFAULT_MAP_SIZE_BYTES = 32 * 1024**3
DEFAULT_MAX_READERS = 512
DEFAULT_MAX_SPARE_TXNS = 64
DEFAULT_CACHE_PATH_ENV = "MODAIC_LMDB_CACHE_PATH"
DEFAULT_MAP_SIZE_ENV = "MODAIC_LMDB_MAP_SIZE_BYTES"

_SEMANTIC_FIELDS = (
    "model",
    "messages",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "max_completion_tokens",
    "repetition_penalty",
)


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for msg in messages:
        norm = dict(msg)
        if isinstance(norm.get("content"), str):
            norm["content"] = norm["content"].strip()
        normalized.append(norm)
    return normalized


def _make_key(request_body: dict[str, Any]) -> bytes:
    canonical: dict[str, Any] = {}
    for field in _SEMANTIC_FIELDS:
        if field in request_body:
            value = request_body[field]
            if field == "messages":
                value = _normalize_messages(value)
            canonical[field] = value
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return f"vllm:req:{hashlib.sha256(blob).hexdigest()}".encode("ascii")


def _default_map_size() -> int:
    raw_value = os.environ.get(DEFAULT_MAP_SIZE_ENV)
    if raw_value is None:
        return DEFAULT_MAP_SIZE_BYTES
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{DEFAULT_MAP_SIZE_ENV} must be an integer number of bytes, got: {raw_value!r}") from exc


class LMDBRequestCache:
    def __init__(
        self,
        path: str | Path | None = None,
        *,
        map_size: Optional[int] = None,
        max_readers: int = DEFAULT_MAX_READERS,
    ) -> None:
        resolved_path = Path(path or os.environ.get(DEFAULT_CACHE_PATH_ENV, ".modaic-vllm-cache")).expanduser()
        resolved_path.mkdir(parents=True, exist_ok=True)
        self.path = resolved_path.resolve()
        self._env = lmdb.open(
            str(self.path),
            create=True,
            subdir=True,
            lock=True,
            readonly=False,
            readahead=False,
            writemap=False,
            map_async=False,
            sync=False,
            metasync=False,
            max_readers=max_readers,
            max_spare_txns=min(max_readers, DEFAULT_MAX_SPARE_TXNS),
            map_size=map_size or _default_map_size(),
        )
        logger.info("Opened LMDB request cache at %s", self.path)

    def close(self) -> None:
        self._env.close()

    def get(self, request_body: dict[str, Any]) -> Optional[dict[str, Any]]:
        key = _make_key(request_body)
        with self._env.begin() as txn:
            raw = txn.get(key)
        if raw is None:
            return None
        logger.debug("Cache hit for key: %s", key.decode("ascii"))
        return json.loads(raw.decode("utf-8"))

    def set(self, request_body: dict[str, Any], response: dict[str, Any]) -> None:
        key = _make_key(request_body)
        value = json.dumps(response, ensure_ascii=False).encode("utf-8")
        try:
            with self._env.begin(write=True) as txn:
                txn.put(key, value)
        except lmdb.MapFullError as exc:
            raise RuntimeError(
                f"LMDB cache at {self.path} exceeded its configured map_size. "
                f"Increase {DEFAULT_MAP_SIZE_ENV} or pass a larger map_size."
            ) from exc
