from __future__ import annotations

import copy
import logging
import os
import time
import zlib
from pathlib import Path
from typing import Any

import cloudpickle
from dspy.clients.cache import Cache

logger = logging.getLogger(__name__)
_last_log = 0
_cache_hits = 0

LMDB_CACHE_ENV_VAR = "LMDB_DSPY_CACHE_DIR"
DEFAULT_LMDB_CACHE_DIR = os.path.expanduser("~/.dspy_lmdb_cache")


class LmdbLMCache(Cache):
    """DSPy-compatible cache backed by in-memory LRU + LMDB."""

    def __init__(
        self,
        path: str,
        *,
        enable_memory_cache: bool = True,
        memory_max_entries: int = 1_000_000,
        map_size: int = 10 * 1024 * 1024 * 1024,
        compress: bool = False,
        max_readers: int = 256,
    ) -> None:
        super().__init__(
            enable_disk_cache=False,
            enable_memory_cache=enable_memory_cache,
            disk_cache_dir="/tmp/unused",
            memory_max_entries=memory_max_entries,
        )

        self.compress = compress

        try:
            import lmdb
        except ImportError as exc:
            raise RuntimeError("lmdb package is required for LmdbLMCache") from exc

        self.env = lmdb.open(
            path, map_size=map_size, max_dbs=0, readahead=False, max_readers=max_readers
        )

    def _serialize(self, value: Any) -> bytes:
        payload = cloudpickle.dumps(value)
        return zlib.compress(payload) if self.compress else payload

    def _deserialize(self, payload: bytes) -> Any:
        raw = zlib.decompress(payload) if self.compress else payload
        return cloudpickle.loads(raw)

    def __contains__(self, key: str) -> bool:
        if self.enable_memory_cache and key in self.memory_cache:
            return True
        with self.env.begin() as txn:
            return txn.get(key.encode()) is not None

    def get(
        self,
        request: dict[str, Any],
        ignored_args_for_cache_key: list[str] | None = None,
    ) -> Any:
        try:
            key = self.cache_key(request, ignored_args_for_cache_key)
        except Exception:
            logger.debug("Failed to generate cache key for request")
            return None

        response = None
        if self.enable_memory_cache and key in self.memory_cache:
            with self._lock:
                response = self.memory_cache[key]
        else:
            with self.env.begin() as txn:
                payload = txn.get(key.encode())
            if payload is not None:
                response = self._deserialize(payload)
                if self.enable_memory_cache:
                    with self._lock:
                        self.memory_cache[key] = response

        if response is None:
            return None

        response = copy.deepcopy(response)
        if hasattr(response, "usage"):
            response.usage = {}
            response.cache_hit = True

        global _last_log, _cache_hits
        _cache_hits += 1
        now = time.time()
        if now - _last_log >= 1.0:
            _last_log = now
            logger.info("cache hit (total: %d)", _cache_hits)
            _cache_hits = 0
        return response

    def put(
        self,
        request: dict[str, Any],
        value: Any,
        ignored_args_for_cache_key: list[str] | None = None,
        enable_memory_cache: bool = True,
    ) -> None:
        write_memory = self.enable_memory_cache and enable_memory_cache

        try:
            key = self.cache_key(request, ignored_args_for_cache_key)
        except Exception:
            logger.debug("Failed to generate cache key for request")
            return

        if write_memory:
            with self._lock:
                self.memory_cache[key] = value

        try:
            payload = self._serialize(value)
            with self.env.begin(write=True) as txn:
                txn.put(key.encode(), payload)
        except Exception as exc:
            logger.debug("LMDB put failed for key=%s: %s", key, exc)


def use_lmdb_lm_cache(
    *,
    path: str | None = None,
    enable_memory_cache: bool = True,
    memory_max_entries: int = 1_000_000,
    map_size: int = 10 * 1024 * 1024 * 1024,
    compress: bool = False,
    max_readers: int = 256,
) -> None:
    """Install LMDB-backed cache into ``dspy.cache``.

    *path* defaults to ``$LMDB_DSPY_CACHE_DIR`` or ``~/.dspy_lmdb_cache``.
    """
    if path is None:
        path = os.getenv(LMDB_CACHE_ENV_VAR, DEFAULT_LMDB_CACHE_DIR)
    Path(path).mkdir(parents=True, exist_ok=True)
    cache = LmdbLMCache(
        path=path,
        enable_memory_cache=enable_memory_cache,
        memory_max_entries=memory_max_entries,
        map_size=map_size,
        compress=compress,
        max_readers=max_readers,
    )

    import dspy

    dspy.cache = cache
    logger.info("Configured DSPy LMDB cache backend path=%s", path)
