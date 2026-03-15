"""Redis-based request cache for deduplicating vLLM requests."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from typing import Any, Optional

import redis as _redis

logger = logging.getLogger(__name__)

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


def _make_key(request_body: dict[str, Any]) -> str:
    canonical: dict[str, Any] = {}
    for field in _SEMANTIC_FIELDS:
        if field in request_body:
            value = request_body[field]
            if field == "messages":
                value = _normalize_messages(value)
            canonical[field] = value
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=True).encode()
    return f"vllm:req:{hashlib.sha256(blob).hexdigest()}"


class RedisRequestCache:
    def __init__(self, host: str = "127.0.0.1", port: int = 6379) -> None:
        self._client = _redis.Redis(host=host, port=port, decode_responses=True)
        self._client.ping()
        logger.info("Connected to Redis at %s:%d", host, port)

    def get(self, request_body: dict[str, Any]) -> Optional[dict[str, Any]]:
        key = _make_key(request_body)
        raw = self._client.get(key)
        if raw is None:
            return None
        print(f"Cache hit for key: {key}")
        return json.loads(raw)

    def set(self, request_body: dict[str, Any], response: dict[str, Any]) -> None:
        key = _make_key(request_body)
        self._client.set(key, json.dumps(response, ensure_ascii=False))

    @staticmethod
    def start_server(data_dir: str = "/redis-data") -> subprocess.Popen:
        proc = subprocess.Popen(
            ["redis-server", "--dir", data_dir, "--appendonly", "yes", "--save", ""],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # Wait for Redis to be ready
        for _ in range(30):
            try:
                client = _redis.Redis()
                client.ping()
                client.close()
                logger.info("Redis server started (pid=%d, data_dir=%s)", proc.pid, data_dir)
                return proc
            except _redis.ConnectionError:
                time.sleep(0.5)
        raise RuntimeError("Redis server failed to start within 15 seconds")

    @staticmethod
    def stop_server(proc: subprocess.Popen) -> None:
        logger.info("Stopping Redis server (pid=%d)", proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
