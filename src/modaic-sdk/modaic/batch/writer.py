from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .clients.base import BatchClient

logger = logging.getLogger(__name__)


class JSONLShardWriter:
    """Streams JSONL lines into one or more shards, rolling over when the
    next line would exceed the client's request, byte, or token cap.
    """

    def __init__(
        self,
        client: BatchClient,
        base_path: Path,
        *,
        token_cap: Optional[int] = None,
    ):
        self.client = client
        self._token_cap_override = token_cap
        self.base_path = Path(base_path)
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        self.shards: list[Path] = []
        self.shard_req_counts: list[int] = []
        self.shard_token_counts: list[Optional[int]] = []
        self._current_path: Optional[Path] = None
        self._current_file = None
        self._current_n = 0
        self._current_bytes = 0
        self._current_tokens: Optional[int] = None

    @property
    def token_cap(self) -> Optional[int]:
        if self._token_cap_override is not None:
            return self._token_cap_override
        return self.client.token_cap

    def add(self, line_obj: dict[str, Any], n_tokens: Optional[int] = None) -> None:
        line = json.dumps(line_obj) + "\n"
        size = len(line.encode("utf-8"))
        if size > self.client.max_file_size:
            raise ValueError(
                f"single request is {size}B which exceeds client {self.client.name} "
                f"max_file_size={self.client.max_file_size}"
            )
        if self._current_file is None or self._would_exceed(size, n_tokens):
            self._roll()
        self._current_file.write(line)
        self._current_n += 1
        self._current_bytes += size
        if n_tokens is not None:
            self._current_tokens = (self._current_tokens or 0) + n_tokens

    def _would_exceed(self, size: int, n_tokens: Optional[int]) -> bool:
        if self._current_bytes + size > self.client.byte_cap:
            return True
        if self._current_n + 1 > self.client.request_cap:
            return True
        token_cap = self.token_cap
        if n_tokens is not None and token_cap is not None:
            current = self._current_tokens or 0
            if current + n_tokens > token_cap:
                return True
        return False

    def _roll(self) -> None:
        self._close_current_shard()
        index = len(self.shards)
        path = self.base_path.with_suffix(f".{index:04d}.jsonl")
        self._current_path = path
        self._current_file = path.open("w", encoding="utf-8")
        self._current_n = 0
        self._current_bytes = 0
        self._current_tokens = None
        self.shards.append(path)
        self.shard_req_counts.append(0)
        self.shard_token_counts.append(None)

    def _close_current_shard(self) -> None:
        if self._current_file is None:
            return
        self._current_file.close()
        self._current_file = None
        # Record final counts for the shard we just closed.
        self.shard_req_counts[-1] = self._current_n
        self.shard_token_counts[-1] = self._current_tokens

    def finalize(self) -> list[Path]:
        self._close_current_shard()
        logger.debug(
            "JSONLShardWriter finalized: client=%s shards=%d total_bytes≈%d",
            self.client.name,
            len(self.shards),
            sum(p.stat().st_size for p in self.shards),
        )
        return list(self.shards)

    def cleanup(self) -> None:
        for p in self.shards:
            p.unlink(missing_ok=True)
