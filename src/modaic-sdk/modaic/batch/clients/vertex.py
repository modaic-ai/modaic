from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .._experimental import experimental
from ..enqueued_limits import vertex_enqueued_limits
from ..token_counting import count_tokens_hf
from ..types import RawResults, ResultItem
from .base import RemoteBatchClient


@experimental
class VertexAIBatchClient(RemoteBatchClient):
    name = "vertex"
    reqs_per_file = 200_000
    max_file_size = 1024 * 1024 * 1024
    endpoint = None
    token_counter = staticmethod(count_tokens_hf)
    enqueued_limits_fn = staticmethod(vertex_enqueued_limits)

    def __init__(
        self,
        api_key: Optional[str] = None,
        poll_interval: float = 30.0,
        max_poll_time: str = "24h",
        *,
        reqs_per_file: Optional[int] = None,
        max_file_size: Optional[int] = None,
        tokens_per_file: Optional[int] = None,
        default_enqueued_reqs: Optional[int] = None,
        default_enqueued_tokens: Optional[int] = None,
        default_enqueued_jobs: Optional[int] = None,
        enable_concurrent_jobs: Optional[bool] = None,
    ):
        super().__init__(
            api_key=api_key,
            poll_interval=poll_interval,
            max_poll_time=max_poll_time,
            reqs_per_file=reqs_per_file,
            max_file_size=max_file_size,
            tokens_per_file=tokens_per_file,
            default_enqueued_reqs=default_enqueued_reqs,
            default_enqueued_tokens=default_enqueued_tokens,
            default_enqueued_jobs=default_enqueued_jobs,
            enable_concurrent_jobs=enable_concurrent_jobs,
        )

    def parse_result(self, raw: dict[str, Any]) -> ResultItem:
        raise NotImplementedError("Vertex AI batch is not implemented yet")

    async def create_batch(self, shard: Path) -> str:
        raise NotImplementedError("Vertex AI batch is not implemented yet")

    async def poll_status(self, batch_id: str) -> tuple[str, Optional[int]]:
        raise NotImplementedError("Vertex AI batch is not implemented yet")

    async def fetch_results(self, batch_id: str) -> RawResults:
        raise NotImplementedError("Vertex AI batch is not implemented yet")

    async def cancel(self, batch_id: str) -> bool:
        raise NotImplementedError("Vertex AI batch is not implemented yet")
