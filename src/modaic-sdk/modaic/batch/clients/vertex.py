from __future__ import annotations

from typing import Optional, Tuple

from ..types import BatchReponse, BatchRequest, ResultItem
from .base import BatchClient


class VertexAIBatchClient(BatchClient):
    provider: str = "vertex_ai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        poll_interval: float = 30.0,
        max_poll_time: str = "24h",
        status_callback=None,
    ):
        super().__init__(
            api_key=api_key,
            poll_interval=poll_interval,
            max_poll_time=max_poll_time,
            status_callback=status_callback,
        )

    def format(self, batch_request: BatchRequest) -> list[dict]:
        raise NotImplementedError("Vertex AI batch is not implemented yet")

    def parse(self, raw_result: dict[str, object]) -> ResultItem:
        raise NotImplementedError("Vertex AI batch is not implemented yet")

    async def _submit_batch_request(self, batch_request: BatchRequest) -> str:
        raise NotImplementedError("Vertex AI batch is not implemented yet")

    async def get_status(self, batch_id: str) -> Tuple[str, Optional[int]]:
        raise NotImplementedError("Vertex AI batch is not implemented yet")

    async def get_results(self, batch_id: str) -> BatchReponse:
        raise NotImplementedError("Vertex AI batch is not implemented yet")
