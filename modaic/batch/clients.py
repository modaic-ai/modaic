import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any, Optional

from ..constants import BATCH_DIR
from .types import BatchRequest, BatchResult, ResultItem

CLEANUP = True


def _parse_time_string(time_str: str) -> float:
    """
    Parse a time string like "30s", "5m", or "24h" into seconds.

    Args:
        time_str: A string with a number followed by 's' (seconds), 'm' (minutes), or 'h' (hours).

    Returns:
        The time in seconds as a float.

    Raises:
        ValueError: If the time string format is invalid.
    """
    import re

    match = re.match(r"^(\d+(?:\.\d+)?)\s*([smh])$", time_str.strip().lower())
    if not match:
        raise ValueError(f"Invalid time format: '{time_str}'. Expected format like '30s', '5m', or '24h'.")

    value = float(match.group(1))
    unit = match.group(2)

    if unit == "s":
        return value
    elif unit == "m":
        return value * 60
    elif unit == "h":
        return value * 3600
    else:
        raise ValueError(f"Unknown time unit: '{unit}'")


class BatchClient:
    """Base class for batch processing clients."""
    provider: str

    def __init__(
        self,
        api_key: Optional[str] = None,
        poll_interval: float = 30.0,
        max_poll_time: str = "24h",
    ):
        """
        Initialize a batch client.

        Args:
            provider: The provider name (e.g., "openai", "together_ai", "fireworks_ai", "azure").
            api_key: Optional API key for the provider. If not provided, will use environment variables.
            poll_interval: Seconds between status checks (default: 30.0).
            max_poll_time: Maximum time to wait for completion as a string like "30s", "5m", or "24h" (default: "24h").
        """
        self.api_key = api_key
        self.poll_interval = poll_interval
        self.max_poll_time = max_poll_time
        self.max_poll_time_s = _parse_time_string(max_poll_time)

    def format(self, batch_request: BatchRequest) -> list[dict]:
        """Format batch request into provider-specific JSONL format."""
        raise NotImplementedError("format is not implemented")

    def parse(self, raw_result: dict[str, Any]) -> ResultItem:
        """
        Parse a raw result line from the batch response into a ResultItem.

        Each provider has a different response format. Subclasses must implement this
        to extract the text (and optionally logprobs/tool_calls) from their response format.

        Args:
            raw_result: A single result dict from the batch response JSONL.

        Returns:
            A ResultItem with at least the 'text' field populated.
        """
        raise NotImplementedError("parse is not implemented")

    def create_jsonl(self, batch_request: BatchRequest, path: Optional[Path] = None) -> Path:
        """Create a JSONL file from batch request."""
        if path is None:
            BATCH_DIR.mkdir(parents=True, exist_ok=True)
            path = BATCH_DIR / f"batch_{id(batch_request)}.jsonl"
        formatted = self.format(batch_request)
        with open(path, "w") as f:
            for item in formatted:
                f.write(json.dumps(item) + "\n")
        return path

    async def _submit_batch_request(self, batch_request: BatchRequest) -> str:
        """
        Internal method to submit a batch request. Override in subclasses.

        Args:
            batch_request: The batch request to submit.

        Returns:
            The batch ID for tracking the job.
        """
        raise NotImplementedError("_submit_batch_request is not implemented")

    async def submit(self, batch_request: BatchRequest) -> str:
        """
        Submit a batch job and return the batch ID.

        Args:
            batch_request: The batch request to submit.

        Returns:
            The batch ID for tracking the job.
        """
        return await self._submit_batch_request(batch_request)

    async def get_status(self, batch_id: str) -> str:
        """
        Get the status of a batch job.

        Args:
            batch_id: The batch ID to check.

        Returns:
            Status string (e.g., "completed", "in_progress", "failed").
        """
        raise NotImplementedError("get_status is not implemented")

    async def get_results(self, batch_id: str) -> BatchResult:
        """
        Get the results of a completed batch job.

        Args:
            batch_id: The batch ID to get results for.

        Returns:
            BatchResult containing the results.
        """
        raise NotImplementedError("get_results is not implemented")

    async def cancel(self, batch_id: str) -> bool:
        """
        Cancel a batch job.

        Args:
            batch_id: The batch ID to cancel.

        Returns:
            True if cancellation was successful.
        """
        raise NotImplementedError("cancel is not implemented")

    async def submit_and_wait(
        self,
        batch_request: BatchRequest,
        show_progress: bool = True,
    ) -> BatchResult:
        """
        Submit a batch job and wait for completion.

        Args:
            batch_request: The batch request to submit.
            show_progress: Whether to show a progress spinner in the terminal.

        Returns:
            BatchResult containing raw results from the API.

        Raises:
            TimeoutError: If the job doesn't complete within max_poll_time.
            RuntimeError: If the job fails.
        """
        num_requests = len(batch_request["requests"])

        if show_progress:
            return await self._submit_and_wait_with_progress(batch_request, num_requests)
        else:
            return await self._submit_and_wait_silent(batch_request)

    async def _submit_and_wait_silent(self, batch_request: BatchRequest) -> BatchResult:
        """Submit and wait without progress display."""
        batch_id = await self._submit_batch_request(batch_request)
        waited = 0.0

        while waited < self.max_poll_time_s:
            status = await self.get_status(batch_id)

            if status in ("completed", "COMPLETED"):
                return await self.get_results(batch_id)
            elif status in ("failed", "FAILED", "cancelled", "CANCELLED", "expired", "EXPIRED"):
                raise RuntimeError(f"Batch job {batch_id} failed with status: {status}")

            await asyncio.sleep(self.poll_interval)
            waited += self.poll_interval

        raise TimeoutError(f"Batch job {batch_id} did not complete within {self.max_poll_time}")

    async def _submit_and_wait_with_progress(
        self,
        batch_request: BatchRequest,
        num_requests: int,
    ) -> BatchResult:
        """Submit and wait with a nice Rich progress display."""
        try:
            import time

            from rich.console import Console, Group
            from rich.live import Live
            from rich.panel import Panel
            from rich.spinner import Spinner
            from rich.table import Table
            from rich.text import Text
        except ImportError:
            # Fall back to silent mode if rich is not installed
            return await self._submit_and_wait_silent(batch_request)

        console = Console()
        batch_id: Optional[str] = None
        status = "submitting"
        start_time = time.time()

        def format_elapsed(seconds: float) -> str:
            """Format elapsed time nicely."""
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                mins, secs = divmod(int(seconds), 60)
                return f"{mins}m {secs}s"
            else:
                hours, remainder = divmod(int(seconds), 3600)
                mins, secs = divmod(remainder, 60)
                return f"{hours}h {mins}m {secs}s"

        def get_status_style(s: str) -> tuple[str, str]:
            """Get color and emoji for status."""
            s_lower = s.lower()
            if s_lower == "completed":
                return "green", "[green]completed[/green]"
            elif s_lower in ("failed", "cancelled", "expired"):
                return "red", f"[red]{s_lower}[/red]"
            elif s_lower in ("in_progress", "running", "validating", "pending"):
                return "yellow", f"[yellow]{s_lower}[/yellow]"
            else:
                return "yellow", f"[yellow]{s}[/yellow]"

        def make_display() -> Panel:
            """Create the progress display panel."""
            elapsed = time.time() - start_time
            _, status_styled = get_status_style(status)

            table = Table.grid(padding=(0, 2))
            table.add_column(style="cyan", justify="right")
            table.add_column(style="white")

            table.add_row("Batch ID:", batch_id or "[dim]submitting...[/dim]")
            table.add_row("Provider:", f"[magenta]{self.provider}[/magenta]")
            table.add_row("Requests:", f"[bold]{num_requests}[/bold]")
            table.add_row("Status:", status_styled)
            table.add_row("Elapsed:", f"[dim]{format_elapsed(elapsed)}[/dim]")

            # Add spinner for in-progress states
            if status.lower() not in ("completed", "failed", "cancelled", "expired"):
                spinner = Spinner("dots", text=Text(" Processing...", style="dim"))
                content = Group(table, Text(""), spinner)
            else:
                content = table

            return Panel(
                content,
                title="[bold blue]Batch Processing[/bold blue]",
                border_style="blue",
                padding=(1, 2),
            )

        with Live(make_display(), console=console, refresh_per_second=4) as live:
            # Submit the batch
            batch_id = await self._submit_batch_request(batch_request)
            status = "submitted"
            live.update(make_display())

            waited = 0.0
            while waited < self.max_poll_time_s:
                status = await self.get_status(batch_id)
                live.update(make_display())

                if status in ("completed", "COMPLETED"):
                    status = "completed"
                    live.update(make_display())
                    result = await self.get_results(batch_id)
                    return result

                elif status in ("failed", "FAILED", "cancelled", "CANCELLED", "expired", "EXPIRED"):
                    raise RuntimeError(f"Batch job {batch_id} failed with status: {status}")

                await asyncio.sleep(self.poll_interval)
                waited += self.poll_interval

        raise TimeoutError(f"Batch job {batch_id} did not complete within {self.max_poll_time}")


class OpenAIBatchClient(BatchClient):
    """
    Batch client for OpenAI using the OpenAI SDK directly.

    Uses the AsyncOpenAI client for batch operations.
    """
    provider: str = "openai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        poll_interval: float = 30.0,
        max_poll_time: str = "24h",
    ):
        from openai import AsyncOpenAI

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        super().__init__(
            api_key=resolved_api_key,
            poll_interval=poll_interval,
            max_poll_time=max_poll_time,
        )
        self._client = AsyncOpenAI(api_key=resolved_api_key)
        self._file_ids: dict[str, str] = {}  # batch_id -> input_file_id mapping

    def format(self, batch_request: BatchRequest) -> list[dict]:
        return [
            {
                "custom_id": request["id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": request["model"], "messages": request["messages"], **request["lm_kwargs"]},
            }
            for request in batch_request["requests"]
        ]

    def parse(self, raw_result: dict[str, Any]) -> ResultItem:
        """
        Parse OpenAI batch result into ResultItem.

        OpenAI format: {"custom_id": "...", "response": {"body": {"choices": [{"message": {"content": "..."}}]}}}
        """
        response = raw_result.get("response", {})
        body = response.get("body", {})
        choices = body.get("choices", [])

        if not choices:
            error = raw_result.get("error") or body.get("error", {})
            raise ValueError(f"Batch request failed: {error}")

        choice = choices[0]
        message = choice.get("message", {})

        result: ResultItem = {"text": message.get("content", "")}
        if "logprobs" in choice and choice["logprobs"] is not None:
            result["logprobs"] = choice["logprobs"]
        if "tool_calls" in message and message["tool_calls"] is not None:
            result["tool_calls"] = message["tool_calls"]

        return result

    async def _submit_batch_request(self, batch_request: BatchRequest) -> str:
        """Submit a batch job to OpenAI."""
        # Create temp JSONL file
        BATCH_DIR.mkdir(parents=True, exist_ok=True)
        jsonl_path = self.create_jsonl(batch_request)

        try:
            # Upload file
            with open(jsonl_path, "rb") as f:
                file_obj = await self._client.files.create(
                    file=f,
                    purpose="batch",
                )

            # Create batch
            batch = await self._client.batches.create(
                completion_window="24h",
                endpoint="/v1/chat/completions",
                input_file_id=file_obj.id,
            )

            # Store file_id for later cleanup if needed
            self._file_ids[batch.id] = file_obj.id

            return batch.id
        finally:
            # Clean up temp file
            if CLEANUP and jsonl_path.exists():
                jsonl_path.unlink()

    async def get_status(self, batch_id: str) -> str:
        """Get status of an OpenAI batch job."""
        batch = await self._client.batches.retrieve(batch_id)
        return batch.status

    async def get_results(self, batch_id: str) -> BatchResult:
        """Get results from a completed OpenAI batch job."""
        batch = await self._client.batches.retrieve(batch_id)

        if batch.status != "completed":
            raise ValueError(f"Batch {batch_id} is not completed. Status: {batch.status}")

        results = []
        errors = []

        # Get output file content
        if batch.output_file_id:
            file_response = await self._client.files.content(batch.output_file_id)
            content = file_response.content.decode("utf-8")
            for line in content.strip().split("\n"):
                if line:
                    result = json.loads(line)
                    results.append(result)

        # Get error file content if exists
        if batch.error_file_id:
            error_response = await self._client.files.content(batch.error_file_id)
            error_content = error_response.content.decode("utf-8")
            for line in error_content.strip().split("\n"):
                if line:
                    error = json.loads(line)
                    errors.append(error)

        return BatchResult(
            batch_id=batch_id,
            status=batch.status,
            results=results,
            errors=errors if errors else None,
        )

    async def cancel(self, batch_id: str) -> bool:
        """Cancel an OpenAI batch job."""
        try:
            await self._client.batches.cancel(batch_id)
            return True
        except Exception:
            return False


class AzureBatchClient(BatchClient):
    """
    Batch client for Azure OpenAI using the OpenAI SDK with Azure configuration.
    """
    provider: str = "azure"

    def __init__(
        self,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        poll_interval: float = 30.0,
        max_poll_time: str = "24h",
    ):
        from openai import AsyncAzureOpenAI

        resolved_api_key = api_key or os.getenv("AZURE_API_KEY")
        resolved_endpoint = azure_endpoint or os.getenv("AZURE_API_BASE")
        resolved_version = api_version or os.getenv("AZURE_API_VERSION", "2024-07-01-preview")

        if not resolved_endpoint:
            raise ValueError("AZURE_API_BASE environment variable is not set")

        super().__init__(
            api_key=resolved_api_key,
            poll_interval=poll_interval,
            max_poll_time=max_poll_time,
        )
        self._client = AsyncAzureOpenAI(
            api_key=resolved_api_key,
            azure_endpoint=resolved_endpoint,
            api_version=resolved_version,
        )
        self._file_ids: dict[str, str] = {}

    def format(self, batch_request: BatchRequest) -> list[dict]:
        return [
            {
                "custom_id": request["id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": request["model"], "messages": request["messages"], **request["lm_kwargs"]},
            }
            for request in batch_request["requests"]
        ]

    def parse(self, raw_result: dict[str, Any]) -> ResultItem:
        """
        Parse Azure OpenAI batch result into ResultItem.

        Azure uses same format as OpenAI.
        """
        response = raw_result.get("response", {})
        body = response.get("body", {})
        choices = body.get("choices", [])

        if not choices:
            error = raw_result.get("error") or body.get("error", {})
            raise ValueError(f"Batch request failed: {error}")

        choice = choices[0]
        message = choice.get("message", {})

        result: ResultItem = {"text": message.get("content", "")}
        if "logprobs" in choice and choice["logprobs"] is not None:
            result["logprobs"] = choice["logprobs"]
        if "tool_calls" in message and message["tool_calls"] is not None:
            result["tool_calls"] = message["tool_calls"]

        return result

    async def _submit_batch_request(self, batch_request: BatchRequest) -> str:
        """Submit a batch job to Azure OpenAI."""
        BATCH_DIR.mkdir(parents=True, exist_ok=True)
        jsonl_path = self.create_jsonl(batch_request)

        try:
            with open(jsonl_path, "rb") as f:
                file_obj = await self._client.files.create(
                    file=f,
                    purpose="batch",
                )

            batch = await self._client.batches.create(
                completion_window="24h",
                endpoint="/v1/chat/completions",
                input_file_id=file_obj.id,
            )

            self._file_ids[batch.id] = file_obj.id
            return batch.id
        finally:
            if CLEANUP and jsonl_path.exists():
                jsonl_path.unlink()

    async def get_status(self, batch_id: str) -> str:
        """Get status of an Azure OpenAI batch job."""
        batch = await self._client.batches.retrieve(batch_id)
        return batch.status

    async def get_results(self, batch_id: str) -> BatchResult:
        """Get results from a completed Azure OpenAI batch job."""
        batch = await self._client.batches.retrieve(batch_id)

        if batch.status != "completed":
            raise ValueError(f"Batch {batch_id} is not completed. Status: {batch.status}")

        results = []
        errors = []

        if batch.output_file_id:
            file_response = await self._client.files.content(batch.output_file_id)
            content = file_response.content.decode("utf-8")
            for line in content.strip().split("\n"):
                if line:
                    results.append(json.loads(line))

        if batch.error_file_id:
            error_response = await self._client.files.content(batch.error_file_id)
            error_content = error_response.content.decode("utf-8")
            for line in error_content.strip().split("\n"):
                if line:
                    errors.append(json.loads(line))

        return BatchResult(
            batch_id=batch_id,
            status=batch.status,
            results=results,
            errors=errors if errors else None,
        )

    async def cancel(self, batch_id: str) -> bool:
        """Cancel an Azure OpenAI batch job."""
        try:
            await self._client.batches.cancel(batch_id)
            return True
        except Exception:
            return False


class TogetherBatchClient(BatchClient):
    """
    Batch client for Together AI.

    Uses the Together SDK for batch processing.
    """
    provider: str = "together_ai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        poll_interval: float = 30.0,
        max_poll_time: str = "24h",
    ):
        from together import Together

        resolved_api_key = api_key or os.getenv("TOGETHERAI_API_KEY") or os.getenv("TOGETHER_API_KEY")
        if not resolved_api_key:
            raise ValueError("TOGETHERAI_API_KEY or TOGETHER_API_KEY environment variable is not set")
        super().__init__(
            api_key=resolved_api_key,
            poll_interval=poll_interval,
            max_poll_time=max_poll_time,
        )
        self._client = Together(api_key=resolved_api_key)

    def format(self, batch_request: BatchRequest) -> list[dict]:
        return [
            {
                "custom_id": request["id"],
                "body": {"model": request["model"], "messages": request["messages"], **request["lm_kwargs"]},
            }
            for request in batch_request["requests"]
        ]

    def parse(self, raw_result: dict[str, Any]) -> ResultItem:
        """
        Parse Together AI batch result into ResultItem.

        Together format: {"custom_id": "...", "response": {"body": {"choices": [{"message": {"content": "..."}}]}}}
        """
        response = raw_result.get("response", {})
        body = response.get("body", {})
        choices = body.get("choices", [])

        if not choices:
            error = raw_result.get("error") or body.get("error", {})
            raise ValueError(f"Batch request failed: {error}")

        choice = choices[0]
        message = choice.get("message", {})

        result: ResultItem = {"text": message.get("content", "")}
        if "logprobs" in choice and choice["logprobs"] is not None:
            result["logprobs"] = choice["logprobs"]
        if "tool_calls" in message and message["tool_calls"] is not None:
            result["tool_calls"] = message["tool_calls"]

        return result

    async def _submit_batch_request(self, batch_request: BatchRequest) -> str:
        """Submit a batch job to Together AI using the Together SDK."""
        # Create temp JSONL file
        BATCH_DIR.mkdir(parents=True, exist_ok=True)
        jsonl_path = self.create_jsonl(batch_request)

        try:
            # Upload file using SDK (sync, run in thread)
            file_obj = await asyncio.to_thread(
                self._client.files.upload,
                file=str(jsonl_path),
                purpose="batch-api",
                check=False,
            )

            # Create batch using SDK (sync, run in thread)
            batch = await asyncio.to_thread(
                self._client.batches.create_batch,
                file_id=file_obj.id,
                endpoint="/v1/chat/completions",
            )

            return batch.id
        finally:
            # Clean up temp file
            if CLEANUP and jsonl_path.exists():
                jsonl_path.unlink()

    async def get_status(self, batch_id: str) -> str:
        """Get status of a Together AI batch job."""
        batch = await asyncio.to_thread(
            self._client.batches.get_batch,
            batch_id,
        )

        # Normalize status to lowercase
        status = (batch.status or "").lower()
        # Map Together statuses to common format
        status_map = {
            "validating": "in_progress",
            "in_progress": "in_progress",
            "completed": "completed",
            "failed": "failed",
            "cancelled": "cancelled",
        }
        return status_map.get(status, status)

    async def get_results(self, batch_id: str) -> BatchResult:
        """Get results from a completed Together AI batch job."""
        batch = await asyncio.to_thread(
            self._client.batches.get_batch,
            batch_id,
        )

        status = (batch.status or "").lower()
        if status != "completed":
            raise ValueError(f"Batch {batch_id} is not completed. Status: {status}")

        results = []
        errors = []

        # Get output file content
        if batch.output_file_id:
            # Download to temp file then read
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                await asyncio.to_thread(
                    self._client.files.retrieve_content,
                    id=batch.output_file_id,
                    output=tmp_path,
                )
                with open(tmp_path) as f:
                    for line in f:
                        if line.strip():
                            results.append(json.loads(line))
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # Get error file content if exists
        if batch.error_file_id:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                await asyncio.to_thread(
                    self._client.files.retrieve_content,
                    id=batch.error_file_id,
                    output=tmp_path,
                )
                with open(tmp_path) as f:
                    for line in f:
                        if line.strip():
                            errors.append(json.loads(line))
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        return BatchResult(
            batch_id=batch_id,
            status="completed",
            results=results,
            errors=errors if errors else None,
        )

    async def cancel(self, batch_id: str) -> bool:
        """Cancel a Together AI batch job."""
        try:
            await asyncio.to_thread(
                self._client.batches.cancel,
                batch_id,
            )
            return True
        except Exception:
            return False


class FireworksBatchClient(BatchClient):
    """
    Batch client for Fireworks AI.

    Uses the Fireworks HTTP API directly for batch processing.
    Fireworks requires model to be specified when creating the job, not per-request.
    """

    BASE_URL = "https://api.fireworks.ai/v1"
    provider: str = "fireworks_ai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        account_id: Optional[str] = None,
        poll_interval: float = 30.0,
        max_poll_time: str = "24h",
    ):
        resolved_api_key = api_key or os.getenv("FIREWORKS_AI_API_KEY")
        if not resolved_api_key:
            raise ValueError("FIREWORKS_AI_API_KEY environment variable is not set")
        super().__init__(
            api_key=resolved_api_key,
            poll_interval=poll_interval,
            max_poll_time=max_poll_time,
        )

        self.account_id = account_id or os.getenv("FIREWORKS_ACCOUNT_ID")
        if not self.account_id:
            raise ValueError("FIREWORKS_ACCOUNT_ID environment variable is not set")

        self._api_key = resolved_api_key
        self._output_dataset_ids: dict[str, str] = {}  # batch_id -> output_dataset_id mapping

    def _get_headers(self, content_type: str = "application/json") -> dict[str, str]:
        """Get common headers for API requests."""
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def format(self, batch_request: BatchRequest) -> list[dict]:
        # Fireworks doesn't include model per-request; model is specified at job creation
        return [
            {
                "custom_id": request["id"],
                "body": {"messages": request["messages"], **request["lm_kwargs"]},
            }
            for request in batch_request["requests"]
        ]

    def parse(self, raw_result: dict[str, Any]) -> ResultItem:
        """
        Parse Fireworks AI batch result into ResultItem.

        Fireworks format: {"custom_id": "...", "response": {"choices": [{"message": {"content": "..."}}]}}
        Note: Fireworks puts choices directly under response (no body wrapper like OpenAI).
        """
        response = raw_result.get("response", {})
        choices = response.get("choices", [])

        if not choices:
            error = raw_result.get("error") or response.get("error")
            raise ValueError(f"Batch request failed: {error}")

        choice = choices[0]
        message = choice.get("message", {})

        result: ResultItem = {"text": message.get("content", "")}
        if "logprobs" in choice and choice["logprobs"] is not None:
            result["logprobs"] = choice["logprobs"]
        if "tool_calls" in message and message["tool_calls"] is not None:
            result["tool_calls"] = message["tool_calls"]

        return result

    async def _submit_batch_request(self, batch_request: BatchRequest) -> str:
        """Submit a batch job to Fireworks AI using the HTTP API."""
        import httpx

        if batch_request["model"] is None:
            raise ValueError("Fireworks batch requires all requests to use the same model")

        # Create temp JSONL file
        BATCH_DIR.mkdir(parents=True, exist_ok=True)
        jsonl_path = self.create_jsonl(batch_request)

        try:
            dataset_id = f"batch-input-{uuid.uuid4().hex[:8]}"
            output_dataset_id = f"batch-output-{uuid.uuid4().hex[:8]}"
            job_id = f"batch-job-{uuid.uuid4().hex[:8]}"

            async with httpx.AsyncClient(timeout=120.0) as client:
                # Step 1: Create input dataset entry
                create_dataset_url = f"{self.BASE_URL}/accounts/{self.account_id}/datasets"
                create_dataset_payload = {
                    "datasetId": dataset_id,
                    "dataset": {"userUploaded": {}},
                }
                resp = await client.post(
                    create_dataset_url,
                    headers=self._get_headers(),
                    json=create_dataset_payload,
                )
                resp.raise_for_status()

                # Step 2: Upload JSONL file to dataset
                upload_url = f"{self.BASE_URL}/accounts/{self.account_id}/datasets/{dataset_id}:upload"
                with open(jsonl_path, "rb") as f:
                    files = {"file": (jsonl_path.name, f, "application/jsonl")}
                    resp = await client.post(
                        upload_url,
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        files=files,
                    )
                resp.raise_for_status()

                # Step 3: Wait for dataset to be READY before creating batch job
                dataset_status_url = f"{self.BASE_URL}/accounts/{self.account_id}/datasets/{dataset_id}"
                for _ in range(60):  # Wait up to 60 seconds
                    resp = await client.get(dataset_status_url, headers=self._get_headers())
                    resp.raise_for_status()
                    state = resp.json().get("state", "")
                    if state == "READY":
                        break
                    if state not in ("UPLOADING", "STATE_UNSPECIFIED"):
                        raise ValueError(f"Dataset in unexpected state: {state}")
                    await asyncio.sleep(1)
                else:
                    raise TimeoutError(f"Dataset {dataset_id} did not become READY within 60 seconds")

                # Step 4: Create batch inference job
                model = batch_request["model"]
                # Fireworks uses format like "accounts/fireworks/models/model-name"
                if not model.startswith("accounts/"):
                    model = f"accounts/fireworks/models/{model}"

                create_job_url = f"{self.BASE_URL}/accounts/{self.account_id}/batchInferenceJobs"
                create_job_payload = {
                    "displayName": f"Batch job {job_id}",
                    "model": model,
                    "inputDatasetId": f"accounts/{self.account_id}/datasets/{dataset_id}",
                    "outputDatasetId": f"accounts/{self.account_id}/datasets/{output_dataset_id}",
                }

                # Add inference parameters from lm_kwargs if present
                lm_kwargs = batch_request.get("lm_kwargs") or {}
                if lm_kwargs:
                    inference_params = {}
                    if "max_tokens" in lm_kwargs:
                        inference_params["maxTokens"] = lm_kwargs["max_tokens"]
                    if "temperature" in lm_kwargs:
                        inference_params["temperature"] = lm_kwargs["temperature"]
                    if "top_p" in lm_kwargs:
                        inference_params["topP"] = lm_kwargs["top_p"]
                    if inference_params:
                        create_job_payload["inferenceParameters"] = inference_params

                resp = await client.post(
                    create_job_url,
                    headers=self._get_headers(),
                    json=create_job_payload,
                    params={"batchInferenceJobId": job_id},
                )
                if resp.status_code != 200:
                    raise ValueError(f"Failed to create batch job: {resp.status_code} - {resp.text}")
                resp.raise_for_status()
                job_response = resp.json()

                # Extract job ID from the response
                # Response contains "name" field like "accounts/{account_id}/batchInferenceJobs/{job_id}"
                batch_id = job_response.get("name", "").split("/")[-1] or job_id
                self._output_dataset_ids[batch_id] = output_dataset_id

                return batch_id
        finally:
            # Clean up temp file
            if CLEANUP and jsonl_path.exists():
                jsonl_path.unlink()

    async def get_status(self, batch_id: str) -> str:
        """Get status of a Fireworks AI batch job."""
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{self.BASE_URL}/accounts/{self.account_id}/batchInferenceJobs/{batch_id}"
            resp = await client.get(url, headers=self._get_headers())
            resp.raise_for_status()
            job = resp.json()

        # Normalize status to lowercase and strip JOB_STATE_ prefix
        status = (job.get("state", "") or "").lower()
        if status.startswith("job_state_"):
            status = status[len("job_state_") :]
        # Map Fireworks statuses to common format
        status_map = {
            "validating": "in_progress",
            "pending": "in_progress",
            "running": "in_progress",
            "creating": "in_progress",
            "writing_results": "in_progress",
            "completed": "completed",
            "failed": "failed",
            "expired": "failed",
            "cancelled": "failed",
        }
        return status_map.get(status, status)

    async def get_results(self, batch_id: str) -> BatchResult:
        """Get results from a completed Fireworks AI batch job."""
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get job status first
            job_url = f"{self.BASE_URL}/accounts/{self.account_id}/batchInferenceJobs/{batch_id}"
            resp = await client.get(job_url, headers=self._get_headers())
            resp.raise_for_status()
            job = resp.json()

            status = (job.get("state", "") or "").lower()
            if status.startswith("job_state_"):
                status = status[len("job_state_") :]
            if status != "completed":
                raise ValueError(f"Batch {batch_id} is not completed. Status: {status}")

            results = []
            errors = []

            # Get output dataset ID from job or our cache
            output_dataset_id = self._output_dataset_ids.get(batch_id)
            if not output_dataset_id:
                # Try to extract from job response
                output_dataset_ref = job.get("outputDatasetId", "")
                if output_dataset_ref:
                    output_dataset_id = output_dataset_ref.split("/")[-1]

            if output_dataset_id:
                # Get download endpoint for the output dataset
                download_url = (
                    f"{self.BASE_URL}/accounts/{self.account_id}/datasets/{output_dataset_id}:getDownloadEndpoint"
                )
                resp = await client.get(download_url, headers=self._get_headers())
                resp.raise_for_status()
                download_info = resp.json()

                # Download all files from signed URLs
                filename_to_urls = download_info.get("filenameToSignedUrls", {})
                for filename, signed_url in filename_to_urls.items():
                    file_resp = await client.get(signed_url)
                    file_resp.raise_for_status()
                    content = file_resp.text

                    # Parse JSONL content
                    for line in content.strip().split("\n"):
                        if line.strip():
                            data = json.loads(line)
                            # Check if this is an error file based on filename
                            if "error" in filename.lower():
                                errors.append(data)
                            else:
                                results.append(data)

        return BatchResult(
            batch_id=batch_id,
            status="completed",
            results=results,
            errors=errors if errors else None,
        )

    async def cancel(self, batch_id: str) -> bool:
        """Cancel a Fireworks AI batch job (not supported by Fireworks once running)."""
        # Fireworks doesn't support cancellation once processing begins
        return False


class AnthropicBatchClient(BatchClient):
    """
    Batch client for Anthropic using the Anthropic SDK.

    Uses the AsyncAnthropic client for batch operations.
    Anthropic batches can take up to 24 hours to complete.
    """
    provider: str = "anthropic"

    def __init__(
        self,
        api_key: Optional[str] = None,
        poll_interval: float = 30.0,
        max_poll_time: str = "24h",
    ):
        import anthropic

        resolved_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        super().__init__(
            api_key=resolved_api_key,
            poll_interval=poll_interval,
            max_poll_time=max_poll_time,
        )
        self._client = anthropic.AsyncAnthropic(api_key=resolved_api_key)

    def format(self, batch_request: BatchRequest) -> list[dict]:
        """
        Format batch request into Anthropic's batch format.

        Anthropic format:
        {
            "custom_id": "request-1",
            "params": {
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello, world"}]
            }
        }
        """
        formatted = []
        for request in batch_request["requests"]:
            params = {
                "model": request["model"],
                "messages": request["messages"],
                **request["lm_kwargs"],
            }
            # Ensure max_tokens is set (required by Anthropic)
            if "max_tokens" not in params:
                params["max_tokens"] = 4096

            formatted.append(
                {
                    "custom_id": request["id"],
                    "params": params,
                }
            )
        return formatted

    def parse(self, raw_result: dict[str, Any]) -> ResultItem:
        """
        Parse Anthropic batch result into ResultItem.

        Anthropic format:
        {
            "custom_id": "...",
            "result": {
                "type": "succeeded",
                "message": {
                    "content": [{"type": "text", "text": "..."}],
                    ...
                }
            }
        }
        """
        result = raw_result.get("result", {})
        result_type = result.get("type", "")

        if result_type == "errored":
            error = result.get("error", {})
            raise ValueError(f"Batch request failed: {error}")
        elif result_type == "canceled":
            raise ValueError("Batch request was canceled")
        elif result_type == "expired":
            raise ValueError("Batch request expired")
        elif result_type != "succeeded":
            raise ValueError(f"Unknown result type: {result_type}")

        message = result.get("message", {})
        content = message.get("content", [])

        # Extract text from content blocks
        text_parts = []
        tool_calls = []
        for block in content:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(block)

        result_item: ResultItem = {"text": "".join(text_parts)}
        if tool_calls:
            result_item["tool_calls"] = tool_calls

        return result_item

    async def _submit_batch_request(self, batch_request: BatchRequest) -> str:
        """Submit a batch job to Anthropic."""
        formatted_requests = self.format(batch_request)

        # Anthropic SDK expects a list of request objects
        batch = await self._client.messages.batches.create(
            requests=formatted_requests,
        )

        return batch.id

    async def get_status(self, batch_id: str) -> str:
        """
        Get status of an Anthropic batch job.

        Anthropic processing_status values:
        - "in_progress": Batch is being processed
        - "canceling": Cancellation has been initiated
        - "ended": Processing has ended (check request_counts for details)
        """
        batch = await self._client.messages.batches.retrieve(batch_id)

        # Map Anthropic status to common format
        status = batch.processing_status
        if status == "ended":
            # Check if there were failures
            counts = batch.request_counts
            if counts.errored > 0 or counts.expired > 0:
                # Still return completed - individual results will show errors
                return "completed"
            return "completed"
        elif status == "canceling":
            return "in_progress"
        else:
            return "in_progress"

    async def get_results(self, batch_id: str) -> BatchResult:
        """Get results from a completed Anthropic batch job."""
        import httpx

        batch = await self._client.messages.batches.retrieve(batch_id)

        if batch.processing_status != "ended":
            raise ValueError(f"Batch {batch_id} is not completed. Status: {batch.processing_status}")

        results = []
        errors = []

        # Fetch results from the results_url
        if batch.results_url:
            async with httpx.AsyncClient(timeout=120.0) as client:
                # The results_url requires authentication
                headers = {
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                }
                resp = await client.get(batch.results_url, headers=headers)
                resp.raise_for_status()
                content = resp.text

                # Parse JSONL content
                for line in content.strip().split("\n"):
                    if line.strip():
                        data = json.loads(line)
                        result_type = data.get("result", {}).get("type", "")
                        if result_type in ("errored", "canceled", "expired"):
                            errors.append(data)
                        else:
                            results.append(data)

        return BatchResult(
            batch_id=batch_id,
            status="completed",
            results=results,
            errors=errors if errors else None,
        )

    async def cancel(self, batch_id: str) -> bool:
        """Cancel an Anthropic batch job."""
        try:
            await self._client.messages.batches.cancel(batch_id)
            return True
        except Exception:
            return False


class VertexAIBatchClient(BatchClient):
    """
    Batch client for Vertex AI.

    Not implemented yet.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        poll_interval: float = 30.0,
        max_poll_time: str = "24h",
    ):
        super().__init__(
            provider="vertex_ai",
            api_key=api_key,
            poll_interval=poll_interval,
            max_poll_time=max_poll_time,
        )

    def format(self, batch_request: BatchRequest) -> list[dict]:
        raise NotImplementedError("Vertex AI batch is not implemented yet")

    def parse(self, raw_result: dict[str, Any]) -> ResultItem:
        raise NotImplementedError("Vertex AI batch is not implemented yet")

    async def _submit_batch_request(self, batch_request: BatchRequest) -> str:
        raise NotImplementedError("Vertex AI batch is not implemented yet")

    async def get_status(self, batch_id: str) -> str:
        raise NotImplementedError("Vertex AI batch is not implemented yet")

    async def get_results(self, batch_id: str) -> BatchResult:
        raise NotImplementedError("Vertex AI batch is not implemented yet")
