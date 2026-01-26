import asyncio
import json
import logging
import os
import random
import uuid
import warnings
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from ..constants import BATCH_DIR
from .types import BatchReponse, BatchRequest, ResultItem

CLEANUP = True
logger = logging.getLogger(__name__)

# Try to import httpx for common network error handling
try:
    import httpx

    HTTPX_ERRORS = (
        httpx.ConnectError,
        httpx.ReadError,
        httpx.WriteError,
        httpx.PoolTimeout,
        httpx.TimeoutException,
        httpx.NetworkError,
    )
except ImportError:
    HTTPX_ERRORS = ()


async def _retry_on_network_error(
    coro_func: Callable[..., Any], *args: Any, max_retries: int = 5, provider_name: str = "provider", **kwargs: Any
) -> Any:
    """
    Helper to retry an async function on common network/connection errors.
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            logger.debug("%s network attempt %d/%d", provider_name, attempt + 1, max_retries)
            return await coro_func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            error_name = type(e).__name__
            error_msg = str(e).lower()

            # Check for common network/connection error indicators
            is_network_error = (
                any(indicator in error_name for indicator in ["Connection", "Timeout", "ReadError", "WriteError"])
                or any(indicator in error_msg for indicator in ["connection", "timeout", "read error", "write error"])
                or "RemoteDisconnected" in error_name
                or (HTTPX_ERRORS and isinstance(e, HTTPX_ERRORS))
            )

            # Also catch provider-specific connection errors
            if not is_network_error:
                if "openai" in provider_name.lower():
                    try:
                        import openai

                        if isinstance(e, openai.APIConnectionError):
                            is_network_error = True
                    except ImportError:
                        pass
                elif "anthropic" in provider_name.lower():
                    try:
                        import anthropic

                        if isinstance(e, anthropic.APIConnectionError):
                            is_network_error = True
                    except ImportError:
                        pass

            if not is_network_error or attempt == max_retries - 1:
                logger.error(
                    "%s operation failed without retry (attempt %d/%d): %s",
                    provider_name,
                    attempt + 1,
                    max_retries,
                    e,
                    exc_info=True,
                )
                raise

            # Exponential backoff with jitter: 2, 4, 8, 16, 32... + random 0-2s
            delay = (2 ** (attempt + 1)) + (random.random() * 2)
            logger.warning(
                "%s network error (attempt %d/%d): %s. Retrying in %.2fs",
                provider_name,
                attempt + 1,
                max_retries,
                e,
                delay,
            )
            await asyncio.sleep(delay)

    if last_exception:
        logger.error(
            "%s operation failed after %d attempts: %s",
            provider_name,
            max_retries,
            last_exception,
            exc_info=True,
        )
        raise last_exception


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
    status_callback: Optional[Callable[[str, str, Optional[int], dict], None]]

    def __init__(
        self,
        api_key: Optional[str] = None,
        poll_interval: float = 30.0,
        max_poll_time: str = "24h",
        status_callback: Optional[Callable[[str, str, Optional[int], dict], None]] = None,
    ):
        """
        Initialize a batch client.

        Args:
            api_key: Optional API key for the provider. If not provided, will use environment variables.
            poll_interval: Seconds between status checks (default: 30.0).
            max_poll_time: Maximum time to wait for completion as a string like "30s", "5m", or "24h" (default: "24h").
            status_callback: Optional callback for status updates: (batch_id, status, progress, metadata).
        """
        self.api_key = api_key
        self.poll_interval = poll_interval
        self.max_poll_time = max_poll_time
        self.max_poll_time_s = _parse_time_string(max_poll_time)
        self.status_callback = status_callback
        self.start_time: Optional[float] = None
        self.num_requests: Optional[int] = None
        self._consecutive_failures = 0

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
        logger.debug("Creating JSONL batch file: path=%s requests=%d", path, len(batch_request["requests"]))
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

    async def get_status(self, batch_id: str) -> Tuple[str, Optional[int]]:
        """
        Get the status of a batch job.

        Args:
            batch_id: The batch ID to check.

        Returns:
            status, progress: Tuple[str, int]
            Status string (e.g., "completed", "in_progress", "failed"), progress percentage (0-100).
        """
        try:
            status, progress = await self._get_status_impl(batch_id)
            self._consecutive_failures = 0
        except Exception as e:
            logger.warning(
                "Batch status check failed: provider=%s batch_id=%s error=%s",
                self.provider,
                batch_id,
                e,
                exc_info=True,
            )
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3:
                import warnings

                warnings.warn(
                    f"Batch status check failed {self._consecutive_failures} times in a row for batch {batch_id} ({self.provider}): {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            status = "in_progress"
            progress = None

        if self.status_callback is not None:
            import time

            elapsed = time.time() - self.start_time if self.start_time else 0.0
            metadata = {
                "provider": self.provider,
                "num_requests": self.num_requests,
                "elapsed_time": elapsed,
            }
            self.status_callback(batch_id, status, progress, metadata)
        logger.debug(
            "Batch status retrieved: provider=%s batch_id=%s status=%s progress=%s",
            self.provider,
            batch_id,
            status,
            progress,
        )
        return status, progress

    async def _get_status_impl(self, batch_id: str) -> Tuple[str, Optional[int]]:
        raise NotImplementedError("get_status is not implemented")

    async def get_results(self, batch_id: str) -> BatchReponse:
        """
        Get the results of a completed batch job.

        Args:
            batch_id: The batch ID to get results for.

        Returns:
            BatchReponse containing the results.
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
    ) -> BatchReponse:
        """
        Submit a batch job and wait for completion.

        Args:
            batch_request: The batch request to submit.
            show_progress: Whether to show progress (ignored, onus shifts to abatch).

        Returns:
            BatchReponse containing raw results from the API.

        Raises:
            TimeoutError: If the job doesn't complete within max_poll_time.
            RuntimeError: If the job fails.
        """
        import time

        self.num_requests = len(batch_request["requests"])
        self.start_time = time.time()

        logger.debug(
            "submit_and_wait start: provider=%s requests=%d show_progress=%s",
            self.provider,
            self.num_requests,
            show_progress,
        )
        batch_id = await self._submit_batch_request(batch_request)
        logger.debug("submit_and_wait submitted: provider=%s batch_id=%s", self.provider, batch_id)
        waited = 0.0

        while waited < self.max_poll_time_s:
            status, progress = await self.get_status(batch_id)
            logger.debug(
                "submit_and_wait poll: provider=%s batch_id=%s status=%s progress=%s waited=%.1fs",
                self.provider,
                batch_id,
                status,
                progress,
                waited,
            )

            if status.lower() == "completed":
                logger.debug("submit_and_wait completed: provider=%s batch_id=%s", self.provider, batch_id)
                return await self.get_results(batch_id)
            elif status.lower() in ("failed", "cancelled", "expired"):
                try:
                    # Try to get failure details
                    failure_results = await self.get_results(batch_id)
                    error_msg = f"Batch job {batch_id} failed with status: {status}"

                    # Consolidate all possible errors
                    all_errors = failure_results.errors or []

                    if all_errors:
                        # Extract the actual error payload for display
                        display_errors = []
                        for e in all_errors[:5]:  # Show up to 5
                            if "batch_error" in e:
                                display_errors.append(e["batch_error"])
                            elif "error" in e:
                                display_errors.append(e["error"])
                            else:
                                display_errors.append(e)

                        error_details = json.dumps(display_errors, indent=2)
                        error_msg += f"\nErrors found:\n{error_details}"
                    else:
                        error_msg += f"\nNo specific error details found in batch. Check {self.provider} dashboard for more info. \n response: {failure_results.raw_response}"

                    logger.error(
                        "submit_and_wait failure: provider=%s batch_id=%s failure_results=%s",
                        self.provider,
                        batch_id,
                        failure_results.raw_response,
                    )
                    raise RuntimeError(error_msg)
                except Exception as e:
                    if isinstance(e, RuntimeError) and "Batch job" in str(e):
                        raise
                    # Fallback if get_results fails
                    logger.error(
                        "submit_and_wait failure details fetch failed: provider=%s batch_id=%s error=%s",
                        self.provider,
                        batch_id,
                        e,
                        exc_info=True,
                    )
                    raise RuntimeError(
                        f"Batch job {batch_id} failed with status: {status}. Also failed to fetch error details: {e}"
                    )

            logger.debug("submit_and_wait sleeping: provider=%s seconds=%.1f", self.provider, self.poll_interval)
            await asyncio.sleep(self.poll_interval)
            waited += self.poll_interval

        logger.error(
            "submit_and_wait timeout: provider=%s batch_id=%s max_poll_time=%s",
            self.provider,
            batch_id,
            self.max_poll_time,
        )
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
        status_callback: Optional[Callable[[str, str, Optional[int], dict], None]] = None,
    ):
        """
        Args:
            api_key: The API key to use for the client.
            poll_interval: The interval in seconds to poll the status of the batch job.
            max_poll_time: The maximum time in seconds to wait for the batch job to complete.
            status_callback: A callback function to call when the status of the batch job changes.
        Example:
            def status_callback(batch_id, status, progress, metadata):
                print(f"Batch job {batch_id} is {status} with progress {progress}% and elapsed time {metadata['elapsed_time']} seconds")
                print(f"Metadata: {metadata}")
        """
        from openai import AsyncOpenAI

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        super().__init__(
            api_key=resolved_api_key,
            poll_interval=poll_interval,
            max_poll_time=max_poll_time,
            status_callback=status_callback,
        )
        self._client = AsyncOpenAI(api_key=resolved_api_key, max_retries=5, timeout=300.0)
        self._file_ids: dict[str, str] = {}  # batch_id -> input_file_id mapping

    def format(self, batch_request: BatchRequest) -> list[dict]:
        requests = [
            {
                "custom_id": request["id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": request["model"], "messages": request["messages"], **request["lm_kwargs"]},
            }
            for request in batch_request["requests"]
        ]
        return requests

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
        logger.debug("OpenAI submit: uploading file %s", jsonl_path)

        async def _do_submit() -> str:
            # Upload file
            with open(jsonl_path, "rb") as f:
                file_obj = await self._client.files.create(
                    file=f,
                    purpose="batch",
                )

            # Create batch
            logger.debug("OpenAI submit: creating batch from file_id=%s", file_obj.id)
            batch = await self._client.batches.create(
                completion_window="24h",
                endpoint="/v1/chat/completions",
                input_file_id=file_obj.id,
            )

            # Store file_id for later cleanup if needed
            self._file_ids[batch.id] = file_obj.id

            logger.debug("OpenAI submit: created batch_id=%s", batch.id)
            return batch.id

        try:
            return await _retry_on_network_error(_do_submit, provider_name=self.provider, max_retries=7)
        finally:
            # Clean up temp file
            if CLEANUP and jsonl_path.exists():
                jsonl_path.unlink()

    async def _get_status_impl(self, batch_id: str) -> Tuple[str, Optional[int]]:
        """Get status of an OpenAI batch job."""
        logger.debug("OpenAI status request: batch_id=%s", batch_id)
        batch = await self._client.batches.retrieve(batch_id)
        logger.debug("OpenAI status retrieved: batch_id=%s batch=%s", batch_id, batch)
        req_counts = batch.request_counts
        if req_counts is None:
            return batch.status, None  # type: ignore

        total = req_counts.total
        finished = req_counts.completed + req_counts.failed
        progress = int((finished / total) * 100) if total > 0 else 0
        return batch.status, progress  # type: ignore

    async def get_results(self, batch_id: str) -> BatchReponse:
        """Get results from a completed OpenAI batch job."""

        async def _do_get() -> BatchReponse:
            logger.debug("OpenAI results request: batch_id=%s", batch_id)
            batch = await self._client.batches.retrieve(batch_id)
            logger.debug("OpenAI results retrieved: batch_id=%s batch=%s", batch_id, batch)

            if batch.status not in ("completed", "failed", "cancelled", "expired"):
                raise ValueError(f"Batch {batch_id} is not in a terminal state. Status: {batch.status}")

            results = []
            errors = []

            # Check for batch-level errors
            batch_errors = getattr(batch, "errors", None)
            if batch_errors:
                # OpenAI SDK: batch.errors can be a list or have a .data attribute
                items = batch_errors.data if hasattr(batch_errors, "data") else batch_errors
                if isinstance(items, list):
                    for err in items:
                        # Extract structured data from BatchError object if possible
                        err_detail = {}
                        for attr in ["code", "message", "param"]:
                            if hasattr(err, attr):
                                err_detail[attr] = getattr(err, attr)

                        if err_detail:
                            errors.append({"batch_error": err_detail})
                        else:
                            errors.append({"batch_error": str(err)})

            # Get output file content
            if batch.output_file_id:
                logger.debug("OpenAI results: downloading output_file_id=%s", batch.output_file_id)
                file_response = await self._client.files.content(batch.output_file_id)
                content = file_response.content.decode("utf-8")
                for line in content.strip().split("\n"):
                    if line:
                        result = json.loads(line)
                        results.append(result)

            # Get error file content if exists
            if batch.error_file_id:
                logger.debug("OpenAI results: downloading error_file_id=%s", batch.error_file_id)
                error_response = await self._client.files.content(batch.error_file_id)
                error_content = error_response.content.decode("utf-8")
                for line in error_content.strip().split("\n"):
                    if line:
                        error = json.loads(line)
                        errors.append(error)

            return BatchReponse(
                batch_id=batch_id,
                status=batch.status,
                results=results,
                errors=errors if errors else None,
            )

        return await _retry_on_network_error(_do_get, provider_name=self.provider)

    async def cancel(self, batch_id: str) -> bool:
        """Cancel an OpenAI batch job."""
        try:
            logger.debug("OpenAI cancel request: batch_id=%s", batch_id)
            await self._client.batches.cancel(batch_id)
            return True
        except Exception:
            logger.error("OpenAI cancel failed: batch_id=%s", batch_id, exc_info=True)
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
        status_callback: Optional[Callable[[str, str, Optional[int], dict], None]] = None,
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
            status_callback=status_callback,
        )
        self._client = AsyncAzureOpenAI(
            api_key=resolved_api_key,
            azure_endpoint=resolved_endpoint,
            api_version=resolved_version,
            timeout=300.0,
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
        logger.debug("Azure submit: uploading file %s", jsonl_path)

        async def _do_submit() -> str:
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
            logger.debug("Azure submit: created batch_id=%s", batch.id)
            return batch.id

        try:
            return await _retry_on_network_error(_do_submit, provider_name=self.provider, max_retries=7)
        finally:
            if CLEANUP and jsonl_path.exists():
                jsonl_path.unlink()

    async def _get_status_impl(self, batch_id: str) -> Tuple[str, Optional[int]]:
        """Get status of an OpenAI batch job."""
        logger.debug("Azure status request: batch_id=%s", batch_id)
        batch = await self._client.batches.retrieve(batch_id)
        logger.debug("Azure status retrieved: batch_id=%s batch=%s", batch_id, batch)
        req_counts = batch.request_counts
        if req_counts is None:
            return batch.status, None  # type: ignore

        total = req_counts.total
        finished = req_counts.completed + req_counts.failed
        progress = int((finished / total) * 100) if total > 0 else 0
        return batch.status, progress  # type: ignore

    async def get_results(self, batch_id: str) -> BatchReponse:
        """Get results from a completed Azure OpenAI batch job."""

        async def _do_get() -> BatchReponse:
            logger.debug("Azure results request: batch_id=%s", batch_id)
            batch = await self._client.batches.retrieve(batch_id)
            logger.debug("Azure results retrieved: batch_id=%s batch=%s", batch_id, batch)
            if batch.status not in ("completed", "failed", "cancelled", "expired"):
                raise ValueError(f"Batch {batch_id} is not in a terminal state. Status: {batch.status}")

            results = []
            errors = []

            if batch.output_file_id:
                logger.debug("Azure results: downloading output_file_id=%s", batch.output_file_id)
                file_response = await self._client.files.content(batch.output_file_id)
                content = file_response.content.decode("utf-8")
                for line in content.strip().split("\n"):
                    if line:
                        results.append(json.loads(line))

            if batch.error_file_id:
                logger.debug("Azure results: downloading error_file_id=%s", batch.error_file_id)
                error_response = await self._client.files.content(batch.error_file_id)
                error_content = error_response.content.decode("utf-8")
                for line in error_content.strip().split("\n"):
                    if line:
                        errors.append(json.loads(line))

            return BatchReponse(
                batch_id=batch_id,
                status=batch.status,
                results=results,
                errors=errors if errors else None,
            )

        return await _retry_on_network_error(_do_get, provider_name=self.provider)

    async def cancel(self, batch_id: str) -> bool:
        """Cancel an Azure OpenAI batch job."""
        try:
            logger.debug("Azure cancel request: batch_id=%s", batch_id)
            await self._client.batches.cancel(batch_id)
            return True
        except Exception:
            logger.error("Azure cancel failed: batch_id=%s", batch_id, exc_info=True)
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
        status_callback: Optional[Callable[[str, str, Optional[int], dict], None]] = None,
    ):
        from together import AsyncTogether

        resolved_api_key = api_key or os.getenv("TOGETHERAI_API_KEY") or os.getenv("TOGETHER_API_KEY")
        if not resolved_api_key:
            raise ValueError("TOGETHERAI_API_KEY or TOGETHER_API_KEY environment variable is not set")
        super().__init__(
            api_key=resolved_api_key,
            poll_interval=poll_interval,
            max_poll_time=max_poll_time,
            status_callback=status_callback,
        )
        self._client = AsyncTogether(api_key=resolved_api_key, timeout=300.0)

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

        Together format: {"custom_id": "...", "response": {"body": {"choices": [...]}}}
        OR Together format: {"custom_id": "...", "response": {"choices": [...]}}
        """
        response = raw_result.get("response", {})

        # Together's response format can vary; sometimes it includes a 'body' wrapper
        # to match OpenAI's batch format exactly, other times fields are direct.
        body = response.get("body", response)
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
        logger.debug("Together submit: uploading file %s", jsonl_path)

        async def _do_submit() -> str:
            # Upload file using SDK
            file_obj = await self._client.files.upload(
                file=str(jsonl_path),
                purpose="batch-api",
                check=False,
            )

            # Create batch using SDK
            logger.debug("Together submit: creating batch from file_id=%s", file_obj.id)
            batch = await self._client.batches.create(
                input_file_id=file_obj.id,
                endpoint="/v1/chat/completions",
            )

            if batch.job and batch.job.id:
                logger.debug("Together submit: created batch_id=%s", batch.job.id)
                return batch.job.id
            raise RuntimeError("Failed to get batch ID from Together API")

        try:
            return await _retry_on_network_error(_do_submit, provider_name=self.provider, max_retries=7)
        finally:
            # Clean up temp file
            if CLEANUP and jsonl_path.exists():
                jsonl_path.unlink()

    async def _get_status_impl(self, batch_id: str) -> Tuple[str, Optional[int]]:
        """Get status of a Together AI batch job."""
        logger.debug("Together status request: batch_id=%s", batch_id)
        batch = await self._client.batches.retrieve(batch_id)
        logger.debug("Together status retrieved: batch_id=%s batch=%s", batch_id, batch)

        # Normalize status to lowercase
        status = (batch.status or "").lower()
        # Map Together statuses to common format
        status_map = {
            "validating": "in_progress",
            "queued": "in_progress",
            "running": "in_progress",
            "processing": "in_progress",
            "in_progress": "in_progress",
            "completed": "completed",
            "failed": "failed",
            "error": "failed",
            "errored": "failed",
            "cancelled": "cancelled",
            "canceled": "cancelled",
            "cancelling": "cancelled",
            "canceling": "cancelled",
            "expired": "expired",
        }
        normalized_status = status_map.get(status, status)
        return normalized_status, batch.progress  # type: ignore

    async def get_results(self, batch_id: str) -> BatchReponse:
        """Get results from a completed Together AI batch job."""

        async def _do_get() -> BatchReponse:
            logger.debug("Together results request: batch_id=%s", batch_id)
            batch = await self._client.batches.retrieve(batch_id)
            logger.debug("Together results retrieved: batch_id=%s batch=%s", batch_id, batch)

            status = (batch.status or "").lower()
            if status not in ("completed", "failed", "cancelled", "expired"):
                raise ValueError(f"Batch {batch_id} is not in a terminal state. Status: {status}")

            results = []
            errors = []

            # Check for batch-level errors
            batch_errors = getattr(batch, "errors", None)
            if batch_errors:
                # OpenAI SDK: batch.errors can be a list or have a .data attribute
                items = batch_errors.data if hasattr(batch_errors, "data") else batch_errors
                if isinstance(items, list):
                    for err in items:
                        # Extract structured data from BatchError object if possible
                        err_detail = {}
                        for attr in ["code", "message", "param"]:
                            if hasattr(err, attr):
                                err_detail[attr] = getattr(err, attr)

                        if err_detail:
                            errors.append({"batch_error": err_detail})
                        else:
                            errors.append({"batch_error": str(err)})

            # Get output file content
            if batch.output_file_id:
                logger.debug("Together results: downloading output_file_id=%s", batch.output_file_id)
                # Download to temp file then read
                import tempfile

                with tempfile.NamedTemporaryFile(mode="wb", suffix=".jsonl", delete=False) as tmp:
                    tmp_path = tmp.name
                    async with self._client.files.with_streaming_response.content(id=batch.output_file_id) as response:
                        async for chunk in response.iter_bytes():
                            tmp.write(chunk)

                try:
                    with open(tmp_path) as f:
                        for line in f:
                            if line.strip():
                                results.append(json.loads(line))
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            # Get error file content if exists
            if batch.error_file_id:
                logger.debug("Together results: downloading error_file_id=%s", batch.error_file_id)
                with tempfile.NamedTemporaryFile(mode="wb", suffix=".jsonl", delete=False) as tmp:
                    tmp_path = tmp.name
                    async with self._client.files.with_streaming_response.content(id=batch.error_file_id) as response:
                        async for chunk in response.iter_bytes():
                            tmp.write(chunk)

                try:
                    with open(tmp_path) as f:
                        for line in f:
                            if line.strip():
                                errors.append(json.loads(line))
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            return BatchReponse(
                batch_id=batch_id,
                status="completed",
                results=results,
                errors=errors if errors else None,
            )

        return await _retry_on_network_error(_do_get, provider_name=self.provider)

    async def cancel(self, batch_id: str) -> bool:
        """Cancel a Together AI batch job."""
        try:
            logger.debug("Together cancel request: batch_id=%s", batch_id)
            await self._client.batches.cancel(batch_id)
            return True
        except Exception:
            logger.error("Together cancel failed: batch_id=%s", batch_id, exc_info=True)
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
        status_callback: Optional[Callable[[str, str, Optional[int], dict], None]] = None,
    ):
        resolved_api_key = api_key or os.getenv("FIREWORKS_AI_API_KEY")
        if not resolved_api_key:
            raise ValueError("FIREWORKS_AI_API_KEY environment variable is not set")
        super().__init__(
            api_key=resolved_api_key,
            poll_interval=poll_interval,
            max_poll_time=max_poll_time,
            status_callback=status_callback,
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

        async def _do_submit() -> str:
            dataset_id = f"batch-input-{uuid.uuid4().hex[:8]}"
            output_dataset_id = f"batch-output-{uuid.uuid4().hex[:8]}"
            job_id = f"batch-job-{uuid.uuid4().hex[:8]}"

            async with httpx.AsyncClient(timeout=120.0) as client:
                # Step 1: Create input dataset entry
                create_dataset_url = f"{self.BASE_URL}/accounts/{self.account_id}/datasets"
                logger.debug("Fireworks request: POST %s", create_dataset_url)
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
                logger.debug("Fireworks request: POST %s (upload)", upload_url)
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
                    logger.debug("Fireworks request: GET %s (dataset status)", dataset_status_url)
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
                logger.debug("Fireworks request: POST %s (create job)", create_job_url)
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

                logger.debug("Fireworks submit: created batch_id=%s", batch_id)
                return batch_id

        try:
            return await _retry_on_network_error(_do_submit, provider_name=self.provider, max_retries=7)
        finally:
            # Clean up temp file
            if CLEANUP and jsonl_path.exists():
                jsonl_path.unlink()

    async def _get_status_impl(self, batch_id: str) -> Tuple[str, Optional[int]]:
        """Get status of a Fireworks AI batch job."""
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{self.BASE_URL}/accounts/{self.account_id}/batchInferenceJobs/{batch_id}"
            logger.debug("Fireworks request: GET %s (job status)", url)
            resp = await client.get(url, headers=self._get_headers())
            resp.raise_for_status()
            job = resp.json()
            logger.debug("Fireworks job retrieved: batch_id=%s job=%s", batch_id, job)

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
        return status_map.get(status, status), (job.get("jobProgress", None) or {}).get("percent", None)

    async def get_results(self, batch_id: str) -> BatchReponse:
        """Get results from a completed Fireworks AI batch job."""
        import httpx

        async def _do_get() -> BatchReponse:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get job status first
                job_url = f"{self.BASE_URL}/accounts/{self.account_id}/batchInferenceJobs/{batch_id}"
                logger.debug("Fireworks request: GET %s (job status)", job_url)
                resp = await client.get(job_url, headers=self._get_headers())
                resp.raise_for_status()
                job = resp.json()
                logger.debug("Fireworks job retrieved: batch_id=%s job=%s", batch_id, job)

                status = (job.get("state", "") or "").lower()
                if status.startswith("job_state_"):
                    status = status[len("job_state_") :]
                if status not in ("completed", "failed", "cancelled", "expired"):
                    raise ValueError(f"Batch {batch_id} is not in a terminal state. Status: {status}")

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
                    logger.debug("Fireworks request: GET %s (download endpoint)", download_url)
                    resp = await client.get(download_url, headers=self._get_headers())
                    resp.raise_for_status()
                    download_info = resp.json()

                    # Download all files from signed URLs
                    filename_to_urls = download_info.get("filenameToSignedUrls", {})
                    for filename, signed_url in filename_to_urls.items():
                        logger.debug("Fireworks request: GET signed_url for %s", filename)
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

            return BatchReponse(
                batch_id=batch_id,
                status="completed",
                results=results,
                errors=errors if errors else None,
            )

        return await _retry_on_network_error(_do_get, provider_name=self.provider)

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
        status_callback: Optional[Callable[[str, str, Optional[int], dict], None]] = None,
    ):
        import anthropic

        resolved_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        super().__init__(
            api_key=resolved_api_key,
            poll_interval=poll_interval,
            max_poll_time=max_poll_time,
            status_callback=status_callback,
        )
        self._client = anthropic.AsyncAnthropic(api_key=resolved_api_key, max_retries=5, timeout=300.0)

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

        async def _do_submit() -> str:
            logger.debug("Anthropic submit: requests=%d", len(formatted_requests))
            # Anthropic SDK expects a list of request objects
            batch = await self._client.messages.batches.create(
                requests=formatted_requests,
            )
            return batch.id

        return await _retry_on_network_error(_do_submit, provider_name=self.provider)

    async def _get_status_impl(self, batch_id: str) -> Tuple[str, Optional[int]]:
        """
        Get status of an Anthropic batch job.

        Anthropic processing_status values:
        - "in_progress": Batch is being processed
        - "canceling": Cancellation has been initiated
        - "ended": Processing has ended (check request_counts for details)
        """
        logger.debug("Anthropic status request: batch_id=%s", batch_id)
        batch = await self._client.messages.batches.retrieve(batch_id)
        logger.debug("Anthropic status retrieved: batch_id=%s batch=%s", batch_id, batch)

        req_counts = batch.request_counts
        total = (
            req_counts.canceled + req_counts.errored + req_counts.expired + req_counts.processing + req_counts.succeeded
        )
        progress = int((1 - req_counts.processing / total) * 100 if total > 0 else 0)

        # Map Anthropic status to common format
        status = batch.processing_status
        if status == "ended":
            # Check if there were failures
            counts = batch.request_counts
            if counts.errored > 0 or counts.expired > 0:
                # Still return completed - individual results will show errors
                return "completed", progress
            return "completed", progress
        elif status == "canceling":
            return "in_progress", progress
        else:
            return "in_progress", progress

    async def get_results(self, batch_id: str) -> BatchReponse:
        """Get results from a completed Anthropic batch job."""
        import httpx

        async def _do_get() -> BatchReponse:
            logger.debug("Anthropic results request: batch_id=%s", batch_id)
            batch = await self._client.messages.batches.retrieve(batch_id)
            logger.debug("Anthropic results retrieved: batch_id=%s batch=%s", batch_id, batch)

            if batch.processing_status not in ("ended", "canceling"):
                raise ValueError(f"Batch {batch_id} is not in a terminal state. Status: {batch.processing_status}")

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
                    logger.debug("Anthropic request: GET results_url=%s", batch.results_url)
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

            return BatchReponse(
                batch_id=batch_id,
                status="completed",
                results=results,
                errors=errors if errors else None,
            )

        return await _retry_on_network_error(_do_get, provider_name=self.provider)

    async def cancel(self, batch_id: str) -> bool:
        """Cancel an Anthropic batch job."""
        try:
            logger.debug("Anthropic cancel request: batch_id=%s", batch_id)
            await self._client.messages.batches.cancel(batch_id)
            return True
        except Exception:
            logger.error("Anthropic cancel failed: batch_id=%s", batch_id, exc_info=True)
            return False


class VertexAIBatchClient(BatchClient):
    """
    Batch client for Vertex AI.

    Not implemented yet.
    """

    provider: str = "vertex_ai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        poll_interval: float = 30.0,
        max_poll_time: str = "24h",
        status_callback: Optional[Callable[[str, str, Optional[int], dict], None]] = None,
    ):
        super().__init__(
            api_key=api_key,
            poll_interval=poll_interval,
            max_poll_time=max_poll_time,
            status_callback=status_callback,
        )

    def format(self, batch_request: BatchRequest) -> list[dict]:
        raise NotImplementedError("Vertex AI batch is not implemented yet")

    def parse(self, raw_result: dict[str, Any]) -> ResultItem:
        raise NotImplementedError("Vertex AI batch is not implemented yet")

    async def _submit_batch_request(self, batch_request: BatchRequest) -> str:
        raise NotImplementedError("Vertex AI batch is not implemented yet")

    async def get_status(self, batch_id: str) -> Tuple[str, Optional[int]]:
        raise NotImplementedError("Vertex AI batch is not implemented yet")

    async def get_results(self, batch_id: str) -> BatchReponse:
        raise NotImplementedError("Vertex AI batch is not implemented yet")
