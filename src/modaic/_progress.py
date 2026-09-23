from __future__ import annotations

from types import TracebackType
from typing import Any

from tqdm.auto import tqdm

from .errors import ModaicTimeoutError
from .types import Alignment, BatchDecision


class JobProgress:
    """Render server-reported progress without changing polling or job state."""

    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled
        self._bar: tqdm[Any] | None = None
        self._snapshot: tuple[int, int | None, str] | None = None

    def __enter__(self) -> JobProgress:
        return self

    def update(self, job: BatchDecision | Alignment) -> None:
        if not self._enabled:
            return
        if isinstance(job, BatchDecision):
            label, unit = "Batch decisions", "examples"
            current = job.progress.completed + job.progress.failed
            total: int | None = job.progress.total
            status = f"{job.status}, {job.progress.failed} failed"
        else:
            label, unit = "Alignment", "metric calls"
            current = (job.progress.metric_calls or 0) if job.progress else 0
            total = job.progress.max_metric_calls if job.progress else None
            stage = job.progress.stage if job.progress else job.phase
            status = f"{job.status}, {stage}"
        snapshot = (current, total, status)
        if snapshot == self._snapshot:
            return
        self._snapshot = snapshot
        if self._bar is None:
            self._bar = tqdm(
                total=total,
                desc=f"{label} {job.id[:8]}",
                unit=unit,
                dynamic_ncols=True,
                leave=True,
            )
        self._bar.total = total
        self._bar.set_postfix_str(status, refresh=False)
        self._bar.update(current - self._bar.n)
        self._bar.refresh()

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        if self._bar is not None:
            if exc is not None:
                status = (
                    "wait timed out" if isinstance(exc, ModaicTimeoutError) else "wait interrupted"
                )
                self._bar.set_postfix_str(status, refresh=False)
            self._bar.close()
