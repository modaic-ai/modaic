from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

logger = logging.getLogger(__name__)

EventKind = Literal["submitting", "shard_started", "shard_progress", "shard_completed", "shard_failed", "completed"]


@dataclass
class ShardEvent:
    kind: EventKind
    shard_index: Optional[int] = None
    total_shards: Optional[int] = None
    batch_id: Optional[str] = None
    percent: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)


ProgressCallback = Callable[[ShardEvent], None]


class BatchProgressDisplay:
    """Rich-based progress display. Pull-only observer: you call update().

    Falls back to a no-op if rich isn't installed or stdout isn't a tty.
    """

    def __init__(self, num_requests: int, provider: str, user_callback: Optional[ProgressCallback] = None):
        self.num_requests = num_requests
        self.provider = provider
        self.user_callback = user_callback
        self.num_cached = 0
        self.shard_percents: dict[int, int] = {}
        self.total_shards: Optional[int] = None
        self.state: str = "submitting"
        self.batch_ids: list[str] = []
        self.start_time = time.time()
        self.live = None
        self._enabled = sys.stdout.isatty()

    def __call__(self, event: ShardEvent) -> None:
        if event.kind == "submitting":
            self.state = "submitting"
            if "num_cached" in event.extra:
                self.num_cached = event.extra["num_cached"]
        elif event.kind == "shard_started":
            self.state = "running"
            self.total_shards = event.total_shards
            if event.batch_id and event.batch_id not in self.batch_ids:
                self.batch_ids.append(event.batch_id)
        elif event.kind == "shard_progress":
            if event.shard_index is not None and event.percent is not None:
                self.shard_percents[event.shard_index] = event.percent
        elif event.kind == "shard_completed":
            if event.shard_index is not None:
                self.shard_percents[event.shard_index] = 100
        elif event.kind == "shard_failed":
            self.state = "failed"
        elif event.kind == "completed":
            self.state = "completed"

        if self.user_callback is not None:
            try:
                self.user_callback(event)
            except Exception:
                logger.exception("user progress callback raised")

        if self.live is not None:
            self.live.update(self._make_panel())

    @property
    def aggregate_percent(self) -> Optional[int]:
        if not self.total_shards:
            return None
        total = sum(self.shard_percents.get(i, 0) for i in range(self.total_shards))
        return int(total / self.total_shards)

    def _make_panel(self):
        from rich.console import Group
        from rich.panel import Panel
        from rich.spinner import Spinner
        from rich.table import Table
        from rich.text import Text

        table = Table.grid(padding=(0, 4))
        table.add_column(style="cyan", justify="right")
        table.add_column(style="white", min_width=20)

        batch_line = (
            ", ".join(self.batch_ids) if self.batch_ids else "[dim]submitting...[/dim]"
        )
        table.add_row("Batch IDs:", batch_line)
        table.add_row("Provider:", f"[magenta]{self.provider}[/magenta]")
        table.add_row()
        table.add_row("Requests:", f"[bold]{self.num_requests}[/bold]")
        table.add_row("Cached:", f"[bold]{self.num_cached}[/bold]")
        if self.total_shards is not None:
            completed_shards = sum(1 for v in self.shard_percents.values() if v >= 100)
            table.add_row("Shards:", f"{completed_shards}/{self.total_shards}")

        status = self.state.lower()
        if status == "completed":
            styled = "[green]completed[/green]"
        elif status in ("failed", "cancelled", "expired"):
            styled = f"[red]{status}[/red]"
        else:
            styled = f"[yellow]{status}[/yellow]"
        table.add_row("Status:", styled)

        pct = self.aggregate_percent
        table.add_row("Progress:", f"[bold]{pct}%[/bold]" if pct is not None else "[dim]N/A[/dim]")

        elapsed = time.time() - self.start_time
        table.add_row("Elapsed:", f"[dim]{_format_elapsed(elapsed)}[/dim]")

        show_spinner = status not in ("completed", "failed", "cancelled", "expired")
        content = Group(table, Text(""), Spinner("dots", text=" Processing...") if show_spinner else Text(""))
        return Panel(content, title="[bold blue]Batch Processing[/bold blue]", border_style="blue", padding=(1, 2))

    async def run(self, task):
        if not self._enabled:
            return await task
        try:
            from rich.live import Live
        except ImportError:
            return await task

        import asyncio

        with Live(self._make_panel(), refresh_per_second=4) as live:
            self.live = live
            try:
                while not task.done():
                    live.update(self._make_panel())
                    await asyncio.sleep(0.5)
                return await task
            finally:
                self.live = None


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}m {secs}s"
    hours, remainder = divmod(int(seconds), 3600)
    mins, secs = divmod(remainder, 60)
    return f"{hours}h {mins}m {secs}s"


def make_progress(
    enabled: bool, num_requests: int, provider: str, user_callback: Optional[ProgressCallback]
) -> Optional[BatchProgressDisplay]:
    if not enabled:
        if user_callback is None:
            return None
        # user wants events but not the default display — wrap them
        display = BatchProgressDisplay(num_requests, provider, user_callback)
        display._enabled = False
        return display
    return BatchProgressDisplay(num_requests, provider, user_callback)
