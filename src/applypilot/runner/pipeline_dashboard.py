"""Live Rich dashboard for `applypilot run` pipeline execution."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import threading
import time
from pathlib import Path

from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from applypilot import config
from applypilot.database import get_connection

from .stage_logging import stage_log_path


_STATUS_STYLE: dict[str, str] = {
    "pending": "dim",
    "active": "bold yellow",
    "ok": "bold green",
    "partial": "bold yellow",
    "error": "bold red",
    "skipped": "dim",
}

_STATUS_LABEL: dict[str, str] = {
    "pending": "PENDING",
    "active": "ACTIVE",
    "ok": "DONE",
    "partial": "PARTIAL",
    "error": "ERROR",
    "skipped": "SKIPPED",
}


@dataclass
class StageView:
    """Mutable state for one pipeline stage in the terminal dashboard."""

    name: str
    desc: str
    unit: str = "jobs"
    status: str = "pending"
    processed: int = 0
    total: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    summary: str = ""
    log_offset: int = 0


def _count_pending(stage: str, min_score: int) -> int:
    conn = get_connection()
    queries: dict[str, tuple[str, tuple[object, ...]]] = {
        "enrich": ("SELECT COUNT(*) FROM jobs WHERE detail_scraped_at IS NULL", ()),
        "score": ("SELECT COUNT(*) FROM jobs WHERE full_description IS NOT NULL AND fit_score IS NULL", ()),
        "tailor": (
            "SELECT COUNT(*) FROM jobs WHERE fit_score >= ? "
            "AND full_description IS NOT NULL "
            "AND tailored_resume_path IS NULL "
            "AND COALESCE(tailor_attempts, 0) < 5",
            (min_score,),
        ),
        "cover": (
            "SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL "
            "AND (cover_letter_path IS NULL OR cover_letter_path = '') "
            "AND COALESCE(cover_attempts, 0) < 5",
            (),
        ),
    }
    query = queries.get(stage)
    if query is None:
        return _count_pending_pdf()
    sql, params = query
    return int(conn.execute(sql, params).fetchone()[0])


def _count_pending_pdf() -> int:
    if not config.TAILORED_DIR.exists():
        return 0
    txt_files = sorted(config.TAILORED_DIR.glob("*.txt"))
    candidates = [path for path in txt_files if not path.name.endswith("_JOB.txt")]
    return sum(1 for path in candidates if not path.with_suffix(".pdf").exists())


def _initial_total(stage: str, min_score: int) -> tuple[int, str]:
    if stage == "discover":
        return 3, "sources"
    if stage == "pdf":
        return _count_pending_pdf(), "files"
    return _count_pending(stage, min_score), "jobs"


def _processed_for_stage(stage: StageView, min_score: int) -> int:
    if stage.name == "discover":
        return min(stage.total, stage.processed)
    current_pending = _count_pending(stage.name, min_score)
    return max(0, min(stage.total, stage.total - current_pending))


def _format_elapsed(started_at: float | None, finished_at: float | None = None) -> str:
    if started_at is None:
        return ""
    end = finished_at or time.time()
    seconds = max(0, int(end - started_at))
    minutes, seconds = divmod(seconds, 60)
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _tail_stage_lines(path: Path, offset: int, limit: int = 5) -> list[str]:
    if not path.exists():
        return []

    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()

    text = data.decode("utf-8", errors="ignore")
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if set(line) == {"="}:
            continue
        if line.startswith("[") and "Stage:" in line:
            continue
        lines.append(line)
    return lines[-limit:]


class PipelineDashboard:
    """Live pipeline dashboard renderable and state manager."""

    def __init__(
        self,
        ordered: list[str],
        stage_meta: dict[str, dict],
        *,
        mode: str,
        min_score: int,
        workers: int,
        validation_mode: str,
        pre_total_jobs: int,
        pre_pending_detail: int,
        terminal_console: Console,
    ):
        self._ordered = ordered
        self._mode = mode
        self._min_score = min_score
        self._workers = workers
        self._validation_mode = validation_mode
        self._pre_total_jobs = pre_total_jobs
        self._pre_pending_detail = pre_pending_detail
        self._terminal_console = terminal_console
        self._started_at = time.time()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._recent_lines: deque[str] = deque(maxlen=5)
        self._recent_stage = ""
        self._stages: dict[str, StageView] = {
            name: StageView(name=name, desc=stage_meta[name]["desc"]) for name in ordered
        }

    def start(self) -> None:
        self._stop_event.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, name="pipeline-dashboard", daemon=True)
        self._poll_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2)
            self._poll_thread = None
        self.refresh()

    def start_stage(self, name: str) -> None:
        total, unit = _initial_total(name, self._min_score)
        log_path = stage_log_path(name)
        log_offset = log_path.stat().st_size if log_path.exists() else 0
        with self._lock:
            stage = self._stages[name]
            stage.status = "active"
            stage.total = total
            stage.unit = unit
            stage.processed = 0
            stage.summary = ""
            stage.started_at = time.time()
            stage.finished_at = None
            stage.log_offset = log_offset
            self._recent_stage = name
            self._recent_lines.clear()

    def advance_stage(self, name: str, step: int = 1, summary: str | None = None) -> None:
        with self._lock:
            stage = self._stages[name]
            stage.processed = min(stage.total, stage.processed + step)
            if summary:
                stage.summary = summary

    def finish_stage(self, name: str, status: str, summary: str = "") -> None:
        with self._lock:
            stage = self._stages[name]
            stage.status = status
            if stage.name != "discover":
                stage.processed = _processed_for_stage(stage, self._min_score)
            else:
                stage.processed = min(stage.total, stage.processed or stage.total)
            stage.summary = summary
            stage.finished_at = time.time()
            self._recent_stage = name
        self.refresh()

    def refresh(self) -> None:
        with self._lock:
            active = [stage for stage in self._stages.values() if stage.status == "active"]
            focus = max(active, key=lambda stage: stage.started_at or 0, default=None)
            if focus is None:
                completed = [stage for stage in self._stages.values() if stage.finished_at is not None]
                focus = max(completed, key=lambda stage: stage.finished_at or 0, default=None)
            if focus is not None:
                if focus.name != "discover":
                    focus.processed = _processed_for_stage(focus, self._min_score)
                self._recent_stage = focus.name
                self._recent_lines = deque(
                    _tail_stage_lines(stage_log_path(focus.name), focus.log_offset, limit=5),
                    maxlen=5,
                )

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(0.4):
            self.refresh()

    def __rich__(self):
        with self._lock:
            snapshot = {
                "stages": [self._stages[name] for name in self._ordered],
                "recent_stage": self._recent_stage,
                "recent_lines": list(self._recent_lines),
                "started_at": self._started_at,
            }
        return self._render(snapshot)

    def _render(self, snapshot: dict) -> Layout:
        header = self._render_header(snapshot)
        stages = self._render_stage_table(snapshot["stages"])
        active = self._render_active_panel(snapshot["stages"])
        recent = self._render_recent_panel(snapshot["recent_stage"], snapshot["recent_lines"])

        layout = Layout()
        layout.split_column(
            Layout(header, size=8),
            Layout(name="body", ratio=1),
            Layout(recent, size=9),
        )
        layout["body"].split_row(
            Layout(stages, ratio=3),
            Layout(active, ratio=2),
        )
        return layout

    def _render_header(self, snapshot: dict) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        elapsed = _format_elapsed(snapshot["started_at"])
        grid.add_row(
            f"[bold]Mode:[/bold] {self._mode}",
            f"[bold]Elapsed:[/bold] {elapsed}",
        )
        grid.add_row(
            f"[bold]Workers:[/bold] {self._workers}",
            f"[bold]Min score:[/bold] {self._min_score}",
        )
        grid.add_row(
            f"[bold]Validation:[/bold] {self._validation_mode}",
            f"[bold]DB:[/bold] {self._pre_total_jobs} jobs, {self._pre_pending_detail} pending enrichment",
        )
        grid.add_row(
            f"[bold]Stages:[/bold] {' -> '.join(self._ordered)}",
            f"[bold]Completed:[/bold] {sum(1 for stage in snapshot['stages'] if stage.status in ('ok', 'partial', 'error', 'skipped'))}/{len(self._ordered)}",
        )
        return Panel(grid, title="ApplyPilot Pipeline", border_style="blue")

    def _render_stage_table(self, stages: list[StageView]) -> Panel:
        table = Table(expand=True, show_edge=False)
        table.add_column("Stage", style="bold")
        table.add_column("Status", width=10)
        table.add_column("Progress", width=18, justify="right")
        table.add_column("Time", width=8, justify="right")
        table.add_column("Summary", overflow="fold")

        for stage in stages:
            label = _STATUS_LABEL.get(stage.status, stage.status.upper())
            style = _STATUS_STYLE.get(stage.status, "")
            progress = f"{stage.processed}/{stage.total} {stage.unit}" if stage.total or stage.processed else f"0 {stage.unit}"
            elapsed = _format_elapsed(stage.started_at, stage.finished_at)
            summary = stage.summary or stage.desc
            table.add_row(
                stage.name,
                Text(label, style=style),
                progress,
                elapsed,
                summary,
            )

        return Panel(table, title="Stages", border_style="cyan")

    def _render_active_panel(self, stages: list[StageView]) -> Panel:
        active = [stage for stage in stages if stage.status == "active"]
        if not active:
            done = [stage for stage in stages if stage.finished_at is not None]
            if not done:
                return Panel("Waiting for pipeline start", title="Current Stage", border_style="yellow")
            latest = max(done, key=lambda stage: stage.finished_at or 0)
            body = Table.grid(padding=(0, 1))
            body.add_row(f"[bold]{latest.name}[/bold]")
            body.add_row(f"Status: {_STATUS_LABEL.get(latest.status, latest.status.upper())}")
            body.add_row(f"Progress: {latest.processed}/{latest.total} {latest.unit}")
            if latest.summary:
                body.add_row(latest.summary)
            return Panel(body, title="Last Stage", border_style="green")

        focus = max(active, key=lambda stage: stage.started_at or 0)
        total = max(1, focus.total)
        body = Table.grid(padding=(0, 1))
        body.add_row(f"[bold]{focus.name}[/bold]")
        body.add_row(f"{focus.processed}/{focus.total} {focus.unit}")
        body.add_row(ProgressBar(total=total, completed=min(focus.processed, total), width=28))
        if focus.summary:
            body.add_row(focus.summary)
        body.add_row(f"Started: {datetime.fromtimestamp(focus.started_at or time.time()).strftime('%H:%M:%S')}")

        if len(active) > 1:
            names = ", ".join(stage.name for stage in active)
            title = "Active Stages"
            body.add_row(f"Also running: {names}")
        else:
            title = "Current Stage"

        return Panel(body, title=title, border_style="yellow")

    def _render_recent_panel(self, stage_name: str, lines: list[str]) -> Panel:
        if not lines:
            content = Text("No log lines yet.", style="dim")
        else:
            rendered = []
            for line in lines:
                style = "red" if "ERROR" in line or "FAILED" in line else "yellow" if "WARN" in line else ""
                rendered.append(Text(line, style=style))
            content = Group(*rendered)
        title = f"Recent Log Lines ({stage_name})" if stage_name else "Recent Log Lines"
        return Panel(content, title=title, border_style="magenta")
