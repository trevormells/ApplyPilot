"""Per-stage log capture for pipeline runs."""

from __future__ import annotations

import contextlib
from datetime import datetime
import logging
from pathlib import Path
import threading
from typing import Iterator

from rich.console import Console

from applypilot import config

_STAGE_LOGGER_NAMES: dict[str, tuple[str, ...]] = {
    "discover": ("applypilot.runner.pipeline", "applypilot.discovery"),
    "enrich": ("applypilot.runner.pipeline", "applypilot.enrichment"),
    "score": ("applypilot.runner.pipeline", "applypilot.scoring.scorer"),
    "tailor": ("applypilot.runner.pipeline", "applypilot.scoring.tailor"),
    "cover": ("applypilot.runner.pipeline", "applypilot.scoring.cover_letter"),
    "pdf": ("applypilot.runner.pipeline", "applypilot.scoring.pdf"),
}

_FILE_FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")


def stage_log_path(stage: str) -> Path:
    """Return the main log file path for a pipeline stage."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    return config.LOG_DIR / f"{stage}.log"


class StageConsole:
    """Proxy a Rich console while mirroring prints into a stage-local file console."""

    def __init__(self, terminal_console: Console):
        self._terminal_console = terminal_console
        self._local = threading.local()

    @property
    def terminal_console(self) -> Console:
        return self._terminal_console

    @contextlib.contextmanager
    def capture(self, file_console: Console) -> Iterator[None]:
        """Mirror console output in the current thread to the given file console."""
        stack = list(getattr(self._local, "file_consoles", ()))
        stack.append(file_console)
        self._local.file_consoles = stack
        try:
            yield
        finally:
            stack.pop()
            self._local.file_consoles = stack

    def print(self, *args, **kwargs) -> None:
        self._terminal_console.print(*args, **kwargs)
        stack = getattr(self._local, "file_consoles", ())
        if stack:
            stack[-1].print(*args, **kwargs)

    def log_only(self, *args, **kwargs) -> None:
        """Write only to the active stage log when stage capture is enabled."""
        stack = getattr(self._local, "file_consoles", ())
        if stack:
            stack[-1].print(*args, **kwargs)
            return
        self._terminal_console.print(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._terminal_console, name)


def _logger_writes_to_path(stage_logger: logging.Logger, path: Path) -> bool:
    target = str(path.resolve())
    for handler in stage_logger.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == target:
            return True
    return False


@contextlib.contextmanager
def capture_stage_output(stage: str, console: StageConsole) -> Iterator[Path]:
    """Capture a stage's log records and pipeline console output into its log file."""
    log_path = stage_log_path(stage)
    logger_names = _STAGE_LOGGER_NAMES.get(stage, ())

    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(
            f"\n{'=' * 80}\n"
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Stage: {stage}\n"
            f"{'=' * 80}\n"
        )
        stream.flush()

        file_console = Console(file=stream, force_terminal=False, color_system=None, width=console.width)
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.INFO)
        handler.setFormatter(_FILE_FORMATTER)

        attached_loggers: list[logging.Logger] = []
        for logger_name in logger_names:
            stage_logger = logging.getLogger(logger_name)
            if _logger_writes_to_path(stage_logger, log_path):
                continue
            stage_logger.addHandler(handler)
            attached_loggers.append(stage_logger)

        try:
            with console.capture(file_console):
                yield log_path
        finally:
            handler.flush()
            for stage_logger in attached_loggers:
                stage_logger.removeHandler(handler)
