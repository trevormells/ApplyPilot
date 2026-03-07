import io
import logging
from pathlib import Path

from rich.console import Console

from applypilot import config
from applypilot.stage_logging import StageConsole, capture_stage_output, stage_log_path


def test_stage_log_path_uses_main_log_dir(monkeypatch) -> None:
    monkeypatch.setattr(config, "LOG_DIR", Path("/tmp/applypilot-stage-logs"))

    assert stage_log_path("discover") == Path("/tmp/applypilot-stage-logs/discover.log")


def test_capture_stage_output_mirrors_console_and_stage_logger(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)

    terminal_buffer = io.StringIO()
    stage_console = StageConsole(Console(file=terminal_buffer, force_terminal=False, color_system=None))
    logger = logging.getLogger("applypilot.discovery.jobspy")
    logger.setLevel(logging.INFO)

    with capture_stage_output("discover", stage_console):
        stage_console.print("discover console line")
        logger.info("discover logger line")

    log_text = (tmp_path / "discover.log").read_text(encoding="utf-8")

    assert "discover console line" in terminal_buffer.getvalue()
    assert "discover console line" in log_text
    assert "discover logger line" in log_text


def test_log_only_skips_terminal_when_stage_capture_is_active(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)

    terminal_buffer = io.StringIO()
    stage_console = StageConsole(Console(file=terminal_buffer, force_terminal=False, color_system=None))

    with capture_stage_output("discover", stage_console):
        stage_console.log_only("file only line")

    log_text = (tmp_path / "discover.log").read_text(encoding="utf-8")

    assert "file only line" not in terminal_buffer.getvalue()
    assert "file only line" in log_text
