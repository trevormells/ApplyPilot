from datetime import datetime
from pathlib import Path

from applypilot import config
from applypilot.apply.browser_use_logs import (
    browser_use_agent_kwargs,
    browser_use_error_path,
    browser_use_history_path,
    browser_use_log_stem,
    browser_use_text_path,
    save_browser_use_history,
)


def test_browser_use_log_paths_are_scoped_to_subdirectory(monkeypatch) -> None:
    monkeypatch.setattr(config, "BROWSER_USE_LOG_DIR", Path("/tmp/applypilot-log-test/logs/browser_use"))
    stem = browser_use_log_stem("Thomson Reuters", worker_id=2, now=datetime(2026, 2, 27, 16, 18, 26))

    assert stem == Path("/tmp/applypilot-log-test/logs/browser_use/browser_use_20260227_161826_w2_Thomson Reuters")
    assert stem.parent.is_dir()
    assert browser_use_text_path(stem).name.endswith(".txt")
    assert browser_use_error_path(stem).name.endswith("_error.txt")
    assert browser_use_history_path(stem).name.endswith("_history.json")


def test_browser_use_agent_kwargs_only_uses_supported_save_path() -> None:
    def agent_with_save_path(task: str, save_conversation_path: str | None = None) -> None:
        del task, save_conversation_path

    def agent_without_save_path(task: str) -> None:
        del task

    stem = Path("/tmp/browser_use_20260227_161826_w0_Test")

    assert browser_use_agent_kwargs(agent_with_save_path, stem) == {"save_conversation_path": str(stem)}
    assert browser_use_agent_kwargs(agent_without_save_path, stem) == {}


def test_save_browser_use_history_calls_save_to_file() -> None:
    calls: list[str] = []

    class History:
        def save_to_file(self, path: str) -> None:
            calls.append(path)

    history_path = Path("/tmp/browser_use_history.json")

    assert save_browser_use_history(History(), history_path) is True
    assert calls == [str(history_path)]


def test_save_browser_use_history_noops_without_save_method() -> None:
    assert save_browser_use_history(object(), Path("/tmp/browser_use_history.json")) is False
