"""Helpers for routing browser-use artifacts into a dedicated log subdirectory."""

from __future__ import annotations

from datetime import datetime
import inspect
import logging
from pathlib import Path
import re
from typing import Any, Callable

from applypilot import config


def _site_component(site: str | None) -> str:
    value = (site or "unknown").strip()
    value = re.sub(r"[\\/]+", "_", value)
    value = re.sub(r"\s+", " ", value)
    value = value[:20].strip()
    return value or "unknown"


def browser_use_log_stem(site: str | None, worker_id: int, now: datetime | None = None) -> Path:
    """Build a stable per-run path stem for browser-use artifacts."""
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    config.BROWSER_USE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return config.BROWSER_USE_LOG_DIR / f"browser_use_{timestamp}_w{worker_id}_{_site_component(site)}"


def browser_use_text_path(log_stem: Path) -> Path:
    """Return the text summary path for a browser-use run."""
    return log_stem.with_suffix(".txt")


def browser_use_error_path(log_stem: Path) -> Path:
    """Return the error trace path for a browser-use run."""
    return log_stem.with_name(f"{log_stem.name}_error.txt")


def browser_use_history_path(log_stem: Path) -> Path:
    """Return the structured history path for a browser-use run."""
    return log_stem.with_name(f"{log_stem.name}_history.json")


def browser_use_agent_kwargs(agent_ctor: Callable[..., Any], log_stem: Path) -> dict[str, str]:
    """Attach supported browser-use persistence kwargs without pinning to one version."""
    try:
        parameters = inspect.signature(agent_ctor).parameters
    except (TypeError, ValueError):
        return {}

    if "save_conversation_path" in parameters:
        return {"save_conversation_path": str(log_stem)}
    return {}


def save_browser_use_history(result_obj: Any, history_path: Path, logger: logging.Logger | None = None) -> bool:
    """Persist browser-use history when the installed version exposes save_to_file()."""
    save_to_file = getattr(result_obj, "save_to_file", None)
    if not callable(save_to_file):
        return False

    try:
        save_to_file(str(history_path))
    except Exception:
        if logger is not None:
            logger.warning("Failed to persist browser-use history to %s", history_path, exc_info=True)
        return False
    return True
