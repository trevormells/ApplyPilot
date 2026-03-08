"""Track per-stage LiteLLM cost estimates during pipeline runs."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
import threading
from typing import Iterator


_current_stage: ContextVar[str | None] = ContextVar("applypilot_llm_stage", default=None)
_tracker_lock = threading.Lock()
_active_tracker: LLMCostTracker | None = None


class LLMCostTracker:
    """Thread-safe aggregate of estimated LLM spend for one pipeline run."""

    def __init__(self, *, stages: Iterable[str] = ()) -> None:
        self._lock = threading.Lock()
        self._stage_costs: dict[str, float] = {stage: 0.0 for stage in stages}
        self._stage_calls: dict[str, int] = {stage: 0 for stage in stages}
        self._stage_unknown_cost_calls: dict[str, int] = {stage: 0 for stage in stages}
        self._total_cost = 0.0
        self._total_calls = 0
        self._unknown_cost_calls = 0

    def record(self, cost: float | None, *, stage: str | None = None) -> None:
        """Record a successful LLM call, with an optional estimated USD cost."""
        resolved_stage = stage if stage is not None else _current_stage.get()
        with self._lock:
            self._total_calls += 1
            if resolved_stage:
                self._stage_calls[resolved_stage] = self._stage_calls.get(resolved_stage, 0) + 1

            if cost is None:
                self._unknown_cost_calls += 1
                if resolved_stage:
                    self._stage_unknown_cost_calls[resolved_stage] = (
                        self._stage_unknown_cost_calls.get(resolved_stage, 0) + 1
                    )
                return

            self._total_cost += cost
            if resolved_stage:
                self._stage_costs[resolved_stage] = self._stage_costs.get(resolved_stage, 0.0) + cost

    def snapshot(self) -> dict[str, object]:
        """Return a copy suitable for dashboard rendering and CLI summaries."""
        with self._lock:
            return {
                "total_cost": self._total_cost,
                "total_calls": self._total_calls,
                "unknown_cost_calls": self._unknown_cost_calls,
                "stage_costs": dict(self._stage_costs),
                "stage_calls": dict(self._stage_calls),
                "stage_unknown_cost_calls": dict(self._stage_unknown_cost_calls),
            }


def install_llm_cost_tracker(tracker: LLMCostTracker) -> None:
    """Make a tracker active for subsequent LLM calls in this process."""
    global _active_tracker
    with _tracker_lock:
        _active_tracker = tracker


def clear_llm_cost_tracker(tracker: LLMCostTracker | None = None) -> None:
    """Remove the active tracker if it matches the provided one."""
    global _active_tracker
    with _tracker_lock:
        if tracker is None or _active_tracker is tracker:
            _active_tracker = None


def get_llm_cost_tracker() -> LLMCostTracker | None:
    """Return the currently active tracker, if any."""
    with _tracker_lock:
        return _active_tracker


def record_llm_cost_estimate(cost: float | None) -> None:
    """Record a cost estimate against the active tracker, if one exists."""
    tracker = get_llm_cost_tracker()
    if tracker is None:
        return
    tracker.record(cost)


def bind_current_llm_cost_context(fn):
    """Wrap a callable so it runs inside the current ContextVar state."""
    ctx = copy_context()

    def _wrapped(*args, **kwargs):
        return ctx.run(fn, *args, **kwargs)

    return _wrapped


@contextmanager
def llm_cost_stage(stage: str) -> Iterator[None]:
    """Attribute nested LLM calls to the given pipeline stage."""
    token = _current_stage.set(stage)
    try:
        yield
    finally:
        _current_stage.reset(token)
