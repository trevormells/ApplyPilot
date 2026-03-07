import io

from rich.console import Console

from applypilot import config
from applypilot.llm_cost import LLMCostTracker, clear_llm_cost_tracker, install_llm_cost_tracker, record_llm_cost_estimate
from applypilot.runner import pipeline as pipeline_module
from applypilot.runner.pipeline_dashboard import PipelineDashboard


def test_run_sequential_attributes_recorded_cost_to_stage(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)

    tracker = LLMCostTracker(stages=("score",))
    install_llm_cost_tracker(tracker)

    def _fake_runner() -> dict:
        record_llm_cost_estimate(0.42)
        return {"status": "ok"}

    monkeypatch.setitem(pipeline_module._STAGE_RUNNERS, "score", _fake_runner)

    try:
        result = pipeline_module._run_sequential(["score"], min_score=7)
    finally:
        clear_llm_cost_tracker(tracker)

    snapshot = tracker.snapshot()

    assert result["stages"][0]["stage"] == "score"
    assert snapshot["stage_costs"]["score"] == 0.42
    assert snapshot["stage_calls"]["score"] == 1
    assert snapshot["total_cost"] == 0.42


def test_pipeline_dashboard_refresh_reads_tracker_costs(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)

    tracker = LLMCostTracker(stages=("score", "cover"))
    tracker.record(0.42, stage="score")
    tracker.record(None, stage="cover")

    dashboard = PipelineDashboard(
        ["score", "cover"],
        {
            "score": {"desc": "LLM scoring"},
            "cover": {"desc": "Cover letters"},
        },
        mode="sequential",
        min_score=7,
        workers=1,
        validation_mode="normal",
        pre_total_jobs=10,
        pre_pending_detail=0,
        pre_pending_detail_blocked=0,
        pre_pending_detail_blocked_sites=[],
        terminal_console=Console(file=io.StringIO(), force_terminal=False, color_system=None),
        cost_tracker=tracker,
    )

    dashboard.refresh()

    assert dashboard._stages["score"].llm_cost_estimate == 0.42
    assert dashboard._stages["score"].llm_calls == 1
    assert dashboard._stages["cover"].llm_unknown_cost_calls == 1

    render_console = Console(record=True, width=120, force_terminal=False, color_system=None)
    render_console.print(dashboard)
    rendered = render_console.export_text()

    assert "LLM est." in rendered
    assert "$0.420" in rendered
    assert "unpriced" in rendered
