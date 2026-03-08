"""ApplyPilot Pipeline Orchestrator.

Runs pipeline stages in sequence or concurrently (streaming mode).

Usage (via CLI):
    applypilot run                        # all stages, sequential
    applypilot run --stream               # all stages, concurrent
    applypilot run discover enrich        # specific stages
    applypilot run score tailor cover     # LLM-only stages
    applypilot run --dry-run              # preview without executing
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from applypilot.config import load_env, ensure_dirs
from applypilot.database import count_pending_detail, get_connection, get_stats, init_db
from applypilot.llm_cost import LLMCostTracker, clear_llm_cost_tracker, install_llm_cost_tracker, llm_cost_stage

from .pipeline_dashboard import PipelineDashboard
from .stage_logging import StageConsole, capture_stage_output

log = logging.getLogger(__name__)
console = StageConsole(Console())


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGE_ORDER = ("discover", "enrich", "score", "tailor", "cover", "pdf")

STAGE_META: dict[str, dict] = {
    "discover": {"desc": "Job discovery (JobSpy + Workday + smart extract)"},
    "enrich": {"desc": "Detail enrichment (full descriptions + apply URLs)"},
    "score": {"desc": "LLM scoring (fit 1-10)"},
    "tailor": {"desc": "Resume tailoring (LLM + validation)"},
    "cover": {"desc": "Cover letter generation"},
    "pdf": {"desc": "PDF conversion (tailored resumes + cover letters)"},
}

# Upstream dependency: a stage only finishes when its upstream is done AND
# it has no remaining pending work.
_UPSTREAM: dict[str, str | None] = {
    "discover": None,
    "enrich": "discover",
    "score": "enrich",
    "tailor": "score",
    "cover": "tailor",
    "pdf": "cover",
}


# ---------------------------------------------------------------------------
# Individual stage runners
# ---------------------------------------------------------------------------


def _run_discover(workers: int = 1, dashboard: PipelineDashboard | None = None) -> dict:
    """Stage: Job discovery — JobSpy, Workday, and smart-extract scrapers."""
    stats: dict = {"jobspy": None, "workday": None, "smartextract": None}

    # JobSpy
    console.log_only("  [cyan]JobSpy full crawl...[/cyan]")
    try:
        from applypilot.discovery.jobspy import run_discovery

        run_discovery()
        stats["jobspy"] = "ok"
        if dashboard is not None:
            dashboard.advance_stage("discover", summary="JobSpy crawl complete")
    except Exception as e:
        log.error("JobSpy crawl failed: %s", e)
        console.log_only(f"  [red]JobSpy error:[/red] {e}")
        stats["jobspy"] = f"error: {e}"
        if dashboard is not None:
            dashboard.advance_stage("discover", summary=f"JobSpy error: {e}")

    # Workday corporate scraper
    console.log_only("  [cyan]Workday corporate scraper...[/cyan]")
    try:
        from applypilot.discovery.workday import run_workday_discovery

        run_workday_discovery(workers=workers)
        stats["workday"] = "ok"
        if dashboard is not None:
            dashboard.advance_stage("discover", summary="Workday crawl complete")
    except Exception as e:
        log.error("Workday scraper failed: %s", e)
        console.log_only(f"  [red]Workday error:[/red] {e}")
        stats["workday"] = f"error: {e}"
        if dashboard is not None:
            dashboard.advance_stage("discover", summary=f"Workday error: {e}")

    # Smart extract
    console.log_only("  [cyan]Smart extract (AI-powered scraping)...[/cyan]")
    try:
        from applypilot.discovery.smartextract import run_smart_extract

        run_smart_extract(workers=workers)
        stats["smartextract"] = "ok"
        if dashboard is not None:
            dashboard.advance_stage("discover", summary="Smart extract complete")
    except Exception as e:
        log.error("Smart extract failed: %s", e)
        console.log_only(f"  [red]Smart extract error:[/red] {e}")
        stats["smartextract"] = f"error: {e}"
        if dashboard is not None:
            dashboard.advance_stage("discover", summary=f"Smart extract error: {e}")

    return stats


def _run_enrich(workers: int = 1) -> dict:
    """Stage: Detail enrichment — scrape full descriptions and apply URLs."""
    try:
        from applypilot.enrichment.detail import run_enrichment

        run_enrichment(workers=workers)
        return {"status": "ok"}
    except Exception as e:
        log.error("Enrichment failed: %s", e)
        return {"status": f"error: {e}"}


def _run_score(workers: int = 1) -> dict:
    """Stage: LLM scoring — assign fit scores 1-10."""
    try:
        from applypilot.scoring.scorer import run_scoring

        run_scoring(workers=workers)
        return {"status": "ok"}
    except Exception as e:
        log.error("Scoring failed: %s", e)
        return {"status": f"error: {e}"}


def _run_tailor(min_score: int = 7, validation_mode: str = "normal", workers: int = 1) -> dict:
    """Stage: Resume tailoring — generate tailored resumes for high-fit jobs."""
    try:
        from applypilot.scoring.tailor import run_tailoring

        run_tailoring(min_score=min_score, validation_mode=validation_mode, workers=workers)
        return {"status": "ok"}
    except Exception as e:
        log.error("Tailoring failed: %s", e)
        return {"status": f"error: {e}"}


def _run_cover(min_score: int = 7, validation_mode: str = "normal", workers: int = 1) -> dict:
    """Stage: Cover letter generation."""
    try:
        from applypilot.scoring.cover_letter import run_cover_letters

        run_cover_letters(min_score=min_score, validation_mode=validation_mode, workers=workers)
        return {"status": "ok"}
    except Exception as e:
        log.error("Cover letter generation failed: %s", e)
        return {"status": f"error: {e}"}


def _run_pdf() -> dict:
    """Stage: PDF conversion — convert tailored resumes and cover letters to PDF."""
    try:
        from applypilot.scoring.pdf import batch_convert

        batch_convert()
        return {"status": "ok"}
    except Exception as e:
        log.error("PDF conversion failed: %s", e)
        return {"status": f"error: {e}"}


# Map stage names to their runner functions
_STAGE_RUNNERS: dict[str, callable] = {
    "discover": _run_discover,
    "enrich": _run_enrich,
    "score": _run_score,
    "tailor": _run_tailor,
    "cover": _run_cover,
    "pdf": _run_pdf,
}


# ---------------------------------------------------------------------------
# Stage resolution
# ---------------------------------------------------------------------------


def _resolve_stages(stage_names: list[str]) -> list[str]:
    """Resolve 'all' and validate/order stage names."""
    if "all" in stage_names:
        return list(STAGE_ORDER)

    resolved = []
    for name in stage_names:
        if name not in STAGE_META:
            console.print(f"[red]Unknown stage:[/red] '{name}'. Available: {', '.join(STAGE_ORDER)}, all")
            raise SystemExit(1)
        if name not in resolved:
            resolved.append(name)

    # Maintain canonical order
    return [s for s in STAGE_ORDER if s in resolved]


# ---------------------------------------------------------------------------
# Streaming pipeline helpers
# ---------------------------------------------------------------------------


class _StageTracker:
    """Thread-safe tracker for which stages have finished producing work."""

    def __init__(self):
        self._events: dict[str, threading.Event] = {stage: threading.Event() for stage in STAGE_ORDER}
        self._results: dict[str, dict] = {}
        self._lock = threading.Lock()

    def mark_done(self, stage: str, result: dict | None = None) -> None:
        with self._lock:
            self._results[stage] = result or {"status": "ok"}
        self._events[stage].set()

    def is_done(self, stage: str) -> bool:
        return self._events[stage].is_set()

    def wait(self, stage: str, timeout: float | None = None) -> bool:
        return self._events[stage].wait(timeout=timeout)

    def get_results(self) -> dict[str, dict]:
        with self._lock:
            return dict(self._results)


# SQL to count pending work for each stage
_PENDING_SQL: dict[str, str] = {
    "score": "SELECT COUNT(*) FROM jobs WHERE full_description IS NOT NULL AND fit_score IS NULL",
    "tailor": (
        "SELECT COUNT(*) FROM jobs WHERE fit_score >= ? "
        "AND full_description IS NOT NULL "
        "AND tailored_resume_path IS NULL "
        "AND COALESCE(tailor_attempts, 0) < 5"
    ),
    "cover": (
        "SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL "
        "AND (cover_letter_path IS NULL OR cover_letter_path = '') "
        "AND COALESCE(cover_attempts, 0) < 5"
    ),
    "pdf": ("SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL AND tailored_resume_path LIKE '%.txt'"),
}

# How long to sleep between polling loops in streaming mode (seconds)
_STREAM_POLL_INTERVAL = 10


def _count_pending(stage: str, min_score: int = 7) -> int:
    """Count pending work items for a stage."""
    if stage == "enrich":
        return count_pending_detail()
    sql = _PENDING_SQL.get(stage)
    if sql is None:
        return 0
    conn = get_connection()
    if "?" in sql:
        return conn.execute(sql, (min_score,)).fetchone()[0]
    return conn.execute(sql).fetchone()[0]


def _format_pending_detail_summary(stats: dict) -> str:
    """Format actionable and blocked enrichment counts for terminal output."""
    summary = f"{stats['pending_detail']} actionable enrichment jobs"
    blocked = stats.get("pending_detail_blocked", 0)
    blocked_sites = stats.get("pending_detail_blocked_sites", [])
    if blocked:
        site_summary = ", ".join(f"{site}:{count}" for site, count in blocked_sites)
        summary += f", {blocked} skipped blocked-site jobs ({site_summary})"
    return summary


def _run_stage_streaming(
    stage: str,
    tracker: _StageTracker,
    stop_event: threading.Event,
    min_score: int = 7,
    workers: int = 1,
    validation_mode: str = "normal",
    dashboard: PipelineDashboard | None = None,
) -> None:
    """Run a single stage in streaming mode: loop until upstream done + no work.

    For discover: runs once, then marks done.
    For all others: polls DB for pending work, runs the batch processor,
    and repeats until upstream is done and no pending work remains.
    """
    runner = _STAGE_RUNNERS[stage]
    kwargs: dict = {}
    if stage in ("tailor", "cover"):
        kwargs["min_score"] = min_score
        kwargs["validation_mode"] = validation_mode
    if stage in ("discover", "enrich", "score", "tailor", "cover"):
        kwargs["workers"] = workers

    upstream = _UPSTREAM[stage]

    with llm_cost_stage(stage):
        with capture_stage_output(stage, console):
            console.log_only(f"\n{'=' * 70}")
            console.log_only(f"  [bold]STAGE: {stage}[/bold] — {STAGE_META[stage]['desc']} (streaming)")
            console.log_only(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
            console.log_only(f"{'=' * 70}")

            if stage == "discover":
                # Discover runs once (its sub-scrapers already do their full crawl)
                try:
                    if dashboard is not None:
                        dashboard.start_stage(stage)
                        kwargs["dashboard"] = dashboard
                    result = runner(**kwargs)
                    tracker.mark_done(stage, result)
                except Exception as e:
                    log.exception("Stage '%s' crashed", stage)
                    tracker.mark_done(stage, {"status": f"error: {e}"})
                    console.log_only(f"\n  [red]STAGE FAILED:[/red] {e}")
                    if dashboard is not None:
                        dashboard.finish_stage(stage, "error", summary=str(e))
                else:
                    console.log_only(f"\n  Stage '{stage}' completed — ok")
                    if dashboard is not None:
                        dashboard.finish_stage(stage, "ok", summary="Discovery complete")
                return

            # For downstream stages: loop until upstream done + no pending work
            passes = 0
            if dashboard is not None:
                dashboard.start_stage(stage)
            while not stop_event.is_set():
                # Wait a bit for upstream to produce some work before first run
                if passes == 0 and upstream and not tracker.is_done(upstream):
                    tracker.wait(upstream, timeout=_STREAM_POLL_INTERVAL)

                pending = _count_pending(stage, min_score)

                if pending > 0:
                    try:
                        runner(**kwargs)
                        passes += 1
                    except Exception as e:
                        log.error("Stage '%s' error (pass %d): %s", stage, passes, e)
                        passes += 1
                else:
                    upstream_done = upstream is None or tracker.is_done(upstream)
                    if upstream_done:
                        break
                    if stop_event.wait(timeout=_STREAM_POLL_INTERVAL):
                        break

            tracker.mark_done(stage, {"status": "ok", "passes": passes})
            console.log_only(f"\n  Stage '{stage}' completed after {passes} pass(es)")
            if dashboard is not None:
                dashboard.finish_stage(stage, "ok", summary=f"{passes} pass(es)")


# ---------------------------------------------------------------------------
# Pipeline orchestrators
# ---------------------------------------------------------------------------


def _run_sequential(
    ordered: list[str],
    min_score: int,
    workers: int = 1,
    validation_mode: str = "normal",
    dashboard: PipelineDashboard | None = None,
) -> dict:
    """Execute stages one at a time (original behavior)."""
    results: list[dict] = []
    errors: dict[str, str] = {}
    pipeline_start = time.time()

    for name in ordered:
        if dashboard is not None:
            dashboard.start_stage(name)
        with llm_cost_stage(name):
            with capture_stage_output(name, console):
                meta = STAGE_META[name]
                console.log_only(f"\n{'=' * 70}")
                console.log_only(f"  [bold]STAGE: {name}[/bold] — {meta['desc']}")
                console.log_only(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
                console.log_only(f"{'=' * 70}")

                t0 = time.time()
                runner = _STAGE_RUNNERS[name]

                try:
                    kwargs: dict = {}
                    if name in ("tailor", "cover"):
                        kwargs["min_score"] = min_score
                        kwargs["validation_mode"] = validation_mode
                    if name in ("discover", "enrich", "score", "tailor", "cover"):
                        kwargs["workers"] = workers
                    if dashboard is not None and name == "discover":
                        kwargs["dashboard"] = dashboard
                    result = runner(**kwargs)
                    elapsed = time.time() - t0

                    status = "ok"
                    if isinstance(result, dict):
                        status = result.get("status", "ok")
                        if name == "discover":
                            sub_errors = [
                                f"{k}: {v}" for k, v in result.items() if isinstance(v, str) and v.startswith("error")
                            ]
                            if sub_errors:
                                status = "partial"

                except Exception as e:
                    elapsed = time.time() - t0
                    status = f"error: {e}"
                    log.exception("Stage '%s' crashed", name)
                    console.log_only(f"\n  [red]STAGE FAILED:[/red] {e}")

                results.append({"stage": name, "status": status, "elapsed": elapsed})
                if status not in ("ok", "partial"):
                    errors[name] = status

                console.log_only(f"\n  Stage '{name}' completed in {elapsed:.1f}s — {status}")
                if dashboard is not None:
                    summary = meta["desc"] if status == "ok" else status
                    dashboard.finish_stage(
                        name,
                        status if status in ("ok", "partial", "error") else "error",
                        summary=summary,
                    )

    total_elapsed = time.time() - pipeline_start
    return {"stages": results, "errors": errors, "elapsed": total_elapsed}


def _run_streaming(
    ordered: list[str],
    min_score: int,
    workers: int = 1,
    validation_mode: str = "normal",
    dashboard: PipelineDashboard | None = None,
) -> dict:
    """Execute stages concurrently with DB as conveyor belt."""
    tracker = _StageTracker()
    stop_event = threading.Event()
    pipeline_start = time.time()

    if dashboard is None:
        console.print("\n  [bold cyan]STREAMING MODE[/bold cyan] — stages run concurrently")
        console.print(f"  Poll interval: {_STREAM_POLL_INTERVAL}s\n")

    # Mark stages NOT in `ordered` as done so downstream doesn't wait for them
    for stage in STAGE_ORDER:
        if stage not in ordered:
            tracker.mark_done(stage, {"status": "skipped"})

    # Launch each stage in its own thread
    threads: dict[str, threading.Thread] = {}
    start_times: dict[str, float] = {}

    for name in ordered:
        start_times[name] = time.time()
        t = threading.Thread(
            target=_run_stage_streaming,
            args=(name, tracker, stop_event, min_score, workers, validation_mode, dashboard),
            name=f"stage-{name}",
            daemon=True,
        )
        threads[name] = t
        t.start()
        if dashboard is None:
            console.print(f"  [dim]Started thread:[/dim] {name}")

    # Wait for all threads to finish
    try:
        for name in ordered:
            threads[name].join()
            elapsed = time.time() - start_times[name]
            if dashboard is None:
                console.print(f"  [green]Completed:[/green] {name} ({elapsed:.1f}s)")
    except KeyboardInterrupt:
        if dashboard is None:
            console.print("\n[yellow]Interrupted — stopping stages...[/yellow]")
        stop_event.set()
        for t in threads.values():
            t.join(timeout=10)

    total_elapsed = time.time() - pipeline_start

    # Build results from tracker
    all_results = tracker.get_results()
    results: list[dict] = []
    errors: dict[str, str] = {}

    for name in ordered:
        r = all_results.get(name, {"status": "unknown"})
        elapsed = time.time() - start_times.get(name, pipeline_start)
        status = r.get("status", "ok")

        results.append({"stage": name, "status": status, "elapsed": elapsed})
        if status not in ("ok", "partial", "skipped"):
            errors[name] = status

    return {"stages": results, "errors": errors, "elapsed": total_elapsed}


def run_pipeline(
    stages: list[str] | None = None,
    min_score: int = 7,
    dry_run: bool = False,
    stream: bool = False,
    workers: int = 1,
    validation_mode: str = "normal",
) -> dict:
    """Run pipeline stages.

    Args:
        stages: List of stage names, or None / ["all"] for full pipeline.
        min_score: Minimum fit score for tailor/cover stages.
        dry_run: If True, preview stages without executing.
        stream: If True, run stages concurrently (streaming mode).
        workers: Number of parallel threads for discovery/enrichment stages.

    Returns:
        Dict with keys: stages (list of result dicts), errors (dict), elapsed (float).
    """
    # Bootstrap
    load_env()
    ensure_dirs()
    init_db()

    # Resolve stages
    if stages is None:
        stages = ["all"]
    ordered = _resolve_stages(stages)

    mode = "streaming" if stream else "sequential"
    pre_stats = get_stats()

    if dry_run:
        console.print()
        console.print(
            Panel.fit(
                f"[bold]ApplyPilot Pipeline[/bold] ({mode})",
                border_style="blue",
            )
        )
        console.print(f"  Min score:  {min_score}")
        console.print(f"  Workers:    {workers}")
        console.print(f"  Validation: {validation_mode}")
        console.print(f"  Stages:     {' -> '.join(ordered)}")
        console.print(f"  DB:        {pre_stats['total']} jobs, {_format_pending_detail_summary(pre_stats)}")
        console.print(f"\n  [yellow]DRY RUN[/yellow] — would execute ({mode}):")
        for name in ordered:
            meta = STAGE_META[name]
            console.print(f"    {name:<12s}  {meta['desc']}")
        console.print("\n  No changes made.")
        return {"stages": [], "errors": {}, "elapsed": 0.0}

    cost_tracker = LLMCostTracker(stages=ordered)
    install_llm_cost_tracker(cost_tracker)

    use_dashboard = bool(console.is_terminal)
    try:
        if use_dashboard:
            dashboard = PipelineDashboard(
                ordered,
                STAGE_META,
                mode=mode,
                min_score=min_score,
                workers=workers,
                validation_mode=validation_mode,
                pre_total_jobs=pre_stats["total"],
                pre_pending_detail=pre_stats["pending_detail"],
                pre_pending_detail_blocked=pre_stats["pending_detail_blocked"],
                pre_pending_detail_blocked_sites=pre_stats["pending_detail_blocked_sites"],
                terminal_console=console,
                cost_tracker=cost_tracker,
            )
            dashboard.start()
            try:
                with Live(dashboard, console=console.terminal_console, refresh_per_second=4, transient=False):
                    if stream:
                        result = _run_streaming(
                            ordered,
                            min_score,
                            workers=workers,
                            validation_mode=validation_mode,
                            dashboard=dashboard,
                        )
                    else:
                        result = _run_sequential(
                            ordered,
                            min_score,
                            workers=workers,
                            validation_mode=validation_mode,
                            dashboard=dashboard,
                        )
            finally:
                dashboard.stop()
                console.print()
        else:
            console.print()
            console.print(
                Panel.fit(
                    f"[bold]ApplyPilot Pipeline[/bold] ({mode})",
                    border_style="blue",
                )
            )
            console.print(f"  Min score:  {min_score}")
            console.print(f"  Workers:    {workers}")
            console.print(f"  Validation: {validation_mode}")
            console.print(f"  Stages:     {' -> '.join(ordered)}")
            console.print(f"  DB:        {pre_stats['total']} jobs, {_format_pending_detail_summary(pre_stats)}")
            if stream:
                result = _run_streaming(ordered, min_score, workers=workers, validation_mode=validation_mode)
            else:
                result = _run_sequential(ordered, min_score, workers=workers, validation_mode=validation_mode)

        cost_snapshot = cost_tracker.snapshot()
    finally:
        clear_llm_cost_tracker(cost_tracker)

    result["llm_cost_estimate"] = float(cost_snapshot["total_cost"])
    result["llm_calls"] = int(cost_snapshot["total_calls"])
    result["llm_cost_unknown_calls"] = int(cost_snapshot["unknown_cost_calls"])
    result["llm_cost_by_stage"] = cost_snapshot["stage_costs"]

    # Summary table
    console.print(f"\n{'=' * 70}")
    summary = Table(title="Pipeline Summary", show_header=True, header_style="bold")
    summary.add_column("Stage", style="bold")
    summary.add_column("Status")
    summary.add_column("Time", justify="right")

    for r in result["stages"]:
        elapsed_str = f"{r['elapsed']:.1f}s"
        status_display = r["status"][:30]
        if r["status"] == "ok":
            style = "green"
        elif r["status"] in ("partial", "skipped"):
            style = "yellow"
        else:
            style = "red"
        summary.add_row(r["stage"], f"[{style}]{status_display}[/{style}]", elapsed_str)

    summary.add_row("", "", "")
    summary.add_row("[bold]Total[/bold]", "", f"[bold]{result['elapsed']:.1f}s[/bold]")
    console.print(summary)
    llm_summary = f"  [bold]Estimated LLM cost:[/bold] ${result['llm_cost_estimate']:.3f}"
    if result["llm_cost_unknown_calls"]:
        llm_summary += f" ({result['llm_cost_unknown_calls']} unpriced call(s))"
    llm_summary += f" across {result['llm_calls']} call(s)"
    console.print(llm_summary)

    # Final DB stats
    final = get_stats()
    console.print("\n  [bold]DB Final State:[/bold]")
    console.print(f"    Total jobs:     {final['total']}")
    console.print(f"    With desc:      {final['with_description']}")
    console.print(f"    Scored:         {final['scored']}")
    console.print(f"    Tailored:       {final['tailored']}")
    console.print(f"    Cover letters:  {final['with_cover_letter']}")
    console.print(f"    Ready to apply: {final['ready_to_apply']}")
    console.print(f"    Applied:        {final['applied']}")
    console.print(f"{'=' * 70}\n")

    return result
