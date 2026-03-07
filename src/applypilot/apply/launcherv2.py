"""Apply orchestration: acquire jobs, run browser-use agents, track results.

This is the main entry point for the apply pipeline. It pulls jobs from
the database, launches Chrome + browser-use for each one, parses the
result, and updates the database. Supports parallel workers via --workers.
"""

import atexit
import asyncio
import contextlib
import inspect
import logging
import platform
import re
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from browser_use import Agent, Browser, ChatAnthropic, ChatBrowserUse, ChatGoogle, ChatOpenAI
from packaging import version
from rich.console import Console
from rich.live import Live

from applypilot import config
from applypilot.apply import prompt as prompt_mod
from applypilot.apply.chrome import (
    launch_chrome,
    cleanup_worker,
    kill_all_chrome,
    reset_worker_dir,
    cleanup_on_exit,
    BASE_CDP_PORT,
)
from applypilot.apply.dashboard import (
    init_worker,
    update_state,
    add_event,
    get_state,
    render_full,
    get_totals,
)
from applypilot.apply import db as launcherv2_db

logger = logging.getLogger(__name__)

# Re-export launcher DB helpers so existing CLI imports keep working.
acquire_job = launcherv2_db.acquire_job
mark_result = launcherv2_db.mark_result
release_lock = launcherv2_db.release_lock
mark_job = launcherv2_db.mark_job
reset_failed = launcherv2_db.reset_failed

# How often to poll the DB when the queue is empty (seconds)
POLL_INTERVAL = config.DEFAULTS["poll_interval"]

# Thread-safe shutdown coordination
_stop_event = threading.Event()

# Track active browser-use runs for skip (Ctrl+C) handling
_active_runs: dict[int, threading.Event] = {}
_active_lock = threading.Lock()

# Register cleanup on exit
atexit.register(cleanup_on_exit)
if platform.system() != "Windows":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))


# ---------------------------------------------------------------------------
# Utility modes (--gen, --mark-applied, --mark-failed, --reset-failed)
# ---------------------------------------------------------------------------


def gen_prompt(
    target_url: str, min_score: int = 7, model: str = "gemini-3-flash-preview", worker_id: int = 0
) -> Path | None:
    """Generate a prompt file for manual debugging.

    Returns:
        Path to the generated prompt file, or None if no job found.
    """
    job = acquire_job(target_url=target_url, min_score=min_score, worker_id=worker_id)
    if not job:
        return None

    # Read resume text
    resume_path = job.get("tailored_resume_path")
    txt_path = Path(resume_path).with_suffix(".txt") if resume_path else None
    resume_text = ""
    if txt_path and txt_path.exists():
        resume_text = txt_path.read_text(encoding="utf-8")

    prompt = prompt_mod.build_prompt(job=job, tailored_resume=resume_text)

    # Release the lock so the job stays available
    release_lock(job["url"])

    # Write prompt file
    config.ensure_dirs()
    site_slug = (job.get("site") or "unknown")[:20].replace(" ", "_")
    prompt_file = config.LOG_DIR / f"prompt_{site_slug}_{job['title'][:30].replace(' ', '_')}.txt"
    prompt_file.write_text(prompt, encoding="utf-8")

    return prompt_file


# ---------------------------------------------------------------------------
# Per-job execution
# ---------------------------------------------------------------------------


class _JobCancelled(Exception):
    """Raised when a worker is interrupted and the current job should be skipped."""


def _as_text(value) -> str:
    """Convert values to compact text while preserving useful content."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(_as_text(v) for v in value if _as_text(v))
    return str(value)


def _extract_agent_output(result_obj) -> tuple[str, int]:
    """Extract final text and approximate action count from browser-use output."""
    if result_obj is None:
        return "", 0

    action_count = 0
    text_parts: list[str] = []

    for field in ("final_result", "result", "final_response", "extracted_content", "all_results", "errors"):
        value = getattr(result_obj, field, None)
        if callable(value):
            with contextlib.suppress(Exception):
                value = value()
        rendered = _as_text(value).strip()
        if rendered:
            text_parts.append(rendered)

    actions = getattr(result_obj, "model_actions", None)
    if callable(actions):
        with contextlib.suppress(Exception):
            actions = actions()
    if actions is None:
        actions = getattr(result_obj, "actions", None)
        if callable(actions):
            with contextlib.suppress(Exception):
                actions = actions()
    if actions is not None:
        with contextlib.suppress(Exception):
            action_count = len(actions)

    if not text_parts:
        text_parts.append(_as_text(result_obj).strip())

    output = "\n\n".join(p for p in text_parts if p).strip()
    return output, action_count


def _extract_usage_cost_usd(usage_obj) -> float:
    """Extract total USD cost from browser-use usage objects."""
    if usage_obj is None:
        return 0.0

    if isinstance(usage_obj, dict):
        for key in ("total_cost", "total_cost_usd"):
            value = usage_obj.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    for field in ("total_cost", "total_cost_usd"):
        value = getattr(usage_obj, field, None)
        if callable(value):
            with contextlib.suppress(Exception):
                value = value()
        if isinstance(value, (int, float)):
            return float(value)

    return 0.0


def _build_browser_use_task(prompt: str) -> str:
    """Wrap the existing prompt with browser-use backend instructions."""
    adapter = """
== EXECUTION ADAPTER ==
You are running in browser-use (not Playwright MCP).
If instructions mention `browser_*` or `mcp__playwright__*`, execute the equivalent browser-use action.
Keep all safety/business rules exactly as written.
Your final response MUST include exactly one terminal status line in this format:
- RESULT:APPLIED
- RESULT:EXPIRED
- RESULT:CAPTCHA
- RESULT:LOGIN_ISSUE
- RESULT:FAILED:reason
"""
    return f"{adapter.strip()}\n\n{prompt}"


async def _close_browser(browser) -> None:
    """Close browser object for whichever browser-use API version is installed."""
    if browser is None:
        return
    for method_name in ("close", "stop", "aclose"):
        method = getattr(browser, method_name, None)
        if method is None:
            continue
        try:
            result = method()
            if inspect.isawaitable(result):
                await result
            return
        except Exception:
            continue


def _build_browser(
    *,
    worker_id: int,
    port: int,
    headless: bool,
    chrome_instance_path: str | None = None,
) -> Browser:
    """Construct a browser session using browser-use's direct Browser kwargs API."""
    cdp_url = f"http://127.0.0.1:{port}"
    user_data_dir = f"/tmp/browser-use-worker-{worker_id}"
    return Browser(
        cdp_url=cdp_url,
        headless=headless,
        user_data_dir=user_data_dir,
        executable_path=chrome_instance_path,
    )


def _build_llm(model: str) -> object:
    """Construct a deterministic browser-use LLM class from model name."""
    normalized_model = model.strip().lower()
    if "gemini" in normalized_model or "gemma" in normalized_model:
        return ChatGoogle(model=model)

    if "claude" in normalized_model or normalized_model.startswith("anthropic/"):
        return ChatAnthropic(model=model)

    if "gpt" in normalized_model or normalized_model.startswith(("o1", "o3", "o4", "codex", "chatgpt")):
        return ChatOpenAI(model=model)

    raise ValueError(f"Unsupported model: {model}")

async def _run_browser_use_agent(
    task: str,
    worker_id: int,
    port: int,
    headless: bool,
    model: str,
    cancel_event: threading.Event,
) -> tuple[str, int, float]:
    """Execute a browser-use agent and return text output, actions, and cost."""
    browser = _build_browser(
        worker_id=worker_id,
        port=port,
        headless=headless,
        chrome_instance_path=config.get_chrome_path(),
    )
    try:
        llm = _build_llm(model=model)
        agent = Agent(task=task, llm=llm, browser=browser, calculate_cost=True)

        run_task = asyncio.create_task(agent.run())
        while not run_task.done():
            if cancel_event.is_set():
                run_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await run_task
                raise _JobCancelled
            await asyncio.sleep(0.25)

        result_obj = await run_task
        output, action_count = _extract_agent_output(result_obj)
        # Native browser-use cost tracking:
        # 1) history.usage from agent.run()
        # 2) token_cost_service usage summary fallback
        usage = getattr(result_obj, "usage", None)
        if callable(usage):
            with contextlib.suppress(Exception):
                usage = usage()
        cost_usd = _extract_usage_cost_usd(usage)
        if cost_usd <= 0:
            token_cost_service = getattr(agent, "token_cost_service", None)
            get_usage_summary = getattr(token_cost_service, "get_usage_summary", None)
            if callable(get_usage_summary):
                with contextlib.suppress(Exception):
                    get_usage_summary = get_usage_summary()
            if inspect.isawaitable(get_usage_summary):
                get_usage_summary = await get_usage_summary
            cost_usd = _extract_usage_cost_usd(get_usage_summary)
        return output, action_count, cost_usd
    finally:
        await _close_browser(browser)


def run_job(
    job: dict,
    port: int,
    worker_id: int = 0,
    model: str = "gemini-3-flash-preview",
    dry_run: bool = False,
    headless: bool = False,
) -> tuple[str, int]:
    """Run one job application via browser-use.

    Returns:
        Tuple of (status_string, duration_ms). Status is one of:
        'applied', 'expired', 'captcha', 'login_issue',
        'failed:reason', or 'skipped'.
    """
    resume_path = job.get("tailored_resume_path")
    txt_path = Path(resume_path).with_suffix(".txt") if resume_path else None
    resume_text = ""
    if txt_path and txt_path.exists():
        resume_text = txt_path.read_text(encoding="utf-8")

    prompt = prompt_mod.build_prompt(
        job=job,
        tailored_resume=resume_text,
        dry_run=dry_run,
    )
    task = _build_browser_use_task(prompt)

    worker_dir = reset_worker_dir(worker_id)

    update_state(
        worker_id,
        status="applying",
        job_title=job["title"],
        company=job.get("site", ""),
        score=job.get("fit_score", 0),
        start_time=time.time(),
        actions=0,
        last_action="starting",
    )
    add_event(f"[W{worker_id}] Starting: {job['title'][:40]} @ {job.get('site', '')}")

    worker_log = config.LOG_DIR / f"worker-{worker_id}.log"
    ts_header = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_header = (
        f"\n{'=' * 60}\n"
        f"[{ts_header}] {job['title']} @ {job.get('site', '')}\n"
        f"URL: {job.get('application_url') or job['url']}\n"
        f"Score: {job.get('fit_score', 'N/A')}/10\n"
        f"Worker dir: {worker_dir}\n"
        f"{'=' * 60}\n"
    )

    start = time.time()
    cancel_event = threading.Event()
    with _active_lock:
        _active_runs[worker_id] = cancel_event

    try:
        output, action_count, cost_usd = asyncio.run(
            _run_browser_use_agent(
                task=task,
                worker_id=worker_id,
                port=port,
                headless=headless,
                model=model,
                cancel_event=cancel_event,
            )
        )
    except _JobCancelled:
        return "skipped", int((time.time() - start) * 1000)
    except Exception as e:
        duration_ms = int((time.time() - start) * 1000)
        err_text = str(e).strip() or e.__class__.__name__
        full_trace = traceback.format_exc()
        logger.exception("Worker %d apply run failed for %s", worker_id, job.get("application_url") or job.get("url"))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_log = config.LOG_DIR / f"browser_use_{ts}_w{worker_id}_{job.get('site', 'unknown')[:20]}_error.txt"

        with open(worker_log, "a", encoding="utf-8") as lf:
            lf.write(log_header)
            lf.write("ERROR:\n")
            lf.write(full_trace)
            lf.write("\n")
        error_log.write_text(f"{log_header}\nERROR:\n{full_trace}\n", encoding="utf-8")

        add_event(f"[W{worker_id}] ERROR: {err_text[:40]}")
        update_state(worker_id, status="failed", last_action=f"ERROR: {err_text[:25]}")
        return f"failed:{err_text}\n{full_trace}", duration_ms
    finally:
        with _active_lock:
            _active_runs.pop(worker_id, None)

    elapsed = int(time.time() - start)
    duration_ms = int((time.time() - start) * 1000)

    with open(worker_log, "a", encoding="utf-8") as lf:
        lf.write(log_header)
        lf.write(output + "\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_log = config.LOG_DIR / f"browser_use_{ts}_w{worker_id}_{job.get('site', 'unknown')[:20]}.txt"
    job_log.write_text(output, encoding="utf-8")

    if action_count > 0:
        update_state(worker_id, actions=action_count, last_action=f"{action_count} action(s)")
    if cost_usd > 0:
        ws = get_state(worker_id)
        prev_cost = ws.total_cost if ws else 0.0
        update_state(worker_id, total_cost=prev_cost + cost_usd)

    def _clean_reason(s: str) -> str:
        return re.sub(r'[*`"]+$', "", s).strip()

    normalized = output.upper()
    for result_status in ["APPLIED", "EXPIRED", "CAPTCHA", "LOGIN_ISSUE"]:
        if f"RESULT:{result_status}" in normalized:
            add_event(f"[W{worker_id}] {result_status} ({elapsed}s): {job['title'][:30]}")
            update_state(worker_id, status=result_status.lower(), last_action=f"{result_status} ({elapsed}s)")
            return result_status.lower(), duration_ms

    for out_line in output.splitlines():
        if "RESULT:FAILED" not in out_line.upper():
            continue
        match = re.search(r"RESULT:FAILED(?::\s*(.+))?$", out_line, flags=re.IGNORECASE)
        reason = _clean_reason((match.group(1) if match else "") or "unknown")
        promote_to_status = {"captcha", "expired", "login_issue"}
        if reason in promote_to_status:
            add_event(f"[W{worker_id}] {reason.upper()} ({elapsed}s): {job['title'][:30]}")
            update_state(worker_id, status=reason, last_action=f"{reason.upper()} ({elapsed}s)")
            return reason, duration_ms
        add_event(f"[W{worker_id}] FAILED ({elapsed}s): {reason[:30]}")
        update_state(worker_id, status="failed", last_action=f"FAILED: {reason[:25]}")
        return f"failed:{reason}", duration_ms

    add_event(f"[W{worker_id}] NO RESULT ({elapsed}s)")
    update_state(worker_id, status="failed", last_action=f"no result ({elapsed}s)")
    return "failed:no_result_line", duration_ms


# ---------------------------------------------------------------------------
# Permanent failure classification
# ---------------------------------------------------------------------------

PERMANENT_FAILURES: set[str] = {
    "expired",
    "captcha",
    "login_issue",
    "not_eligible_location",
    "not_eligible_salary",
    "already_applied",
    "account_required",
    "not_a_job_application",
    "unsafe_permissions",
    "unsafe_verification",
    "sso_required",
    "site_blocked",
    "cloudflare_blocked",
    "blocked_by_cloudflare",
}

PERMANENT_PREFIXES: tuple[str, ...] = ("site_blocked", "cloudflare", "blocked_by")


def _is_permanent_failure(result: str) -> bool:
    """Determine if a failure should never be retried."""
    reason = result.split(":", 1)[-1] if ":" in result else result
    return (
        result in PERMANENT_FAILURES
        or reason in PERMANENT_FAILURES
        or any(reason.startswith(p) for p in PERMANENT_PREFIXES)
    )


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------


def worker_loop(
    worker_id: int = 0,
    limit: int = 1,
    target_url: str | None = None,
    min_score: int = 7,
    headless: bool = False,
    model: str = "gemini-3-flash-preview",
    dry_run: bool = False,
) -> tuple[int, int]:
    """Run jobs sequentially until limit is reached or queue is empty.

    Args:
        worker_id: Numeric worker identifier.
        limit: Max jobs to process (0 = continuous).
        target_url: Apply to a specific URL.
        min_score: Minimum fit_score threshold.
        headless: Run Chrome headless.
        model: browser-use model hint.
        dry_run: Don't click Submit.

    Returns:
        Tuple of (applied_count, failed_count).
    """
    applied = 0
    failed = 0
    continuous = limit == 0
    jobs_done = 0
    empty_polls = 0
    port = BASE_CDP_PORT + worker_id

    while not _stop_event.is_set():
        if not continuous and jobs_done >= limit:
            break

        update_state(worker_id, status="idle", job_title="", company="", last_action="waiting for job", actions=0)

        job = acquire_job(target_url=target_url, min_score=min_score, worker_id=worker_id)
        if not job:
            if not continuous:
                add_event(f"[W{worker_id}] Queue empty")
                update_state(worker_id, status="done", last_action="queue empty")
                break
            empty_polls += 1
            update_state(worker_id, status="idle", last_action=f"polling ({empty_polls})")
            if empty_polls == 1:
                add_event(f"[W{worker_id}] Queue empty, polling every {POLL_INTERVAL}s...")
            # Use Event.wait for interruptible sleep
            if _stop_event.wait(timeout=POLL_INTERVAL):
                break  # Stop was requested during wait
            continue

        empty_polls = 0

        chrome_proc = None
        try:
            add_event(f"[W{worker_id}] Launching Chrome...")
            chrome_proc = launch_chrome(worker_id, port=port, headless=headless)

            result, duration_ms = run_job(
                job,
                port=port,
                worker_id=worker_id,
                model=model,
                dry_run=dry_run,
                headless=headless,
            )

            if result == "skipped":
                release_lock(job["url"])
                add_event(f"[W{worker_id}] Skipped: {job['title'][:30]}")
                continue
            elif result == "applied":
                mark_result(job["url"], "applied", duration_ms=duration_ms)
                applied += 1
                update_state(worker_id, jobs_applied=applied, jobs_done=applied + failed)
            else:
                reason = result.split(":", 1)[-1] if ":" in result else result
                mark_result(
                    job["url"], "failed", reason, permanent=_is_permanent_failure(result), duration_ms=duration_ms
                )
                failed += 1
                update_state(worker_id, jobs_failed=failed, jobs_done=applied + failed)

        except KeyboardInterrupt:
            release_lock(job["url"])
            if _stop_event.is_set():
                break
            add_event(f"[W{worker_id}] Job skipped (Ctrl+C)")
            continue
        except Exception as e:
            logger.exception("Worker %d launcher error", worker_id)
            add_event(f"[W{worker_id}] Launcher error: {str(e)[:40]}")
            release_lock(job["url"])
            failed += 1
            update_state(worker_id, jobs_failed=failed)
        finally:
            if chrome_proc:
                cleanup_worker(worker_id, chrome_proc)

        jobs_done += 1
        if target_url:
            break

    update_state(worker_id, status="done", last_action="finished")
    return applied, failed


# ---------------------------------------------------------------------------
# Main entry point (called from cli.py)
# ---------------------------------------------------------------------------


def main(
    limit: int = 1,
    target_url: str | None = None,
    min_score: int = 7,
    headless: bool = False,
    model: str = "gemini-3-flash-preview",
    dry_run: bool = False,
    continuous: bool = False,
    poll_interval: int = 60,
    workers: int = 1,
) -> None:
    """Launch the apply pipeline.

    Args:
        limit: Max jobs to apply to (0 or with continuous=True means run forever).
        target_url: Apply to a specific URL.
        min_score: Minimum fit_score threshold.
        headless: Run Chrome in headless mode.
        model: browser-use model hint.
        dry_run: Don't click Submit.
        continuous: Run forever, polling for new jobs.
        poll_interval: Seconds between DB polls when queue is empty.
        workers: Number of parallel workers (default 1).
    """
    global POLL_INTERVAL
    POLL_INTERVAL = poll_interval
    _stop_event.clear()

    config.ensure_dirs()
    console = Console()

    if continuous:
        effective_limit = 0
        mode_label = "continuous"
    else:
        effective_limit = limit
        mode_label = f"{limit} jobs"

    # Initialize dashboard for all workers
    for i in range(workers):
        init_worker(i)

    worker_label = f"{workers} worker{'s' if workers > 1 else ''}"
    console.print(f"Launching apply pipeline ({mode_label}, {worker_label}, poll every {POLL_INTERVAL}s)...")
    console.print("[dim]Ctrl+C = skip current job(s) | Ctrl+C x2 = stop[/dim]")

    # Double Ctrl+C handler
    _ctrl_c_count = 0

    def _sigint_handler(sig, frame):
        nonlocal _ctrl_c_count
        _ctrl_c_count += 1
        if _ctrl_c_count == 1:
            console.print("\n[yellow]Skipping current job(s)... (Ctrl+C again to STOP)[/yellow]")
            with _active_lock:
                for cancel_event in _active_runs.values():
                    cancel_event.set()
            kill_all_chrome()
        else:
            console.print("\n[red bold]STOPPING[/red bold]")
            _stop_event.set()
            with _active_lock:
                for cancel_event in _active_runs.values():
                    cancel_event.set()
            kill_all_chrome()
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        with Live(render_full(), console=console, refresh_per_second=2) as live:
            # Daemon thread for display refresh only (no business logic)
            _dashboard_running = True

            def _refresh():
                while _dashboard_running:
                    live.update(render_full())
                    time.sleep(0.5)

            refresh_thread = threading.Thread(target=_refresh, daemon=True)
            refresh_thread.start()

            if workers == 1:
                # Single worker — run directly in main thread
                total_applied, total_failed = worker_loop(
                    worker_id=0,
                    limit=effective_limit,
                    target_url=target_url,
                    min_score=min_score,
                    headless=headless,
                    model=model,
                    dry_run=dry_run,
                )
            else:
                # Multi-worker — distribute limit across workers
                if effective_limit:
                    base = effective_limit // workers
                    extra = effective_limit % workers
                    limits = [base + (1 if i < extra else 0) for i in range(workers)]
                else:
                    limits = [0] * workers  # continuous mode

                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="apply-worker") as executor:
                    futures = {
                        executor.submit(
                            worker_loop,
                            worker_id=i,
                            limit=limits[i],
                            target_url=target_url,
                            min_score=min_score,
                            headless=headless,
                            model=model,
                            dry_run=dry_run,
                        ): i
                        for i in range(workers)
                    }

                    results: list[tuple[int, int]] = []
                    for future in as_completed(futures):
                        wid = futures[future]
                        try:
                            results.append(future.result())
                        except Exception:
                            logger.exception("Worker %d crashed", wid)
                            results.append((0, 0))

                total_applied = sum(r[0] for r in results)
                total_failed = sum(r[1] for r in results)

            _dashboard_running = False
            refresh_thread.join(timeout=2)
            live.update(render_full())

        totals = get_totals()
        console.print(f"\n[bold]Done: {total_applied} applied, {total_failed} failed (${totals['cost']:.3f})[/bold]")
        console.print(f"Logs: {config.LOG_DIR}")

    except KeyboardInterrupt:
        pass
    finally:
        _stop_event.set()
        kill_all_chrome()
