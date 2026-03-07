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

from browser_use import Agent, Browser, ChatAnthropic, ChatGoogle, ChatOpenAI
from browser_use.agent.views import (
    AgentHistoryList,
    AgentOutput,
    AgentStructuredOutput,
)

from rich.console import Console
from rich.live import Live

from applypilot import config
from applypilot.apply import promptv2 as prompt_mod
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
from applypilot.apply.captcha import build_captcha_tools
from applypilot.apply import db as launcherv2_db

logger = logging.getLogger(__name__)
logging.getLogger("browser_use").setLevel(logging.WARNING)

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


def _extract_agent_output(result_obj: AgentHistoryList[AgentStructuredOutput]) -> str:
    """Extract final text and action count from browser-use AgentHistoryList."""

    final_result = result_obj.final_result()
    if final_result:
        return final_result.strip()

    extracted_content = [value.strip() for value in result_obj.extracted_content() if value and value.strip()]
    errors = [value.strip() for value in result_obj.errors() if value and value.strip()]

    text_parts: list[str] = []
    if extracted_content:
        text_parts.append("\n\n".join(extracted_content))
    if errors:
        text_parts.append("\n".join(f"ERROR: {error}" for error in errors))

    output = "\n\n".join(text_parts).strip()
    if not output:
        output = str(result_obj)

    return output


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


def _format_last_action(agent_out: AgentOutput) -> str:
    """Format the last action from an AgentOutput into a short readable label."""
    if not agent_out or not agent_out.action:
        return "thinking"
    last = agent_out.action[-1]
    action_dict = last.model_dump(exclude_none=True, mode="json")
    action_name = next((k for k in action_dict if k != "interacted_element"), None)
    if not action_name:
        return "unknown"
    params = action_dict.get(action_name) or {}
    if not isinstance(params, dict):
        return action_name.replace("_", " ")
    if action_name in ("navigate", "go_to_url", "open_url"):
        url = str(params.get("url", ""))
        domain = url.split("/")[2] if "//" in url else url
        return f"\u2192 {domain[:30]}"
    if action_name in ("input_text", "type", "fill"):
        text = str(params.get("text", params.get("value", "")))[:22]
        return f"type: {text}"
    if action_name in ("click", "click_element", "click_element_by_index"):
        return "click"
    if action_name == "scroll":
        direction = "down" if params.get("down", True) else "up"
        return f"scroll {direction}"
    if action_name in ("done", "finish", "complete"):
        text = str(params.get("text", params.get("message", "")))[:20]
        return f"done: {text}" if text else "done"
    if action_name in ("extract_content", "extract"):
        return "extract"
    if action_name in ("search_google", "search"):
        query = str(params.get("query", ""))[:20]
        return f"search: {query}"
    return action_name.replace("_", " ")[:30]


async def _run_browser_use_agent(
    task: str,
    worker_id: int,
    port: int,
    headless: bool,
    model: str,
    cancel_event: threading.Event,
    cost_baseline: float = 0.0,
) -> AgentHistoryList[AgentStructuredOutput]:
    """Execute a browser-use agent and return the raw AgentHistoryList result."""
    cdp_url = f"http://127.0.0.1:{port}"
    user_data_dir = f"/tmp/browser-use-worker-{worker_id}"
    browser = Browser(
        cdp_url=cdp_url,
        headless=headless,
        user_data_dir=user_data_dir,
        executable_path=config.get_chrome_path(),
    )
    try:
        llm = _build_llm(model=model)
        captcha_tools = build_captcha_tools()
        agent = Agent(task=task, llm=llm, browser=browser, calculate_cost=True, tools=captcha_tools)

        def _on_step(*args):
            # browser_use callback signature: (browser_state, agent_output, step_number)
            step_n = next((a for a in args if isinstance(a, int)), 0)
            agent_out = next((a for a in args if isinstance(a, AgentOutput)), None)
            label = _format_last_action(agent_out) if agent_out else f"step {step_n}"
            update_state(worker_id, actions=step_n, last_action=label)
            add_event(f"[W{worker_id}] step {step_n}: {label}")
            try:
                running_cost = agent.history.usage.total_cost
                if running_cost > 0:
                    update_state(worker_id, total_cost=cost_baseline + running_cost)
            except Exception:
                pass

        if hasattr(agent, "register_new_step_callback"):
            agent.register_new_step_callback = _on_step

        run_task = asyncio.create_task(agent.run())
        while not run_task.done():
            if cancel_event.is_set():
                run_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await run_task
                raise _JobCancelled
            await asyncio.sleep(0.25)

        return await run_task
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

    ws_before = get_state(worker_id)
    cost_baseline = ws_before.total_cost if ws_before else 0.0

    start = time.time()
    cancel_event = threading.Event()
    with _active_lock:
        _active_runs[worker_id] = cancel_event

    try:
        result_obj: AgentHistoryList[AgentStructuredOutput] = asyncio.run(
            _run_browser_use_agent(
                task=prompt,
                worker_id=worker_id,
                port=port,
                headless=headless,
                model=model,
                cancel_event=cancel_event,
                cost_baseline=cost_baseline,
            )
        )
        output = _extract_agent_output(result_obj)
        action_count = len(result_obj.model_actions())
        cost_usd = result_obj.usage.total_cost
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
    site_slug = job.get("site", "unknown")[:20]
    job_log = config.LOG_DIR / f"browser_use_{ts}_w{worker_id}_{site_slug}.txt"
    steps_text = "\n".join(result_obj.agent_steps())
    job_log.write_text(
        f"{output}\n\n{'=' * 60}\nAGENT HISTORY\n{'=' * 60}\n{steps_text}",
        encoding="utf-8",
    )

    if action_count > 0:
        update_state(worker_id, actions=action_count, last_action=f"{action_count} action(s)")
    if cost_usd > 0:
        update_state(worker_id, total_cost=cost_baseline + cost_usd)

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
            job_url = job.get("application_url") or job["url"]
            chrome_proc = launch_chrome(worker_id, port=port, headless=headless, start_url=job_url)

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
        # Mute all StreamHandlers while the Live dashboard is active so that
        # stray log lines don't corrupt the Rich terminal output.
        _muted_handlers: list[tuple[logging.StreamHandler, int]] = []
        for _h in logging.root.handlers:
            if isinstance(_h, logging.StreamHandler):
                _muted_handlers.append((_h, _h.level))
                _h.setLevel(logging.CRITICAL + 1)

        # Let Rich own the refresh loop to avoid concurrent redraw races.
        with Live(console=console, refresh_per_second=2, get_renderable=render_full) as live:
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

            # Force one last render with final totals before leaving Live mode.
            live.refresh()

        # Restore StreamHandler levels now that the Live dashboard is gone.
        for _h, _lvl in _muted_handlers:
            _h.setLevel(_lvl)

        totals = get_totals()
        console.print(f"\n[bold]Done: {total_applied} applied, {total_failed} failed (${totals['cost']:.3f})[/bold]")
        console.print(f"Logs: {config.LOG_DIR}")

    except KeyboardInterrupt:
        pass
    finally:
        _stop_event.set()
        kill_all_chrome()
