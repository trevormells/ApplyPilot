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
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.live import Live

from applypilot import config
from applypilot.apply import prompt as prompt_mod
from applypilot.apply.chrome import (
    launch_chrome,
    cleanup_worker,
    kill_all_chrome,
    reset_worker_dir,
    setup_worker_profile,
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
from applypilot.database import get_connection

logger = logging.getLogger(__name__)


# Blocked sites loaded from config/sites.yaml
def _load_blocked():
    from applypilot.config import load_blocked_sites

    return load_blocked_sites()


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
# Database operations
# ---------------------------------------------------------------------------


def acquire_job(target_url: str | None = None, min_score: int = 7, worker_id: int = 0) -> dict | None:
    """Atomically acquire the next job to apply to.

    Args:
        target_url: Apply to a specific URL instead of picking from queue.
        min_score: Minimum fit_score threshold.
        worker_id: Worker claiming this job (for tracking).

    Returns:
        Job dict or None if the queue is empty.
    """
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")

        if target_url:
            like = f"%{target_url.split('?')[0].rstrip('/')}%"
            row = conn.execute(
                """
                SELECT url, title, site, application_url, tailored_resume_path,
                       fit_score, location, full_description, cover_letter_path
                FROM jobs
                WHERE (url = ? OR application_url = ? OR application_url LIKE ? OR url LIKE ?)
                  AND tailored_resume_path IS NOT NULL
                  AND apply_status != 'in_progress'
                LIMIT 1
            """,
                (target_url, target_url, like, like),
            ).fetchone()
        else:
            blocked_sites, blocked_patterns = _load_blocked()
            # Build parameterized filters to avoid SQL injection
            params: list = [min_score]
            site_clause = ""
            if blocked_sites:
                placeholders = ",".join("?" * len(blocked_sites))
                site_clause = f"AND site NOT IN ({placeholders})"
                params.extend(blocked_sites)
            url_clauses = ""
            if blocked_patterns:
                url_clauses = " ".join("AND url NOT LIKE ?" for _ in blocked_patterns)
                params.extend(blocked_patterns)
            row = conn.execute(
                f"""
                SELECT url, title, site, application_url, tailored_resume_path,
                       fit_score, location, full_description, cover_letter_path
                FROM jobs
                WHERE tailored_resume_path IS NOT NULL
                  AND (apply_status IS NULL OR apply_status = 'failed')
                  AND (apply_attempts IS NULL OR apply_attempts < ?)
                  AND fit_score >= ?
                  {site_clause}
                  {url_clauses}
                ORDER BY fit_score DESC, url
                LIMIT 1
            """,
                [config.DEFAULTS["max_apply_attempts"]] + params,
            ).fetchone()

        if not row:
            conn.rollback()
            return None

        # Skip manual ATS sites (unsolvable CAPTCHAs)
        from applypilot.config import is_manual_ats

        apply_url = row["application_url"] or row["url"]
        if is_manual_ats(apply_url):
            conn.execute(
                "UPDATE jobs SET apply_status = 'manual', apply_error = 'manual ATS' WHERE url = ?",
                (row["url"],),
            )
            conn.commit()
            logger.info("Skipping manual ATS: %s", row["url"][:80])
            return None

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            UPDATE jobs SET apply_status = 'in_progress',
                           agent_id = ?,
                           last_attempted_at = ?
            WHERE url = ?
        """,
            (f"worker-{worker_id}", now, row["url"]),
        )
        conn.commit()

        return dict(row)
    except Exception:
        conn.rollback()
        raise


def mark_result(
    url: str,
    status: str,
    error: str | None = None,
    permanent: bool = False,
    duration_ms: int | None = None,
    task_id: str | None = None,
) -> None:
    """Update a job's apply status in the database."""
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    if status == "applied":
        conn.execute(
            """
            UPDATE jobs SET apply_status = 'applied', applied_at = ?,
                           apply_error = NULL, agent_id = NULL,
                           apply_duration_ms = ?, apply_task_id = ?
            WHERE url = ?
        """,
            (now, duration_ms, task_id, url),
        )
    else:
        attempts = 99 if permanent else "COALESCE(apply_attempts, 0) + 1"
        conn.execute(
            f"""
            UPDATE jobs SET apply_status = ?, apply_error = ?,
                           apply_attempts = {attempts}, agent_id = NULL,
                           apply_duration_ms = ?, apply_task_id = ?
            WHERE url = ?
        """,
            (status, error or "unknown", duration_ms, task_id, url),
        )
    conn.commit()


def release_lock(url: str) -> None:
    """Release the in_progress lock without changing status."""
    conn = get_connection()
    conn.execute(
        "UPDATE jobs SET apply_status = NULL, agent_id = NULL WHERE url = ? AND apply_status = 'in_progress'",
        (url,),
    )
    conn.commit()


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


def mark_job(url: str, status: str, reason: str | None = None) -> None:
    """Manually mark a job's apply status in the database.

    Args:
        url: Job URL to mark.
        status: Either 'applied' or 'failed'.
        reason: Failure reason (only for status='failed').
    """
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    if status == "applied":
        conn.execute(
            """
            UPDATE jobs SET apply_status = 'applied', applied_at = ?,
                           apply_error = NULL, agent_id = NULL
            WHERE url = ?
        """,
            (now, url),
        )
    else:
        conn.execute(
            """
            UPDATE jobs SET apply_status = 'failed', apply_error = ?,
                           apply_attempts = 99, agent_id = NULL
            WHERE url = ?
        """,
            (reason or "manual", url),
        )
    conn.commit()


def reset_failed() -> int:
    """Reset all failed jobs so they can be retried.

    Returns:
        Number of jobs reset.
    """
    conn = get_connection()
    cursor = conn.execute("""
        UPDATE jobs SET apply_status = NULL, apply_error = NULL,
                       apply_attempts = 0, agent_id = NULL
        WHERE apply_status = 'failed'
          OR (apply_status IS NOT NULL AND apply_status != 'applied'
              AND apply_status != 'in_progress')
    """)
    conn.commit()
    return cursor.rowcount


# ---------------------------------------------------------------------------
# Per-job execution
# ---------------------------------------------------------------------------


class _JobCancelled(Exception):
    """Raised when a worker is interrupted and the current job should be skipped."""


def _filter_kwargs(callable_obj, kwargs: dict) -> dict:
    """Return only kwargs supported by callable_obj signature."""
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs

    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _supports_kwarg(callable_obj, name: str) -> bool:
    """Check whether a callable explicitly declares a kwarg name."""
    try:
        return name in inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False


def _maybe_call(attr) -> object | None:
    """Invoke zero-arg callables or return the value as-is."""
    if callable(attr):
        try:
            return attr()
        except Exception:
            return None
    return attr


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
        value = _maybe_call(getattr(result_obj, field, None))
        rendered = _as_text(value).strip()
        if rendered:
            text_parts.append(rendered)

    actions = _maybe_call(getattr(result_obj, "model_actions", None))
    if actions is None:
        actions = _maybe_call(getattr(result_obj, "actions", None))
    if actions is not None:
        with contextlib.suppress(Exception):
            action_count = len(actions)

    if not text_parts:
        text_parts.append(_as_text(result_obj).strip())

    output = "\n\n".join(p for p in text_parts if p).strip()
    return output, action_count


def _extract_cost_usd(result_obj) -> float:
    """Extract best-effort cost from result object."""
    if result_obj is None:
        return 0.0

    for field in ("total_cost_usd", "cost_usd", "total_cost"):
        value = _maybe_call(getattr(result_obj, field, None))
        if isinstance(value, (int, float)):
            return float(value)

    usage = _maybe_call(getattr(result_obj, "usage", None))
    if isinstance(usage, dict):
        for key in ("total_cost_usd", "cost_usd", "total_cost"):
            value = usage.get(key)
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


def _build_browser(Browser, BrowserConfig, worker_id: int, port: int, headless: bool):
    """Construct a browser instance while tolerating browser-use API drift."""
    profile_dir = setup_worker_profile(worker_id)

    base_kwargs: dict[str, object] = {}
    base_kwargs["headless"] = headless
    base_kwargs["cdp_url"] = f"http://127.0.0.1:{port}"
    base_kwargs["user_data_dir"] = str(profile_dir)
    with contextlib.suppress(Exception):
        base_kwargs["chrome_instance_path"] = config.get_chrome_path()

    if BrowserConfig is not None and _supports_kwarg(Browser, "config"):
        config_kwargs = _filter_kwargs(BrowserConfig, base_kwargs)
        return Browser(config=BrowserConfig(**config_kwargs))

    filtered = _filter_kwargs(Browser, base_kwargs)
    with contextlib.suppress(TypeError):
        return Browser(**filtered)
    return Browser()


def _build_llm(
    model: str,
    ChatBrowserUse,
    ChatGoogle=None,
    ChatAnthropic=None,
    ChatOpenAI=None,
):
    """Construct the correct browser-use LLM class for the requested model."""
    model = (model or "").strip()
    model_lower = model.lower()

    browser_use_models = {"bu-latest", "bu-1-0", "bu-2-0"}
    if not model or model_lower in browser_use_models or model_lower.startswith("browser-use/"):
        llm_kwargs = _filter_kwargs(ChatBrowserUse, {"model": model or "bu-latest"})
        with contextlib.suppress(TypeError):
            return ChatBrowserUse(**llm_kwargs)
        return ChatBrowserUse()

    if ("gemini" in model_lower or "gemma" in model_lower) and ChatGoogle is not None:
        llm_kwargs = _filter_kwargs(ChatGoogle, {"model": model})
        return ChatGoogle(**llm_kwargs)

    if ("claude" in model_lower or "anthropic/" in model_lower) and ChatAnthropic is not None:
        llm_kwargs = _filter_kwargs(ChatAnthropic, {"model": model})
        return ChatAnthropic(**llm_kwargs)

    if (
        ("gpt" in model_lower or model_lower.startswith(("o1", "o3", "o4", "codex", "chatgpt")))
        and ChatOpenAI is not None
    ):
        llm_kwargs = _filter_kwargs(ChatOpenAI, {"model": model})
        return ChatOpenAI(**llm_kwargs)

    # Final fallback to browser-use hosted model class with explicit failure surface.
    llm_kwargs = _filter_kwargs(ChatBrowserUse, {"model": model})
    return ChatBrowserUse(**llm_kwargs)


async def _run_browser_use_agent(
    task: str,
    worker_id: int,
    port: int,
    headless: bool,
    model: str,
    cancel_event: threading.Event,
) -> tuple[str, int, float]:
    """Execute a browser-use agent and return text output, actions, and cost."""
    try:
        from browser_use import Agent, Browser, ChatBrowserUse
    except Exception as exc:
        raise RuntimeError(
            "browser-use is not installed. Install dependencies and retry (pip install -e . or pip install browser-use)."
        ) from exc

    with contextlib.suppress(Exception):
        from browser_use import BrowserConfig  # type: ignore
    with contextlib.suppress(Exception):
        from browser_use import ChatGoogle  # type: ignore
    with contextlib.suppress(Exception):
        from browser_use import ChatAnthropic  # type: ignore
    with contextlib.suppress(Exception):
        from browser_use import ChatOpenAI  # type: ignore
    BrowserConfig = locals().get("BrowserConfig")
    ChatGoogle = locals().get("ChatGoogle")
    ChatAnthropic = locals().get("ChatAnthropic")
    ChatOpenAI = locals().get("ChatOpenAI")

    browser = _build_browser(Browser, BrowserConfig, worker_id=worker_id, port=port, headless=headless)
    try:
        llm = _build_llm(
            model=model,
            ChatBrowserUse=ChatBrowserUse,
            ChatGoogle=ChatGoogle,
            ChatAnthropic=ChatAnthropic,
            ChatOpenAI=ChatOpenAI,
        )
        agent = Agent(**_filter_kwargs(Agent, {"task": task, "llm": llm, "browser": browser}))

        run_task = asyncio.create_task(agent.run(**_filter_kwargs(agent.run, {})))
        while not run_task.done():
            if cancel_event.is_set():
                run_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await run_task
                raise _JobCancelled
            await asyncio.sleep(0.25)

        result_obj = await run_task
        output, action_count = _extract_agent_output(result_obj)
        cost_usd = _extract_cost_usd(result_obj)
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
