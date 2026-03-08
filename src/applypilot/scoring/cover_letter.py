"""Cover letter generation: LLM-powered, profile-driven, with validation.

Generates concise, engineering-voice cover letters tailored to specific job
postings. All personal data (name, skills, achievements) comes from the user's
profile at runtime. No hardcoded personal information.
"""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

from applypilot.config import COVER_LETTER_DIR, RESUME_PATH, load_profile
from applypilot.database import get_connection
from applypilot.llm import get_client
from applypilot.llm_cost import bind_current_llm_cost_context
from applypilot.scoring.validator import (
    BANNED_WORDS,
    LLM_LEAK_PHRASES,
    sanitize_text,
    validate_cover_letter,
)

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 5  # max cross-run retries before giving up


# ── Prompt Builder (profile-driven) ──────────────────────────────────────


def _build_cover_letter_prompt(profile: dict) -> str:
    """Build the cover letter system prompt from the user's profile.

    All personal data, skills, and sign-off name come from the profile.
    """
    personal = profile.get("personal", {})
    boundary = profile.get("skills_boundary", {})
    resume_facts = profile.get("resume_facts", {})

    # Preferred name for the sign-off (falls back to full name)
    sign_off_name = personal.get("preferred_name") or personal.get("full_name", "")

    # Flatten all allowed skills
    all_skills: list[str] = []
    for items in boundary.values():
        if isinstance(items, list):
            all_skills.extend(items)
    skills_str = ", ".join(all_skills) if all_skills else "the tools listed in the resume"

    # Real metrics from resume_facts
    real_metrics = resume_facts.get("real_metrics", [])
    preserved_projects = resume_facts.get("preserved_projects", [])

    # Build achievement examples for the prompt
    projects_hint = ""
    if preserved_projects:
        projects_hint = f"\nKnown projects to reference: {', '.join(preserved_projects)}"

    metrics_hint = ""
    if real_metrics:
        metrics_hint = f"\nReal metrics to use: {', '.join(real_metrics)}"

    # Build the full banned list from the validator so the prompt stays in sync
    # with what will actually be rejected — the validator checks all of these.
    all_banned = ", ".join(f'"{w}"' for w in BANNED_WORDS)
    leak_banned = ", ".join(f'"{p}"' for p in LLM_LEAK_PHRASES)

    return f"""Write a cover letter for {sign_off_name}. The goal is to get an interview.

STRUCTURE: 3 short paragraphs. Under 250 words. Every sentence must earn its place.

PARAGRAPH 1 (2-3 sentences): Open with a specific thing YOU built that solves THEIR problem. Not "I'm excited about this role." Not "This role aligns with my experience." Start with the work.

PARAGRAPH 2 (3-4 sentences): Pick 2 achievements from the resume that are MOST relevant to THIS job. Use numbers. Frame as solving their problem, not listing your accomplishments.{projects_hint}{metrics_hint}

PARAGRAPH 3 (1-2 sentences): One specific thing about the company from the job description (a product, a technical challenge, a team structure). Then close. "Happy to walk through any of this in more detail." or "Let's discuss." Nothing else.

BANNED WORDS AND PHRASES (automated validator rejects ANY of these — do not use even once):
{all_banned}

ALSO BANNED (meta-commentary the validator catches):
{leak_banned}

BANNED PUNCTUATION: No em dashes (—) or en dashes (–). Use commas or periods.

VOICE:
- Write like a real engineer emailing someone they respect. Not formal, not casual. Just direct.
- NEVER narrate or explain what you're doing. BAD: "This demonstrates my commitment to X." GOOD: Just state the fact and move on.
- NEVER hedge. BAD: "might address some of your challenges." GOOD: "solves the same problem your team is facing."
- Every sentence should contain either a number, a tool name, or a specific outcome. If it doesn't, cut it.
- Read it out loud. If it sounds like a robot wrote it, rewrite it.

FABRICATION = INSTANT REJECTION:
The candidate's real tools are ONLY: {skills_str}.
Do NOT mention ANY tool not in this list. If the job asks for tools not listed, talk about the work you did, not the tools.

Sign off: just "{sign_off_name}"

Output ONLY the letter text. No subject lines. No "Here is the cover letter:" preamble. No notes after the sign-off.
Start DIRECTLY with "Dear Hiring Manager," and end with the name."""


# ── Helpers ──────────────────────────────────────────────────────────────


def _strip_preamble(text: str) -> str:
    """Remove LLM preamble before 'Dear Hiring Manager,' if present.

    Gemini and other models sometimes output "Here is the cover letter:" or
    similar meta-commentary before the actual letter text. Strip everything
    before the first occurrence of "Dear" so the validator's start-check passes.
    """
    dear_idx = text.lower().find("dear")
    if dear_idx > 0:
        return text[dear_idx:]
    return text


# ── Core Generation ──────────────────────────────────────────────────────


def generate_cover_letter(
    resume_text: str,
    job: dict,
    profile: dict,
    max_retries: int = 3,
    validation_mode: str = "normal",
) -> str:
    """Generate a cover letter with fresh context on each retry + auto-sanitize.

    Same design as tailor_resume: fresh conversation per attempt, issues noted
    in the prompt, no conversation history stacking.

    Args:
        resume_text:      The candidate's resume text (base or tailored).
        job:              Job dict with title, site, location, full_description.
        profile:          User profile dict.
        max_retries:      Maximum retry attempts.
        validation_mode:  "strict", "normal", or "lenient".

    Returns:
        The cover letter text (best attempt even if validation failed).
    """
    job_text = (
        f"TITLE: {job['title']}\n"
        f"COMPANY: {job['site']}\n"
        f"LOCATION: {job.get('location', 'N/A')}\n\n"
        f"DESCRIPTION:\n{(job.get('full_description') or '')[:6000]}"
    )

    avoid_notes: list[str] = []
    letter = ""
    client = get_client(model_tier="high")
    cl_prompt_base = _build_cover_letter_prompt(profile)
    timings: dict[str, object] = {
        "generation_seconds": 0.0,
        "attempts": [],
    }

    for attempt in range(max_retries + 1):
        attempt_timing = {
            "attempt": attempt + 1,
            "generation_seconds": 0.0,
            "outcome": "pending",
        }
        # Fresh conversation every attempt
        prompt = cl_prompt_base
        if avoid_notes:
            prompt += "\n\n## AVOID THESE ISSUES:\n" + "\n".join(f"- {n}" for n in avoid_notes[-5:])

        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (f"RESUME:\n{resume_text}\n\n---\n\nTARGET JOB:\n{job_text}\n\nWrite the cover letter:"),
            },
        ]

        generation_started = time.perf_counter()
        letter = client.chat(messages, max_output_tokens=10000)
        generation_elapsed = time.perf_counter() - generation_started
        attempt_timing["generation_seconds"] = generation_elapsed
        timings["generation_seconds"] = float(timings["generation_seconds"]) + generation_elapsed
        letter = sanitize_text(letter)  # auto-fix em dashes, smart quotes
        letter = _strip_preamble(letter)  # remove any "Here is the letter:" prefix

        validation = validate_cover_letter(letter, mode=validation_mode)
        if validation["passed"]:
            attempt_timing["outcome"] = "approved"
            attempt_timing["validation_errors"] = []
            cast_attempts = timings["attempts"]
            assert isinstance(cast_attempts, list)
            cast_attempts.append(attempt_timing)
            return letter, {
                "attempts": attempt + 1,
                "status": "approved",
                "validation_mode": validation_mode,
                "timings": timings,
            }

        avoid_notes.extend(validation["errors"])
        attempt_timing["outcome"] = "failed_validation"
        attempt_timing["validation_errors"] = list(validation["errors"])
        cast_attempts = timings["attempts"]
        assert isinstance(cast_attempts, list)
        cast_attempts.append(attempt_timing)
        # Warnings never block — only hard errors trigger a retry
        log.debug(
            "Cover letter attempt %d/%d failed: %s",
            attempt + 1,
            max_retries + 1,
            validation["errors"],
        )

    return letter, {
        "attempts": max_retries + 1,
        "status": "failed_validation",
        "validation_mode": validation_mode,
        "timings": timings,
    }


def _cover_filename_prefix(job: dict) -> str:
    """Build a stable artifact filename prefix for a cover-letter job."""
    safe_title = re.sub(r"[^\w\s-]", "", job["title"])[:50].strip().replace(" ", "_")
    safe_site = re.sub(r"[^\w\s-]", "", job["site"])[:20].strip().replace(" ", "_")
    return f"{safe_site}_{safe_title}"


def _process_cover_job(resume_text: str, job: dict, profile: dict, validation_mode: str) -> dict:
    """Generate one cover letter, persist artifacts, and update DB state."""
    started = time.perf_counter()

    try:
        letter, report = generate_cover_letter(resume_text, job, profile, validation_mode=validation_mode)
        prefix = _cover_filename_prefix(job)

        cl_path = COVER_LETTER_DIR / f"{prefix}_CL.txt"
        cl_path.write_text(letter, encoding="utf-8")

        pdf_path = None
        try:
            from applypilot.scoring.pdf import convert_to_pdf

            pdf_path = str(convert_to_pdf(cl_path))
        except Exception:
            log.debug("PDF generation failed for %s", cl_path, exc_info=True)

        result = {
            "url": job["url"],
            "path": str(cl_path),
            "pdf_path": pdf_path,
            "title": job["title"],
            "site": job["site"],
            "status": report["status"],
            "attempts": report["attempts"],
            "timings": report.get("timings", {}),
        }
    except Exception as exc:
        log.error("Cover letter generation failed for %s -- %s", job["title"][:40], exc)
        result = {
            "url": job["url"],
            "title": job["title"],
            "site": job["site"],
            "path": None,
            "pdf_path": None,
            "status": "error",
            "attempts": 0,
            "timings": {"generation_seconds": 0.0, "attempts": []},
            "error": str(exc),
        }

    try:
        conn = get_connection()
        now = datetime.now(timezone.utc).isoformat()
        if result["status"] == "approved":
            conn.execute(
                "UPDATE jobs SET cover_letter_path=?, cover_letter_at=?, "
                "cover_attempts=COALESCE(cover_attempts,0)+1 WHERE url=?",
                (result["path"], now, result["url"]),
            )
        else:
            conn.execute(
                "UPDATE jobs SET cover_attempts=COALESCE(cover_attempts,0)+1 WHERE url=?",
                (result["url"],),
            )
        conn.commit()
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        log.error("Failed to persist cover letter for %s -- %s", job["title"][:40], exc)

    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def _log_cover_completion(completed: int, total: int, result: dict, started_at: float) -> None:
    """Emit a per-job completion line with generation timing."""
    elapsed = time.time() - started_at
    rate = completed / elapsed if elapsed > 0 else 0
    timings = result.get("timings", {})
    generation_seconds = float(timings.get("generation_seconds", 0.0) or 0.0)
    total_seconds = float(result.get("elapsed_seconds", generation_seconds) or 0.0)
    status_label = "OK" if result["status"] == "approved" else result["status"].upper()
    log.info(
        "%d/%d [%s] attempts=%s | gen=%.1fs total=%.1fs | %.1f jobs/min | %s",
        completed,
        total,
        status_label,
        result.get("attempts", "?"),
        generation_seconds,
        total_seconds,
        rate * 60,
        result["title"][:40],
    )


# ── Batch Entry Point ────────────────────────────────────────────────────


def run_cover_letters(
    min_score: int = 7,
    limit: Optional[int] = None,
    validation_mode: str = "normal",
    workers: int = 1,
) -> dict:
    """Generate cover letters for high-scoring jobs that have tailored resumes.

    Args:
        min_score:       Minimum fit_score threshold.
        limit:           Maximum jobs to process. `None` or `<= 0` means unlimited.
        validation_mode: "strict", "normal", or "lenient".
        workers:         Number of jobs to generate concurrently.

    Returns:
        {"generated": int, "errors": int, "elapsed": float}
    """
    profile = load_profile()
    resume_text = RESUME_PATH.read_text(encoding="utf-8")
    conn = get_connection()

    # Fetch jobs that have tailored resumes but no cover letter yet
    query = (
        "SELECT * FROM jobs "
        "WHERE fit_score >= ? AND tailored_resume_path IS NOT NULL "
        "AND full_description IS NOT NULL "
        "AND (cover_letter_path IS NULL OR cover_letter_path = '') "
        "AND COALESCE(cover_attempts, 0) < ? "
        "ORDER BY fit_score DESC"
    )
    params: list[object] = [min_score, MAX_ATTEMPTS]
    if limit is not None and limit > 0:
        query += " LIMIT ?"
        params.append(limit)
    jobs = conn.execute(query, params).fetchall()

    if not jobs:
        log.info("No jobs needing cover letters (score >= %d).", min_score)
        return {"generated": 0, "errors": 0, "elapsed": 0.0}

    # Convert rows to dicts
    if jobs and not isinstance(jobs[0], dict):
        columns = jobs[0].keys()
        jobs = [dict(zip(columns, row)) for row in jobs]

    workers = max(1, workers)
    COVER_LETTER_DIR.mkdir(parents=True, exist_ok=True)
    log.info(
        "Generating cover letters for %d jobs (score >= %d, workers=%d)...",
        len(jobs),
        min_score,
        min(workers, len(jobs)),
    )
    t0 = time.time()
    completed = 0
    error_count = 0
    saved = 0

    if workers == 1 or len(jobs) == 1:
        results = (
            _process_cover_job(resume_text, job, profile, validation_mode=validation_mode)
            for job in jobs
        )
        for result in results:
            completed += 1
            if result["status"] == "approved":
                saved += 1
            else:
                error_count += 1
            _log_cover_completion(completed, len(jobs), result, t0)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs)), thread_name_prefix="cover-worker") as pool:
            future_to_job = {
                pool.submit(
                    bind_current_llm_cost_context(_process_cover_job),
                    resume_text,
                    job,
                    profile,
                    validation_mode,
                ): job
                for job in jobs
            }
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                except Exception as exc:
                    log.exception("Cover letter generation crashed for %s", job["title"][:40])
                    result = {
                        "url": job["url"],
                        "title": job["title"],
                        "site": job["site"],
                        "path": None,
                        "pdf_path": None,
                        "status": "error",
                        "attempts": 0,
                        "timings": {"generation_seconds": 0.0, "attempts": []},
                        "elapsed_seconds": 0.0,
                        "error": str(exc),
                    }
                completed += 1
                if result["status"] == "approved":
                    saved += 1
                else:
                    error_count += 1
                _log_cover_completion(completed, len(jobs), result, t0)

    elapsed = time.time() - t0
    log.info("Cover letters done in %.1fs: %d generated, %d errors", elapsed, saved, error_count)

    return {
        "generated": saved,
        "errors": error_count,
        "elapsed": elapsed,
    }
