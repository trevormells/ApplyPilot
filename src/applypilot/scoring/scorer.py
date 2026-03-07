"""Job fit scoring: LLM-powered evaluation of candidate-job match quality.

Scores jobs on a 1-10 scale by comparing the user's resume against each
job description. All personal data is loaded at runtime from the user's
profile and resume file.
"""

import logging
import re
import time
from datetime import datetime, timezone

from applypilot.config import RESUME_PATH
from applypilot.database import get_connection, get_jobs_by_stage
from applypilot.llm import get_client

log = logging.getLogger(__name__)


# ── Scoring Prompt ────────────────────────────────────────────────────────

SCORE_PROMPT = """You are a job fit evaluator. Given a candidate's resume and a job description, score how well the candidate fits the role.

SCORING CRITERIA:
- 9-10: Perfect match. Candidate has direct experience in nearly all required skills and qualifications.
- 7-8: Strong match. Candidate has most required skills, minor gaps easily bridged.
- 5-6: Moderate match. Candidate has some relevant skills but missing key requirements.
- 3-4: Weak match. Significant skill gaps, would need substantial ramp-up.
- 1-2: Poor match. Completely different field or experience level.

IMPORTANT FACTORS:
- Weight technical skills heavily (programming languages, frameworks, tools)
- Consider transferable experience (automation, scripting, API work)
- Factor in the candidate's project experience
- Be realistic about experience level vs. job requirements (years of experience, seniority)

RESPOND IN EXACTLY THIS FORMAT (no other text):
SCORE: [1-10]
KEYWORDS: [comma-separated ATS keywords from the job description that match or could match the candidate]
REASONING: [2-3 sentences explaining the score]"""


def _job_value(job: dict | None, key: str, default: str = "") -> str:
    """Safely read a string-ish field from a job dict."""
    if not isinstance(job, dict):
        return default
    value = job.get(key, default)
    if value is None:
        return default
    return str(value)


def _parse_score_response(response: str) -> dict:
    """Parse the LLM's score response into structured data.

    Args:
        response: Raw LLM response text.

    Returns:
        {"score": int, "keywords": str, "reasoning": str}
    """
    if not isinstance(response, str):
        return {
            "score": 0,
            "keywords": "",
            "reasoning": f"Unexpected LLM response type: {type(response).__name__}",
        }

    score = 0
    keywords = ""
    reasoning = response

    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("SCORE:"):
            try:
                score = int(re.search(r"\d+", line).group())
                score = max(1, min(10, score))
            except (AttributeError, ValueError):
                score = 0
        elif line.startswith("KEYWORDS:"):
            keywords = line.replace("KEYWORDS:", "").strip()
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()

    return {"score": score, "keywords": keywords, "reasoning": reasoning}


def _log_reasoning_snippet(reasoning: str, limit: int = 160) -> str:
    """Collapse LLM reasoning to one line for per-job progress logs."""
    compact = " ".join(reasoning.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3].rstrip()}..."


def _normalize_score_result(result: dict | None) -> dict:
    """Normalize a scorer result or raise if the shape is invalid."""
    if not isinstance(result, dict):
        raise TypeError(f"score_job returned {type(result).__name__}, expected dict")

    try:
        score = int(result.get("score", 0) or 0)
    except (TypeError, ValueError):
        score = 0

    return {
        "score": max(0, min(10, score)),
        "keywords": str(result.get("keywords", "") or ""),
        "reasoning": str(result.get("reasoning", "") or ""),
    }


def score_job(resume_text: str, job: dict) -> dict:
    """Score a single job against the resume.

    Args:
        resume_text: The candidate's full resume text.
        job: Job dict with keys: title, site, location, full_description.

    Returns:
        {"score": int, "keywords": str, "reasoning": str}
    """
    job_text = (
        f"TITLE: {_job_value(job, 'title', '?')}\n"
        f"COMPANY: {_job_value(job, 'site', '?')}\n"
        f"LOCATION: {_job_value(job, 'location', 'N/A')}\n\n"
        f"DESCRIPTION:\n{_job_value(job, 'full_description', '')[:6000]}"
    )

    messages = [
        {"role": "system", "content": SCORE_PROMPT},
        {"role": "user", "content": f"RESUME:\n{resume_text}\n\n---\n\nJOB POSTING:\n{job_text}"},
    ]

    try:
        client = get_client()
        response = client.chat(messages, max_output_tokens=512)
        return _parse_score_response(response)
    except Exception as e:
        log.error("LLM error scoring job '%s': %s", job.get("title", "?"), e)
        return {"score": 0, "keywords": "", "reasoning": f"LLM error: {e}"}


def run_scoring(limit: int = 0, rescore: bool = False) -> dict:
    """Score unscored jobs that have full descriptions.

    Args:
        limit: Maximum number of jobs to score in this run.
        rescore: If True, re-score all jobs (not just unscored ones).

    Returns:
        {"scored": int, "errors": int, "elapsed": float, "distribution": list}
    """
    resume_text = RESUME_PATH.read_text(encoding="utf-8")
    conn = get_connection()

    if rescore:
        query = "SELECT * FROM jobs WHERE full_description IS NOT NULL"
        if limit > 0:
            query += f" LIMIT {limit}"
        jobs = conn.execute(query).fetchall()
    else:
        jobs = get_jobs_by_stage(conn=conn, stage="pending_score", limit=limit)

    if not jobs:
        log.info("No unscored jobs with descriptions found.")
        return {"scored": 0, "errors": 0, "elapsed": 0.0, "distribution": []}

    # Convert sqlite3.Row to dicts if needed
    if jobs and not isinstance(jobs[0], dict):
        columns = jobs[0].keys()
        jobs = [dict(zip(columns, row)) for row in jobs]

    log.info("Scoring %d jobs sequentially...", len(jobs))
    t0 = time.time()
    completed = 0
    errors = 0
    persisted = 0

    for job in jobs:
        completed += 1
        title = _job_value(job, "title", "?")[:60]
        url = _job_value(job, "url", "")

        try:
            result = _normalize_score_result(score_job(resume_text, job))
        except Exception as e:
            errors += 1
            fallback = {
                "score": 0,
                "keywords": "",
                "reasoning": f"Scoring error: {e}",
                "url": url,
            }
            result = fallback
            log.exception("[%d/%d] scoring failed  %s", completed, len(jobs), title)

        result["url"] = url

        if result["score"] == 0:
            errors += 1

        if not url:
            errors += 1
            log.error("[%d/%d] score=%d  %s | missing job url; skipping DB update", completed, len(jobs), result["score"], title)
            continue

        try:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "UPDATE jobs SET fit_score = ?, score_reasoning = ?, scored_at = ? WHERE url = ?",
                (result["score"], f"{result['keywords']}\n{result['reasoning']}", now, result["url"]),
            )
            conn.commit()
            persisted += 1
        except Exception:
            errors += 1
            log.exception("Failed to persist score for %s", result.get("url", "?"))
            continue

        log.info(
            "[%d/%d] score=%d  %s | %s",
            completed,
            len(jobs),
            result["score"],
            title,
            _log_reasoning_snippet(result.get("reasoning", "")),
        )

    elapsed = time.time() - t0
    log.info(
        "Done: %d scored in %.1fs (%.1f jobs/sec)", persisted, elapsed, persisted / elapsed if elapsed > 0 else 0
    )

    # Score distribution
    dist = conn.execute("""
        SELECT fit_score, COUNT(*) FROM jobs
        WHERE fit_score IS NOT NULL
        GROUP BY fit_score ORDER BY fit_score DESC
    """).fetchall()
    distribution = [(row[0], row[1]) for row in dist]

    return {
        "scored": persisted,
        "errors": errors,
        "elapsed": elapsed,
        "distribution": distribution,
    }
