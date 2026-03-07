"""Database operations for launcher v2."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from applypilot import config
from applypilot.database import get_connection

logger = logging.getLogger(__name__)


def _load_blocked() -> tuple[list[str], list[str]]:
    from applypilot.config import load_blocked_sites

    return load_blocked_sites()


def acquire_job(target_url: str | None = None, min_score: int = 7, worker_id: int = 0) -> dict | None:
    """Atomically acquire the next job to apply to."""
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
                WITH ranked_jobs AS (
                    SELECT url, title, site, application_url, tailored_resume_path,
                           fit_score, location, full_description, cover_letter_path,
                           ROW_NUMBER() OVER (
                               PARTITION BY COALESCE(NULLIF(TRIM(site), ''), url)
                               ORDER BY fit_score DESC, url
                           ) AS company_rank
                    FROM jobs
                    WHERE tailored_resume_path IS NOT NULL
                      AND (apply_status IS NULL OR apply_status = 'failed')
                      AND (apply_attempts IS NULL OR apply_attempts < ?)
                      AND fit_score >= ?
                      {site_clause}
                      {url_clauses}
                )
                SELECT url, title, site, application_url, tailored_resume_path,
                       fit_score, location, full_description, cover_letter_path
                FROM ranked_jobs
                ORDER BY company_rank, fit_score DESC, url
                LIMIT 1
            """,
                [config.DEFAULTS["max_apply_attempts"]] + params,
            ).fetchone()

        if not row:
            conn.rollback()
            return None

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


def mark_job(url: str, status: str, reason: str | None = None) -> None:
    """Manually mark a job's apply status in the database."""
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
    """Reset all failed jobs so they can be retried."""
    conn = get_connection()
    cursor = conn.execute(
        """
        UPDATE jobs SET apply_status = NULL, apply_error = NULL,
                       apply_attempts = 0, agent_id = NULL
        WHERE apply_status = 'failed'
          OR (apply_status IS NOT NULL AND apply_status != 'applied'
              AND apply_status != 'in_progress')
    """
    )
    conn.commit()
    return cursor.rowcount
