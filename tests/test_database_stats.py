from applypilot.database import close_connection, get_stats, init_db


def test_get_stats_includes_stage_workload_and_apply_breakdown(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "jobs.db"
    monkeypatch.setattr("applypilot.database.load_blocked_sites", lambda: ({"BlockedSite"}, []))

    conn = init_db(db_path)

    jobs = [
        {"url": "https://jobs.test/1", "site": "ActiveSite", "title": "Pending enrich"},
        {"url": "https://jobs.test/2", "site": "BlockedSite", "title": "Blocked enrich"},
        {
            "url": "https://jobs.test/3",
            "site": "ActiveSite",
            "title": "Enrich error",
            "detail_scraped_at": "2026-03-07T00:00:00+00:00",
            "detail_error": "timeout",
        },
        {
            "url": "https://jobs.test/4",
            "site": "ActiveSite",
            "title": "Pending score",
            "full_description": "desc",
        },
        {
            "url": "https://jobs.test/5",
            "site": "ActiveSite",
            "title": "Below threshold",
            "full_description": "desc",
            "fit_score": 5,
        },
        {
            "url": "https://jobs.test/6",
            "site": "ActiveSite",
            "title": "Pending tailor",
            "full_description": "desc",
            "fit_score": 8,
            "application_url": "https://apply.test/6",
        },
        {
            "url": "https://jobs.test/7",
            "site": "ActiveSite",
            "title": "Tailor exhausted",
            "full_description": "desc",
            "fit_score": 8,
            "tailor_attempts": 5,
            "application_url": "https://apply.test/7",
        },
        {
            "url": "https://jobs.test/8",
            "site": "ActiveSite",
            "title": "Pending cover",
            "full_description": "desc",
            "fit_score": 8,
            "tailored_resume_path": "/tmp/resume-8.txt",
            "application_url": "https://apply.test/8",
        },
        {
            "url": "https://jobs.test/9",
            "site": "ActiveSite",
            "title": "Cover exhausted",
            "full_description": "desc",
            "fit_score": 8,
            "tailored_resume_path": "/tmp/resume-9.txt",
            "cover_attempts": 5,
            "application_url": "https://apply.test/9",
        },
        {
            "url": "https://jobs.test/10",
            "site": "ActiveSite",
            "title": "Ready to apply",
            "full_description": "desc",
            "fit_score": 8,
            "tailored_resume_path": "/tmp/resume-10.txt",
            "application_url": "https://apply.test/10",
        },
        {
            "url": "https://jobs.test/11",
            "site": "ActiveSite",
            "title": "Apply in progress",
            "full_description": "desc",
            "fit_score": 8,
            "tailored_resume_path": "/tmp/resume-11.txt",
            "application_url": "https://apply.test/11",
            "apply_status": "in_progress",
        },
        {
            "url": "https://jobs.test/12",
            "site": "ActiveSite",
            "title": "Apply failed",
            "full_description": "desc",
            "fit_score": 8,
            "tailored_resume_path": "/tmp/resume-12.txt",
            "application_url": "https://apply.test/12",
            "apply_status": "failed",
            "apply_error": "captcha",
        },
        {
            "url": "https://jobs.test/13",
            "site": "ActiveSite",
            "title": "Manual ATS",
            "full_description": "desc",
            "fit_score": 8,
            "tailored_resume_path": "/tmp/resume-13.txt",
            "application_url": "https://apply.test/13",
            "apply_status": "manual",
            "apply_error": "manual ATS",
        },
        {
            "url": "https://jobs.test/14",
            "site": "ActiveSite",
            "title": "Applied",
            "full_description": "desc",
            "fit_score": 9,
            "tailored_resume_path": "/tmp/resume-14.txt",
            "cover_letter_path": "/tmp/cover-14.txt",
            "application_url": "https://apply.test/14",
            "apply_status": "applied",
            "applied_at": "2026-03-07T00:00:00+00:00",
        },
    ]

    for job in jobs:
        columns = ", ".join(job.keys())
        placeholders = ", ".join("?" for _ in job)
        conn.execute(f"INSERT INTO jobs ({columns}) VALUES ({placeholders})", tuple(job.values()))
    conn.commit()

    stats = get_stats(conn=conn, min_score=7)

    assert stats["total"] == 14
    assert stats["source_count"] == 2
    assert stats["pending_detail"] == 1
    assert stats["pending_detail_blocked"] == 1
    assert stats["detail_errors"] == 1
    assert stats["unscored"] == 1
    assert stats["untailored_eligible"] == 1
    assert stats["tailor_exhausted"] == 1
    assert stats["pending_cover"] == 5
    assert stats["with_cover_letter"] == 1
    assert stats["ready_to_apply"] == 6
    assert stats["applied"] == 1
    assert stats["apply_in_progress"] == 1
    assert stats["apply_failed"] == 1
    assert stats["apply_manual"] == 1
    assert stats["apply_status_breakdown"] == [
        ("not_started", 10),
        ("applied", 1),
        ("failed", 1),
        ("in_progress", 1),
        ("manual", 1),
    ]

    close_connection(db_path)
