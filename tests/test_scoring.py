import logging

from applypilot.database import init_db
from applypilot.scoring import scorer


def test_run_scoring_continues_when_score_job_returns_none(monkeypatch, tmp_path, caplog) -> None:
    db_path = tmp_path / "jobs.db"
    resume_path = tmp_path / "resume.txt"
    resume_path.write_text("Example resume", encoding="utf-8")
    conn = init_db(db_path)

    jobs = [
        {
            "url": "https://jobs.test/1",
            "title": "Broken job",
            "site": "ExampleSite",
            "full_description": "desc",
        },
        {
            "url": "https://jobs.test/2",
            "title": "Healthy job",
            "site": "ExampleSite",
            "full_description": "desc",
        },
    ]

    for job in jobs:
        columns = ", ".join(job.keys())
        placeholders = ", ".join("?" for _ in job)
        conn.execute(f"INSERT INTO jobs ({columns}) VALUES ({placeholders})", tuple(job.values()))
    conn.commit()

    results = iter(
        [
            None,
            {"score": 8, "keywords": "python, sql", "reasoning": "Strong overlap with required skills."},
        ]
    )

    monkeypatch.setattr(scorer, "RESUME_PATH", resume_path)
    monkeypatch.setattr(scorer, "get_connection", lambda: conn)
    monkeypatch.setattr(scorer, "get_jobs_by_stage", lambda conn, stage, limit: jobs)
    monkeypatch.setattr(scorer, "score_job", lambda resume_text, job: next(results))

    with caplog.at_level(logging.ERROR):
        stats = scorer.run_scoring()

    rows = conn.execute(
        "SELECT url, fit_score, score_reasoning FROM jobs ORDER BY url"
    ).fetchall()

    assert stats["scored"] == 2
    assert stats["errors"] == 1
    assert rows[0][1] == 0
    assert "Scoring error: score_job returned NoneType, expected dict" in rows[0][2]
    assert rows[1][1] == 8
    assert "Strong overlap with required skills." in rows[1][2]
    assert "scoring failed" in caplog.text
