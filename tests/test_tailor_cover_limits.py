from pathlib import Path

from applypilot.database import init_db
from applypilot.scoring import cover_letter, tailor


def _insert_jobs(conn, count: int, *, tailored: bool = False) -> None:
    for idx in range(count):
        job = {
            "url": f"https://jobs.test/{idx}",
            "title": f"Role {idx}",
            "site": "ExampleSite",
            "location": "Remote",
            "full_description": "Build backend systems.",
            "fit_score": 8,
        }
        if tailored:
            job["tailored_resume_path"] = f"/tmp/resume-{idx}.txt"

        columns = ", ".join(job.keys())
        placeholders = ", ".join("?" for _ in job)
        conn.execute(f"INSERT INTO jobs ({columns}) VALUES ({placeholders})", tuple(job.values()))
    conn.commit()


def test_run_tailoring_default_limit_is_unlimited(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "jobs.db"
    resume_path = tmp_path / "resume.txt"
    tailored_dir = tmp_path / "tailored"
    resume_path.write_text("Example resume", encoding="utf-8")
    conn = init_db(db_path)
    _insert_jobs(conn, 21)

    monkeypatch.setattr(tailor, "RESUME_PATH", resume_path)
    monkeypatch.setattr(tailor, "TAILORED_DIR", tailored_dir)
    monkeypatch.setattr(tailor, "get_connection", lambda: conn)
    monkeypatch.setattr(tailor, "load_profile", lambda: {"skills_boundary": {}, "resume_facts": {}, "experience": {}})
    monkeypatch.setattr(
        tailor,
        "tailor_resume",
        lambda resume_text, job, profile, validation_mode="normal": (
            f"Tailored resume for {job['title']}",
            {"status": "approved", "attempts": 1, "validator": {"passed": True}, "judge": {"passed": True}},
        ),
    )

    stats = tailor.run_tailoring()

    tailored_count = conn.execute("SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL").fetchone()[0]

    assert stats["approved"] == 21
    assert tailored_count == 21


def test_run_cover_letters_default_limit_is_unlimited(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "jobs.db"
    resume_path = tmp_path / "resume.txt"
    cover_dir = tmp_path / "cover_letters"
    resume_path.write_text("Example resume", encoding="utf-8")
    conn = init_db(db_path)
    _insert_jobs(conn, 21, tailored=True)

    monkeypatch.setattr(cover_letter, "RESUME_PATH", resume_path)
    monkeypatch.setattr(cover_letter, "COVER_LETTER_DIR", cover_dir)
    monkeypatch.setattr(cover_letter, "get_connection", lambda: conn)
    monkeypatch.setattr(cover_letter, "load_profile", lambda: {"personal": {}, "skills_boundary": {}, "resume_facts": {}})
    monkeypatch.setattr(
        cover_letter,
        "generate_cover_letter",
        lambda resume_text, job, profile, validation_mode="normal": f"Dear Hiring Manager,\n\nLetter for {job['title']}\n\nName",
    )

    stats = cover_letter.run_cover_letters()

    cover_count = conn.execute("SELECT COUNT(*) FROM jobs WHERE cover_letter_path IS NOT NULL").fetchone()[0]

    assert stats["generated"] == 21
    assert cover_count == 21
