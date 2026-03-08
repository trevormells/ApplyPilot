import sqlite3
import threading
import time
from contextlib import contextmanager

from applypilot.database import init_db
from applypilot.discovery import smartextract
from applypilot.enrichment import detail
from applypilot.scoring import scorer


class InflightTracker:
    def __init__(self) -> None:
        self.current = 0
        self.max_seen = 0
        self._lock = threading.Lock()

    @contextmanager
    def track(self):
        with self._lock:
            self.current += 1
            self.max_seen = max(self.max_seen, self.current)
        try:
            yield
        finally:
            with self._lock:
                self.current -= 1


class FakeClient:
    def __init__(self, tracker: InflightTracker, response_text: str, delay: float = 0.05) -> None:
        self._tracker = tracker
        self._response_text = response_text
        self._delay = delay
        self.calls = 0

    def chat(self, messages: list[dict], **kwargs: object) -> str:
        _ = messages, kwargs
        with self._tracker.track():
            self.calls += 1
            time.sleep(self._delay)
            return self._response_text


class _DummyResponse:
    status = 200


class _DummyPage:
    def __init__(self) -> None:
        self.url = ""

    def goto(self, url: str, timeout: int = 0) -> _DummyResponse:
        _ = timeout
        self.url = url
        return _DummyResponse()

    def wait_for_load_state(self, state: str, timeout: int = 0) -> None:
        _ = state, timeout

    def title(self) -> str:
        return "Example job"

    def query_selector_all(self, selector: str) -> list[object]:
        _ = selector
        return []

    def query_selector(self, selector: str) -> None:
        _ = selector
        return None


class _DummyContext:
    def new_page(self) -> _DummyPage:
        return _DummyPage()


class _DummyBrowser:
    def new_context(self, user_agent: str) -> _DummyContext:
        _ = user_agent
        return _DummyContext()

    def close(self) -> None:
        return None


class _DummyPlaywright:
    def __enter__(self) -> "_DummyPlaywright":
        self.chromium = self
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _ = exc_type, exc, tb
        return False

    def launch(self, **kwargs: object) -> _DummyBrowser:
        _ = kwargs
        return _DummyBrowser()


def test_run_scoring_overlaps_llm_requests_with_multiple_workers(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "score.db"
    resume_path = tmp_path / "resume.txt"
    resume_path.write_text("Example resume", encoding="utf-8")
    conn = init_db(db_path)

    jobs = [
        {
            "url": f"https://jobs.test/{index}",
            "title": f"Job {index}",
            "site": "ExampleSite",
            "full_description": "Detailed description",
        }
        for index in range(3)
    ]

    for job in jobs:
        columns = ", ".join(job.keys())
        placeholders = ", ".join("?" for _ in job)
        conn.execute(f"INSERT INTO jobs ({columns}) VALUES ({placeholders})", tuple(job.values()))
    conn.commit()

    tracker = InflightTracker()
    client = FakeClient(
        tracker,
        "SCORE: 8\nKEYWORDS: python, sql\nREASONING: Strong overlap with required skills.",
    )

    def connection_factory():
        thread_conn = sqlite3.connect(db_path, timeout=30)
        thread_conn.row_factory = sqlite3.Row
        return thread_conn

    monkeypatch.setattr(scorer, "RESUME_PATH", resume_path)
    monkeypatch.setattr(scorer, "get_connection", connection_factory)
    monkeypatch.setattr(scorer, "get_jobs_by_stage", lambda conn, stage, limit: jobs)
    monkeypatch.setattr(scorer, "get_client", lambda: client)

    stats = scorer.run_scoring(workers=3)

    assert stats["scored"] == 3
    assert client.calls == 3
    assert tracker.max_seen >= 2


def test_smart_extract_overlaps_llm_requests_across_targets(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "discover.db"
    tracker = InflightTracker()
    client = FakeClient(
        tracker,
        (
            '{"strategy":"json_ld","reasoning":"use structured data","extraction":'
            '{"title":"title","salary":null,"description":"description",'
            '"location":"jobLocation[0].address.addressCountry","url":"url"}}'
        ),
    )

    def _fake_intel(url: str, headless: bool = True) -> dict:
        _ = headless
        return {
            "url": url,
            "page_title": "Jobs",
            "json_ld": [
                {
                    "@type": "JobPosting",
                    "title": f"Role from {url}",
                    "description": "Structured description",
                    "jobLocation": [{"address": {"addressCountry": "Remote"}}],
                    "url": f"{url}/apply",
                }
            ],
            "api_responses": [],
            "data_testids": [],
            "dom_stats": {},
            "card_candidates": [],
            "full_html": "<main>" + ("<article>job</article>" * 400) + "</main>",
        }

    monkeypatch.setattr(smartextract, "init_db", lambda: init_db(db_path))
    monkeypatch.setattr(smartextract, "get_stats", lambda conn: {"total": 0, "pending_detail": 0, "pending_detail_blocked": 0})
    monkeypatch.setattr(smartextract, "collect_page_intelligence", _fake_intel)
    monkeypatch.setattr(smartextract, "get_client", lambda: client)

    result = smartextract._run_all(
        [
            {"name": "Site A", "url": "https://site-a.example/jobs"},
            {"name": "Site B", "url": "https://site-b.example/jobs"},
        ],
        accept_locs=[],
        reject_locs=[],
        workers=2,
    )

    assert result["passed"] == 2
    assert client.calls == 2
    assert tracker.max_seen >= 2


def _install_detail_llm_test_doubles(monkeypatch, db_path, tracker: InflightTracker) -> FakeClient:
    client = FakeClient(
        tracker,
        '{"full_description":"LLM extracted description","application_url":"https://apply.example/form"}',
    )

    monkeypatch.setattr(detail, "init_db", lambda: init_db(db_path))
    monkeypatch.setattr(detail, "sync_playwright", lambda: _DummyPlaywright())
    monkeypatch.setattr(detail, "collect_detail_intelligence", lambda page: {"json_ld": [], "page_title": page.title(), "final_url": page.url})
    monkeypatch.setattr(detail, "extract_from_json_ld", lambda intel: None)
    monkeypatch.setattr(detail, "extract_description_deterministic", lambda page: None)
    monkeypatch.setattr(detail, "extract_apply_url_deterministic", lambda page: "https://apply.example/fallback")
    monkeypatch.setattr(detail, "extract_main_content", lambda page: "Full page content for LLM extraction")
    monkeypatch.setattr(detail, "get_client", lambda: client)

    return client


def test_enrichment_overlaps_llm_requests_across_sites(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "enrich-parallel.db"
    conn = init_db(db_path)
    rows = [
        ("https://jobs.test/a", "Job A", "Site A"),
        ("https://jobs.test/b", "Job B", "Site B"),
    ]
    for url, title, site in rows:
        conn.execute("INSERT INTO jobs (url, title, site) VALUES (?, ?, ?)", (url, title, site))
    conn.commit()

    tracker = InflightTracker()
    client = _install_detail_llm_test_doubles(monkeypatch, db_path, tracker)
    monkeypatch.setitem(detail.SITE_DELAYS, "Site A", 0.0)
    monkeypatch.setitem(detail.SITE_DELAYS, "Site B", 0.0)

    stats = detail._run_detail_scraper(conn, workers=2)

    assert stats["processed"] == 2
    assert stats["tiers"][3] == 2
    assert client.calls == 2
    assert tracker.max_seen >= 2


def test_enrichment_serializes_llm_requests_within_single_site_batch(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "enrich-serial.db"
    conn = init_db(db_path)
    rows = [
        ("https://jobs.test/1", "Job 1", "Only Site"),
        ("https://jobs.test/2", "Job 2", "Only Site"),
    ]
    for url, title, site in rows:
        conn.execute("INSERT INTO jobs (url, title, site) VALUES (?, ?, ?)", (url, title, site))
    conn.commit()

    tracker = InflightTracker()
    client = _install_detail_llm_test_doubles(monkeypatch, db_path, tracker)
    monkeypatch.setitem(detail.SITE_DELAYS, "Only Site", 0.0)

    stats = detail._run_detail_scraper(conn, workers=4)

    assert stats["processed"] == 2
    assert stats["tiers"][3] == 2
    assert client.calls == 2
    assert tracker.max_seen == 1
