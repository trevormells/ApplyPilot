"""Rich terminal dashboard for database pipeline stats."""

from __future__ import annotations

from rich.columns import Columns
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _render_pipeline_table(stats: dict, min_score: int) -> Panel:
    table = Table(expand=True, show_edge=False)
    table.add_column("Stage", style="bold")
    table.add_column("Done", justify="right", style="green")
    table.add_column("Queued", justify="right", style="yellow")
    table.add_column("Skipped", justify="right", style="dim")
    table.add_column("Attention", justify="right", style="red")
    table.add_column("Notes", overflow="fold")

    rows = [
        ("Discover", stats["total"], 0, 0, 0, "All rows currently stored in SQLite."),
        (
            "Enrich",
            stats["with_description"],
            stats["pending_detail"],
            stats["pending_detail_blocked"],
            stats["detail_errors"],
            (
                f"{stats['pending_detail_blocked']} blocked-site job(s), "
                f"{stats['detail_errors']} scrape error(s)."
            ),
        ),
        ("Score", stats["scored"], stats["unscored"], 0, 0, "Jobs with descriptions waiting for fit scoring."),
        (
            "Tailor",
            stats["tailored"],
            stats["untailored_eligible"],
            0,
            stats["tailor_exhausted"],
            f"Eligible means fit score >= {min_score}.",
        ),
        (
            "Cover",
            stats["with_cover_letter"],
            stats["pending_cover"],
            0,
            stats["cover_exhausted"],
            f"Uses the same fit score threshold ({min_score}+).",
        ),
        (
            "Apply",
            stats["applied"],
            stats["ready_to_apply"],
            stats["apply_manual"],
            stats["apply_in_progress"] + stats["apply_failed"],
            (
                f"{stats['apply_in_progress']} in progress, "
                f"{stats['apply_failed']} failed, {stats['apply_manual']} manual."
            ),
        ),
    ]

    for stage, done, queued, skipped, attention, notes in rows:
        table.add_row(stage, str(done), str(queued), str(skipped), str(attention), notes)

    return Panel(table, title="Pipeline Stage Workload", border_style="blue")


def _render_apply_breakdown(stats: dict) -> Panel:
    table = Table(expand=True, show_edge=False)
    table.add_column("Apply Status")
    table.add_column("Jobs", justify="right")

    labels = {
        "not_started": "Not started",
        "in_progress": "In progress",
        "failed": "Failed",
        "manual": "Manual ATS",
        "applied": "Applied",
    }
    for status, count in stats["apply_status_breakdown"]:
        table.add_row(labels.get(status, status), str(count))

    return Panel(table, title="Apply Status", border_style="yellow")


def _render_score_distribution(stats: dict) -> Panel:
    table = Table(expand=True, show_edge=False)
    table.add_column("Score", justify="center")
    table.add_column("Jobs", justify="right")
    table.add_column("Bar")

    distribution = stats["score_distribution"]
    if not distribution:
        table.add_row("-", "0", "No scored jobs yet.")
        return Panel(table, title="Score Distribution", border_style="green")

    max_count = max(count for _, count in distribution) or 1
    for score, count in distribution:
        filled = max(1, int((count / max_count) * 20))
        if score >= 7:
            color = "green"
        elif score >= 5:
            color = "yellow"
        else:
            color = "red"
        bar = f"[{color}]{'=' * filled}[/{color}]"
        table.add_row(str(score), str(count), bar)

    return Panel(table, title="Score Distribution", border_style="green")


def _render_sources(stats: dict) -> Panel:
    table = Table(expand=True, show_edge=False)
    table.add_column("Source")
    table.add_column("Jobs", justify="right")

    by_site = stats["by_site"]
    if not by_site:
        table.add_row("No jobs yet", "0")
    else:
        for site, count in by_site[:10]:
            table.add_row(site or "Unknown", str(count))

    return Panel(table, title="Top Sources", border_style="magenta")


def _render_blocked_sites(stats: dict) -> Panel:
    table = Table(expand=True, show_edge=False)
    table.add_column("Blocked Site")
    table.add_column("Queued", justify="right")

    blocked = stats["pending_detail_blocked_sites"]
    if not blocked:
        table.add_row("None", "0")
    else:
        for site, count in blocked:
            table.add_row(site, str(count))

    return Panel(table, title="Blocked Enrichment Sites", border_style="red")


def render_status_dashboard(stats: dict, *, min_score: int) -> Group:
    """Render the terminal DB status dashboard."""
    return Group(
        Text("ApplyPilot DB Status", style="bold"),
        _render_pipeline_table(stats, min_score),
        Columns(
            [
                _render_score_distribution(stats),
                _render_sources(stats),
                _render_apply_breakdown(stats),
                _render_blocked_sites(stats),
            ],
            expand=False,
            equal=False,
            align="left",
        ),
    )
