"""ApplyPilot configuration: paths, platform detection, user data."""

from importlib.util import find_spec
import os
import platform
import shutil
from pathlib import Path

# User data directory — all user-specific files live here
APP_DIR = Path(os.environ.get("APPLYPILOT_DIR", Path.home() / ".applypilot"))

# Core paths
DB_PATH = APP_DIR / "applypilot.db"
PROFILE_PATH = APP_DIR / "profile.json"
RESUME_PATH = APP_DIR / "resume.txt"
RESUME_PDF_PATH = APP_DIR / "resume.pdf"
SEARCH_CONFIG_PATH = APP_DIR / "searches.yaml"
ENV_PATH = APP_DIR / ".env"

# Generated output
TAILORED_DIR = APP_DIR / "tailored_resumes"
COVER_LETTER_DIR = APP_DIR / "cover_letters"
LOG_DIR = APP_DIR / "logs"
BROWSER_USE_LOG_DIR = LOG_DIR / "browser_use"

# Chrome worker isolation
CHROME_WORKER_DIR = APP_DIR / "chrome-workers"
APPLY_WORKER_DIR = APP_DIR / "apply-workers"

# Package-shipped config (YAML registries)
PACKAGE_DIR = Path(__file__).parent
CONFIG_DIR = PACKAGE_DIR / "config"


def get_chrome_path() -> str:
    """Auto-detect Chrome/Chromium executable path, cross-platform.

    Override with CHROME_PATH environment variable.
    """
    env_path = os.environ.get("CHROME_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    system = platform.system()

    if system == "Windows":
        candidates = [
            Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
            / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
        ]
    elif system == "Darwin":
        candidates = [
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
        ]
    else:  # Linux
        candidates = []
        for name in ("google-chrome", "google-chrome-stable", "chromium-browser", "chromium"):
            found = shutil.which(name)
            if found:
                candidates.append(Path(found))

    for c in candidates:
        if c and c.exists():
            return str(c)

    # Fall back to PATH search
    for name in ("google-chrome", "google-chrome-stable", "chromium-browser", "chromium", "chrome"):
        found = shutil.which(name)
        if found:
            return found

    raise FileNotFoundError("Chrome/Chromium not found. Install Chrome or set CHROME_PATH environment variable.")


def get_chrome_user_data() -> Path:
    """Default Chrome user data directory, cross-platform."""
    system = platform.system()
    if system == "Windows":
        return Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "User Data"
    elif system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Google" / "Chrome"
    else:
        return Path.home() / ".config" / "google-chrome"


def ensure_dirs():
    """Create all required directories."""
    for d in [APP_DIR, TAILORED_DIR, COVER_LETTER_DIR, LOG_DIR, BROWSER_USE_LOG_DIR, CHROME_WORKER_DIR, APPLY_WORKER_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_profile() -> dict:
    """Load user profile from ~/.applypilot/profile.json."""
    import json

    if not PROFILE_PATH.exists():
        raise FileNotFoundError(f"Profile not found at {PROFILE_PATH}. Run `applypilot init` first.")
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def load_search_config() -> dict:
    """Load search configuration from ~/.applypilot/searches.yaml."""
    import yaml

    if not SEARCH_CONFIG_PATH.exists():
        # Fall back to package-shipped example
        example = CONFIG_DIR / "searches.example.yaml"
        if example.exists():
            return _normalize_search_config(yaml.safe_load(example.read_text(encoding="utf-8")))
        return {}
    return _normalize_search_config(yaml.safe_load(SEARCH_CONFIG_PATH.read_text(encoding="utf-8")))


def _clean_string_list(values: object) -> list[str]:
    """Normalize YAML list-like values into non-empty strings."""
    if not isinstance(values, list):
        return []

    cleaned: list[str] = []
    for value in values:
        if value is None:
            continue
        text = value if isinstance(value, str) else str(value)
        text = text.strip()
        if text:
            cleaned.append(text)
    return cleaned


def _normalize_search_config(raw_cfg: dict | None) -> dict:
    """Map legacy/example search config shapes into discovery's expected schema."""
    cfg = dict(raw_cfg or {})

    defaults = dict(cfg.get("defaults") or {})

    raw_queries = cfg.get("queries") or cfg.get("searches") or []
    queries: list[dict] = []
    for item in raw_queries:
        if isinstance(item, str):
            query = item.strip()
            if query:
                queries.append({"query": query})
            continue
        if not isinstance(item, dict):
            continue
        query = item.get("query") or item.get("search")
        if query is None:
            continue
        query_text = str(query).strip()
        if not query_text:
            continue
        normalized = dict(item)
        normalized["query"] = query_text
        normalized.pop("search", None)
        normalized.pop("tier", None)
        queries.append(normalized)
    cfg["queries"] = queries

    raw_locations = cfg.get("search_locations")
    if raw_locations is None:
        raw_locations = cfg.get("locations") or []
    locations: list[dict] = []
    for item in raw_locations:
        if isinstance(item, str):
            location = item.strip()
            if location:
                locations.append({"label": location, "location": location, "remote": "remote" in location.lower()})
            continue
        if not isinstance(item, dict):
            continue
        location_value = item.get("location") or item.get("label")
        if location_value is None:
            continue
        location = str(location_value).strip()
        if not location:
            continue
        label_value = item.get("label", location)
        label = str(label_value).strip() if label_value is not None else location
        normalized = dict(item)
        normalized["label"] = label or location
        normalized["location"] = location
        normalized["remote"] = bool(item.get("remote", False))
        locations.append(normalized)

    cfg["search_locations"] = locations
    cfg["locations"] = locations

    sites = _clean_string_list(cfg.get("sites"))
    if not sites:
        sites = _clean_string_list(cfg.get("boards"))
    cfg["sites"] = sites or None

    country = cfg.get("country_indeed")
    if country is None:
        country = cfg.get("country")
    if country is not None:
        defaults["country_indeed"] = str(country).strip().lower()
    cfg["defaults"] = defaults

    location_block = cfg.get("location_rules")
    if location_block is None:
        location_block = cfg.get("location") or {}
    if not cfg.get("location_accept"):
        cfg["location_accept"] = _clean_string_list(location_block.get("accept_patterns"))
    else:
        cfg["location_accept"] = _clean_string_list(cfg.get("location_accept"))
    if not cfg.get("location_reject_non_remote"):
        cfg["location_reject_non_remote"] = _clean_string_list(location_block.get("reject_patterns"))
    else:
        cfg["location_reject_non_remote"] = _clean_string_list(cfg.get("location_reject_non_remote"))
    cfg["location_rules"] = {
        "accept_patterns": cfg["location_accept"],
        "reject_patterns": cfg["location_reject_non_remote"],
    }

    location_labels = cfg.get("location_labels")
    if isinstance(location_labels, list):
        cfg["location_labels"] = _clean_string_list(location_labels) or None

    return cfg


def load_sites_config() -> dict:
    """Load sites.yaml configuration (sites list, manual_ats, blocked, etc.)."""
    import yaml

    path = CONFIG_DIR / "sites.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def is_manual_ats(url: str | None) -> bool:
    """Check if a URL routes through an ATS that requires manual application."""
    if not url:
        return False
    sites_cfg = load_sites_config()
    domains = sites_cfg.get("manual_ats", [])
    url_lower = url.lower()
    return any(domain in url_lower for domain in domains)


def load_blocked_sites() -> tuple[set[str], list[str]]:
    """Load blocked sites and URL patterns from sites.yaml.

    Returns:
        (blocked_site_names, blocked_url_patterns)
    """
    cfg = load_sites_config()
    blocked = cfg.get("blocked", {})
    sites = set(blocked.get("sites", []))
    patterns = blocked.get("url_patterns", [])
    return sites, patterns


def load_blocked_sso() -> list[str]:
    """Load blocked SSO domains from sites.yaml."""
    cfg = load_sites_config()
    return cfg.get("blocked_sso", [])


def load_base_urls() -> dict[str, str | None]:
    """Load site base URLs for URL resolution from sites.yaml."""
    cfg = load_sites_config()
    return cfg.get("base_urls", {})


# ---------------------------------------------------------------------------
# Default values — referenced across modules instead of magic numbers
# ---------------------------------------------------------------------------

DEFAULTS = {
    "min_score": 7,
    "max_apply_attempts": 3,
    "max_tailor_attempts": 5,
    "poll_interval": 60,
    "apply_timeout": 300,
    "viewport": "1280x900",
}


def load_env():
    """Load environment variables from ~/.applypilot/.env if it exists."""
    from dotenv import load_dotenv

    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    # Also try CWD .env as fallback
    load_dotenv()


# ---------------------------------------------------------------------------
# Tier system — feature gating by installed dependencies
# ---------------------------------------------------------------------------

TIER_LABELS = {
    1: "Discovery",
    2: "AI Scoring & Tailoring",
    3: "Full Auto-Apply",
}

TIER_COMMANDS: dict[int, list[str]] = {
    1: ["init", "run discover", "run enrich", "status", "dashboard"],
    2: ["run score", "run tailor", "run cover", "run pdf", "run"],
    3: ["apply"],
}


def get_tier() -> int:
    """Detect the current tier based on available dependencies.

    Tier 1 (Discovery):            Python + pip
    Tier 2 (AI Scoring & Tailoring): + LLM API key
    Tier 3 (Full Auto-Apply):       + browser-use + Chrome
    """
    load_env()

    has_provider_source = any(
        os.environ.get(k) for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_URL")
    )
    has_model_and_generic_key = bool((os.environ.get("LLM_MODEL") or "").strip()) and bool(
        (os.environ.get("LLM_API_KEY") or "").strip()
    )
    has_llm = has_provider_source or has_model_and_generic_key
    if not has_llm:
        return 1

    has_browser_use = find_spec("browser_use") is not None
    try:
        get_chrome_path()
        has_chrome = True
    except FileNotFoundError:
        has_chrome = False

    if has_browser_use and has_chrome:
        return 3

    return 2


def check_tier(required: int, feature: str) -> None:
    """Raise SystemExit with a clear message if the current tier is too low.

    Args:
        required: Minimum tier needed (1, 2, or 3).
        feature: Human-readable description of the feature being gated.
    """
    current = get_tier()
    if current >= required:
        return

    from rich.console import Console

    _console = Console(stderr=True)

    missing: list[str] = []
    has_provider_source = any(
        os.environ.get(k) for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_URL")
    )
    has_model_and_generic_key = bool((os.environ.get("LLM_MODEL") or "").strip()) and bool(
        (os.environ.get("LLM_API_KEY") or "").strip()
    )
    if required >= 2 and not (has_provider_source or has_model_and_generic_key):
        missing.append(
            "LLM config — run [bold]applypilot init[/bold] or set one of "
            "GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY / LLM_URL "
            "(or set LLM_MODEL with LLM_API_KEY)"
        )
    if required >= 3:
        if find_spec("browser_use") is None:
            missing.append(
                "browser-use — install with [bold]pip install -e .[/bold] or [bold]pip install browser-use[/bold]"
            )
        try:
            get_chrome_path()
        except FileNotFoundError:
            missing.append("Chrome/Chromium — install or set CHROME_PATH")

    _console.print(
        f"\n[red]'{feature}' requires {TIER_LABELS.get(required, f'Tier {required}')} (Tier {required}).[/red]\n"
        f"Current tier: {TIER_LABELS.get(current, f'Tier {current}')} (Tier {current})."
    )
    if missing:
        _console.print("\n[yellow]Missing:[/yellow]")
        for m in missing:
            _console.print(f"  - {m}")
    _console.print()
    raise SystemExit(1)
