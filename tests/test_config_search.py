from applypilot import config


def test_load_search_config_normalizes_legacy_keys_and_empty_sites(tmp_path, monkeypatch) -> None:
    search_path = tmp_path / "searches.yaml"
    search_path.write_text(
        """
defaults:
  location: "Remote or Hybrid"
  distance: 50
  hours_old: 72
  results_per_site: 100
locations:
  - location: "Remote or Hybrid"
    remote: false
queries:
  - query: "Data Engineer"
    tier: "2"
  - query:
boards:
  - indeed
  - linkedin
country: "USA"
location:
  accept_patterns:
    - "Remote"
    - "California"
  reject_patterns:
    - "India"
sites:
  -
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "SEARCH_CONFIG_PATH", search_path)

    cfg = config.load_search_config()

    assert cfg["sites"] == ["indeed", "linkedin"]
    assert cfg["defaults"]["country_indeed"] == "usa"
    assert cfg["queries"] == [{"query": "Data Engineer", "tier": 2}]
    assert cfg["locations"] == [{"label": "Remote or Hybrid", "location": "Remote or Hybrid", "remote": False}]
    assert cfg["location_accept"] == ["Remote", "California"]
    assert cfg["location_reject_non_remote"] == ["India"]


def test_load_search_config_builds_location_from_defaults_when_missing(tmp_path, monkeypatch) -> None:
    search_path = tmp_path / "searches.yaml"
    search_path.write_text(
        """
defaults:
  location: "Remote"
  distance: 0
queries:
  - "Backend Engineer"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "SEARCH_CONFIG_PATH", search_path)

    cfg = config.load_search_config()

    assert cfg["queries"] == [{"query": "Backend Engineer", "tier": 1}]
    assert cfg["locations"] == [{"label": "Remote", "location": "Remote", "remote": True}]
    assert cfg["sites"] is None
