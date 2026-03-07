from applypilot import config


def test_load_search_config_prefers_renamed_location_keys(tmp_path, monkeypatch) -> None:
    search_path = tmp_path / "searches.yaml"
    search_path.write_text(
        """
search_locations:
  - location: "San Francisco, CA"
    remote: false
location_rules:
  accept_patterns:
    - "San Francisco"
  reject_patterns:
    - "New York only"
queries:
  - "Backend Engineer"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "SEARCH_CONFIG_PATH", search_path)

    cfg = config.load_search_config()

    assert cfg["queries"] == [{"query": "Backend Engineer"}]
    assert cfg["search_locations"] == [{"label": "San Francisco, CA", "location": "San Francisco, CA", "remote": False}]
    assert cfg["locations"] == [{"label": "San Francisco, CA", "location": "San Francisco, CA", "remote": False}]
    assert cfg["location_rules"] == {
        "accept_patterns": ["San Francisco"],
        "reject_patterns": ["New York only"],
    }
    assert cfg["location_accept"] == ["San Francisco"]
    assert cfg["location_reject_non_remote"] == ["New York only"]


def test_load_search_config_normalizes_legacy_keys_and_empty_sites(tmp_path, monkeypatch) -> None:
    search_path = tmp_path / "searches.yaml"
    search_path.write_text(
        """
defaults:
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
    assert cfg["queries"] == [{"query": "Data Engineer"}]
    assert cfg["search_locations"] == [{"label": "Remote or Hybrid", "location": "Remote or Hybrid", "remote": False}]
    assert cfg["locations"] == [{"label": "Remote or Hybrid", "location": "Remote or Hybrid", "remote": False}]
    assert cfg["location_accept"] == ["Remote", "California"]
    assert cfg["location_reject_non_remote"] == ["India"]
    assert cfg["location_rules"] == {
        "accept_patterns": ["Remote", "California"],
        "reject_patterns": ["India"],
    }


def test_load_search_config_does_not_build_location_from_defaults_when_missing(tmp_path, monkeypatch) -> None:
    search_path = tmp_path / "searches.yaml"
    search_path.write_text(
        """
defaults:
  distance: 0
queries:
  - "Backend Engineer"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "SEARCH_CONFIG_PATH", search_path)

    cfg = config.load_search_config()

    assert cfg["queries"] == [{"query": "Backend Engineer"}]
    assert cfg["search_locations"] == []
    assert cfg["locations"] == []
    assert cfg["sites"] is None
