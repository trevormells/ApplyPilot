from applypilot.scoring.validator import validate_json_fields


def _profile() -> dict:
    return {
        "resume_facts": {
            "preserved_companies": ["Meta", "Bitfocus", "Alameda County"],
            "preserved_school": "",
        }
    }


def test_validate_json_fields_accepts_company_in_subtitle() -> None:
    data = {
        "title": "AI Consultant",
        "summary": "Built AI and data systems.",
        "skills": {"Languages": "Python"},
        "experience": [
            {
                "header": "Sr. Python Engineer",
                "subtitle": "Meta | June 2025 - Present",
                "bullets": ["Built multimodal training pipelines."],
            },
            {
                "header": "Senior Data Engineer",
                "subtitle": "Bitfocus | Nov 2019 - July 2022",
                "bullets": ["Built analytics infrastructure."],
            },
            {
                "header": "Data Systems Engineer",
                "subtitle": "Alameda County | Aug 2017 - Nov 2019",
                "bullets": ["Built reporting pipelines."],
            },
        ],
        "projects": [],
        "education": "UC Santa Barbara",
    }

    result = validate_json_fields(data, _profile())

    assert result["passed"] is True
    assert result["errors"] == []


def test_validate_json_fields_rejects_missing_preserved_company() -> None:
    data = {
        "title": "AI Consultant",
        "summary": "Built AI and data systems.",
        "skills": {"Languages": "Python"},
        "experience": [
            {
                "header": "Sr. Python Engineer",
                "subtitle": "Meta | June 2025 - Present",
                "bullets": ["Built multimodal training pipelines."],
            },
            {
                "header": "Senior Data Engineer",
                "subtitle": "Bitfocus | Nov 2019 - July 2022",
                "bullets": ["Built analytics infrastructure."],
            },
        ],
        "projects": [],
        "education": "UC Santa Barbara",
    }

    result = validate_json_fields(data, _profile())

    assert result["passed"] is False
    assert "Company 'Alameda County' missing from experience" in result["errors"]
