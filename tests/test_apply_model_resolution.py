from applypilot.apply import launcherv2


def test_resolve_apply_model_defaults_to_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "o-key")

    assert launcherv2.resolve_apply_model() == "openai/gpt-4o-mini"


def test_resolve_apply_model_normalizes_unprefixed_override(monkeypatch) -> None:
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "o-key")

    assert launcherv2.resolve_apply_model("gpt-4o-mini") == "openai/gpt-4o-mini"


def test_build_llm_strips_openai_provider_prefix(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class FakeChatOpenAI:
        def __init__(self, *, model: str) -> None:
            captured["model"] = model

    monkeypatch.setattr(launcherv2, "ChatOpenAI", FakeChatOpenAI)

    launcherv2._build_llm("openai/local-model")

    assert captured["model"] == "local-model"
