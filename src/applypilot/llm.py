"""Unified LLM client for ApplyPilot using LiteLLM.

Runtime contract:
  - If set, LLM_MODEL must be a fully-qualified LiteLLM model string
    (for example: openai/gpt-4o-mini, anthropic/claude-3-5-haiku-latest,
    gemini/gemini-3.0-flash).
  - If set, LLM_MODEL_HIGH follows the same model contract and is intended for
    quality-sensitive writing tasks. If unset, it falls back to LLM_MODEL.
  - If the active model env var is unset, provider is inferred by first
    configured source: GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY,
    then LLM_URL.
  - Credentials come from provider env vars or generic LLM_API_KEY.
  - LLM_URL is optional for custom OpenAI-compatible endpoints.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
import math
import os
from typing import Any, Literal, TypedDict, Unpack
import warnings

import litellm

from applypilot.llm_cost import record_llm_cost_estimate

# Suppress pydantic serialization warnings emitted by litellm internals when
# provider responses have fewer fields than the full ModelResponse schema.
warnings.filterwarnings("ignore", category=UserWarning, message="Pydantic serializer warnings")

log = logging.getLogger(__name__)

_MAX_RETRIES = 5
_TIMEOUT = 120  # seconds
_INFERRED_SOURCE_ORDER: tuple[tuple[str, str], ...] = (
    ("gemini", "GEMINI_API_KEY"),
    ("openai", "OPENAI_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("openai", "LLM_URL"),
)
_DEFAULT_MODEL_BY_PROVIDER = {
    "gemini": "gemini/gemini-3.0-flash",
    "openai": "openai/gpt-5-mini",
    "anthropic": "anthropic/claude-haiku-4-5",
}
_DEFAULT_LOCAL_MODEL = "openai/local-model"

_COST_HELPER_CALLS: tuple[tuple[str, tuple[dict[str, object], ...]], ...] = (
    (
        "response_cost",
        (
            {"completion_response": None},
            {"response_object": None},
            {"response": None},
        ),
    ),
    (
        "completion_cost",
        (
            {"completion_response": None},
            {"completion_response": None, "model": None},
            {"response_object": None},
            {"response_object": None, "model": None},
        ),
    ),
    (
        "response_cost_calculator",
        (
            {"response_object": None},
            {"response_object": None, "model": None},
            {"response": None},
            {"response": None, "model": None},
        ),
    ),
)


@dataclass(frozen=True)
class LLMConfig:
    """LLM configuration consumed by LLMClient."""

    provider: str
    api_base: str | None
    model: str
    api_key: str


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class LiteLLMExtra(TypedDict, total=False):
    stop: str | list[str]
    top_p: float
    seed: int
    stream: bool
    response_format: dict[str, Any]
    tools: list[dict[str, Any]]
    tool_choice: str | dict[str, Any]
    fallbacks: list[str]


ModelTier = Literal["default", "high"]


def _env_get(env: Mapping[str, str], key: str) -> str:
    value = env.get(key, "")
    if value is None:
        return ""
    return str(value).strip()


def _provider_from_model(model: str) -> str:
    provider, _, model_name = model.partition("/")
    if not provider or not model_name:
        raise RuntimeError("LLM_MODEL must include a provider prefix (for example 'openai/gpt-4o-mini').")
    return provider


def _model_env_var(model_tier: ModelTier) -> str:
    return "LLM_MODEL_HIGH" if model_tier == "high" else "LLM_MODEL"


def _configured_model(env: Mapping[str, str], model_tier: ModelTier) -> str:
    model = _env_get(env, _model_env_var(model_tier))
    if model or model_tier == "default":
        return model
    return _env_get(env, "LLM_MODEL")


def _normalize_cost(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        cost = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(cost) or math.isinf(cost) or cost < 0:
        return None
    return cost


def _estimate_response_cost(response: object, *, model: str) -> float | None:
    """Best-effort LiteLLM response pricing using built-in helpers."""
    errors: list[str] = []

    for helper_name, call_variants in _COST_HELPER_CALLS:
        helper = getattr(litellm, helper_name, None)
        if helper is None:
            continue

        for variant in call_variants:
            kwargs = dict(variant)
            for key, value in list(kwargs.items()):
                if value is None:
                    kwargs[key] = response if "response" in key else model
            try:
                cost = _normalize_cost(helper(**kwargs))
            except TypeError:
                continue
            except Exception as exc:  # pragma: no cover - helper internals vary by LiteLLM version.
                errors.append(f"{helper_name}: {exc}")
                break
            if cost is not None:
                return cost

        for args in ((response,), (response, model)):
            try:
                cost = _normalize_cost(helper(*args))
            except TypeError:
                continue
            except Exception as exc:  # pragma: no cover - helper internals vary by LiteLLM version.
                errors.append(f"{helper_name}: {exc}")
                break
            if cost is not None:
                return cost

    if errors:
        log.debug("LiteLLM cost estimate unavailable for %s: %s", model, "; ".join(errors))
    return None


def _infer_provider_and_source(env: Mapping[str, str]) -> tuple[str, str] | None:
    for provider, env_key in _INFERRED_SOURCE_ORDER:
        if _env_get(env, env_key):
            return provider, env_key
    return None


def _infer_provider_for_tier(env: Mapping[str, str], model_tier: ModelTier) -> tuple[str, str] | None:
    inferred = _infer_provider_and_source(env)
    if inferred is not None:
        return inferred
    if model_tier == "high":
        base_model = _env_get(env, "LLM_MODEL")
        if "/" in base_model:
            return _provider_from_model(base_model), "LLM_MODEL"
    return None


def resolve_llm_config(env: Mapping[str, str] | None = None, *, model_tier: ModelTier = "default") -> LLMConfig:
    """Resolve LLM configuration from environment."""
    env_map = env if env is not None else os.environ

    model_env_var = _model_env_var(model_tier)
    model = _configured_model(env_map, model_tier)
    local_url = _env_get(env_map, "LLM_URL")
    inferred = _infer_provider_for_tier(env_map, model_tier)
    if model:
        if "/" in model:
            provider = _provider_from_model(model)
        elif inferred:
            provider, _ = inferred
            model = f"{provider}/{model}"
        else:
            raise RuntimeError(f"{model_env_var} must include a provider prefix (for example 'openai/gpt-4o-mini').")
    else:
        if not inferred:
            raise RuntimeError(
                "No LLM provider configured. Set one of GEMINI_API_KEY, OPENAI_API_KEY, "
                "ANTHROPIC_API_KEY, LLM_URL, or LLM_MODEL."
            )
        provider, source = inferred
        if source == "LLM_URL":
            model = _DEFAULT_LOCAL_MODEL
        else:
            model = _DEFAULT_MODEL_BY_PROVIDER[provider]

    provider_api_key_env = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    api_key_env = provider_api_key_env.get(provider, "LLM_API_KEY")
    api_key = _env_get(env_map, api_key_env) or _env_get(env_map, "LLM_API_KEY")

    if not api_key and not local_url:
        key_help = f"{api_key_env} or LLM_API_KEY" if provider in provider_api_key_env else "LLM_API_KEY"
        raise RuntimeError(
            f"Missing credentials for model '{model}'. Set {key_help}, or set LLM_URL for "
            "a local OpenAI-compatible endpoint."
        )

    return LLMConfig(
        provider=provider,
        api_base=local_url.rstrip("/") if local_url else None,
        model=model,
        api_key=api_key,
    )


class LLMClient:
    """Thin wrapper around LiteLLM completion()."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.provider = config.provider
        self.model = config.model
        litellm.suppress_debug_info = True

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_output_tokens: int = 10000,
        temperature: float | None = None,
        timeout: int = _TIMEOUT,
        num_retries: int = _MAX_RETRIES,
        drop_params: bool = True,
        **extra: Unpack[LiteLLMExtra],
    ) -> str:
        """Send a completion request and return plain text content."""
        try:
            if temperature is None:
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_output_tokens,
                    timeout=timeout,
                    num_retries=num_retries,
                    drop_params=drop_params,
                    api_key=self.config.api_key or None,
                    api_base=self.config.api_base or None,
                    **extra,
                )
            else:
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_output_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    num_retries=num_retries,
                    drop_params=drop_params,
                    api_key=self.config.api_key or None,
                    api_base=self.config.api_base or None,
                    **extra,
                )

            record_llm_cost_estimate(_estimate_response_cost(response, model=self.model))

            choices = getattr(response, "choices", None)
            if not choices:
                raise RuntimeError("LLM response contained no choices.")
            content = response.choices[0].message.content
            text = content.strip() if isinstance(content, str) else str(content).strip()

            if not text:
                raise RuntimeError("LLM response contained no text content.")
            return text
        except Exception as exc:  # pragma: no cover - provider SDK exception types vary by backend/version.
            raise RuntimeError(f"LLM request failed ({self.provider}/{self.model}): {exc}") from exc

    def close(self) -> None:
        """No-op. LiteLLM completion() is stateless per call."""
        return None

    def ask(
        self,
        prompt: str,
        *,
        max_tokens: int = 10000,
        temperature: float | None = None,
        **extra: Unpack[LiteLLMExtra],
    ) -> str:
        """Compatibility wrapper for older single-prompt callers."""
        return self.chat(
            [{"role": "user", "content": prompt}],
            max_output_tokens=max_tokens,
            temperature=temperature,
            **extra,
        )


_instances: dict[tuple[str, str | None, str, str], LLMClient] = {}


def get_client(*, model_tier: ModelTier = "default") -> LLMClient:
    """Return (or create) a cached LLMClient for the requested model tier."""
    try:
        from applypilot.config import load_env

        load_env()
    except ModuleNotFoundError:
        log.debug("python-dotenv not installed; skipping .env auto-load in llm.get_client().")

    config = resolve_llm_config(model_tier=model_tier)
    cache_key = (config.provider, config.api_base, config.model, config.api_key)
    client = _instances.get(cache_key)
    if client is None:
        log.info(
            "LLM provider (%s): %s  model: %s",
            model_tier,
            config.provider,
            config.model,
        )
        client = LLMClient(config)
        _instances[cache_key] = client
    return client


def validate_api_key(provider: str, api_key: str, model: str = "", endpoint: str = "") -> tuple[bool, str]:
    """Validate an API key by making a minimal test request.

    Args:
        provider: "gemini", "openai", or "local"
        api_key: The API key to validate
        model: Optional model name override
        endpoint: Required for "local" provider

    Returns:
        (is_valid, error_message) - error_message is empty if valid
    """
    try:
        env: dict[str, str] = {}
        if provider == "gemini":
            env["GEMINI_API_KEY"] = api_key
            env["LLM_MODEL"] = model or "gemini-2.0-flash"
        elif provider == "openai":
            env["OPENAI_API_KEY"] = api_key
            env["LLM_MODEL"] = model or "gpt-4o-mini"
        elif provider == "local":
            if not endpoint:
                return False, "Local endpoint URL is required"
            env["LLM_URL"] = endpoint.rstrip("/")
            env["LLM_MODEL"] = model or "local-model"
            if api_key:
                env["LLM_API_KEY"] = api_key
        else:
            return False, f"Unknown provider: {provider}"

        config = resolve_llm_config(env)

        # Simple test request
        response = litellm.completion(
            model=config.model,
            messages=[{"role": "user", "content": "Reply with only the word 'ok'."}],
            max_tokens=10,
            temperature=0.0,
            timeout=30,
            num_retries=0,
            drop_params=True,
            api_key=config.api_key or None,
            api_base=config.api_base or None,
        )

        choices = getattr(response, "choices", None)
        content = choices[0].message.content if choices else ""
        text = content.strip() if isinstance(content, str) else str(content).strip()

        if text:
            return True, ""
        return False, "Empty response from API"

    except litellm.AuthenticationError:
        return False, "Invalid API key"
    except litellm.RateLimitError:
        return True, ""
    except litellm.APIConnectionError:
        return False, "Could not connect to API endpoint"
    except litellm.Timeout:
        return False, "API request timed out"
    except litellm.APIError as e:
        if e.status_code == 403:
            return False, "API key lacks required permissions"
        if e.status_code == 429:
            return True, ""
        return False, f"API error: {e.status_code}"
    except litellm.BadRequestError as e:
        return False, f"Invalid model or request: {e}"
    except Exception as e:
        return False, f"Validation failed: {str(e)}"
