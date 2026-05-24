"""Provider-switchable DeepEval judge configuration."""
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


DEFAULT_LOCAL_JUDGE_MODEL = "granite4.1:8b"
DEFAULT_LOCAL_CAPABILITY_MODEL = "granite4"
DEFAULT_OPENAI_JUDGE_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_JUDGE_MODEL = "claude-sonnet-4-6-20250514"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _float_env(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _int_env(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def get_judge_provider() -> str:
    return (_env("DEEPEVAL_PROVIDER", "ollama") or "ollama").lower()


def get_judge_model_name() -> str:
    provider = get_judge_provider()
    configured = _env("DEEPEVAL_MODEL")
    if configured:
        return configured
    if provider == "ollama":
        return DEFAULT_LOCAL_JUDGE_MODEL
    if provider == "openai":
        return DEFAULT_OPENAI_JUDGE_MODEL
    if provider == "anthropic":
        return DEFAULT_ANTHROPIC_JUDGE_MODEL
    raise ValueError(f"Unsupported DEEPEVAL_PROVIDER={provider!r}")


def build_judge_model() -> Any:
    """Return the DeepEval model object for metrics."""
    provider = get_judge_provider()
    model_name = get_judge_model_name()
    temperature = _float_env("DEEPEVAL_TEMPERATURE", 0.0)

    if provider == "ollama":
        from deepeval.models import OllamaModel
        from deepeval.models.llms.constants import OLLAMA_MODELS_DATA

        judge = OllamaModel(
            model=model_name,
            base_url=_env("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
            temperature=temperature,
            generation_kwargs={"num_predict": _int_env("DEEPEVAL_MAX_TOKENS", 1024)},
        )
        if judge.model_data is None:
            capability_model = _env(
                "DEEPEVAL_OLLAMA_CAPABILITY_MODEL", DEFAULT_LOCAL_CAPABILITY_MODEL
            )
            judge.model_data = OLLAMA_MODELS_DATA.get(capability_model)
        return judge

    if provider == "openai":
        from deepeval.models import GPTModel

        return GPTModel(
            model=model_name,
            api_key=_env("DEEPEVAL_API_KEY") or _env("OPENAI_API_KEY"),
            base_url=_env("DEEPEVAL_BASE_URL"),
            temperature=temperature,
        )

    if provider == "anthropic":
        from deepeval.models import AnthropicModel

        return AnthropicModel(
            model=model_name,
            api_key=_env("DEEPEVAL_API_KEY") or _env("ANTHROPIC_API_KEY"),
            temperature=temperature,
            max_tokens=_int_env("DEEPEVAL_MAX_TOKENS", 1024),
        )

    raise ValueError(f"Unsupported DEEPEVAL_PROVIDER={provider!r}")


def describe_judge_model() -> str:
    return f"judge={get_judge_provider()}:{get_judge_model_name()}"
