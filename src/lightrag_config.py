"""Provider-switchable LightRAG configuration.

Set providers and models in .env instead of editing the ingest/eval scripts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

load_dotenv()


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_LOCAL_LLM_MODEL = "granite4.1:8b"
DEFAULT_LOCAL_EMBED_MODEL = "qwen3-embedding:4b"
DEFAULT_LOCAL_EMBED_DIM = 2560
DEFAULT_OPENAI_LLM_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_EMBED_DIM = 1536
DEFAULT_EMBED_MAX_TOKENS = 8192


@dataclass(frozen=True)
class RAGConfig:
    llm_provider: str
    llm_model: str
    embed_provider: str
    embed_model: str
    embed_dim: int
    embed_max_tokens: int
    storage_dir: str


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _int_env(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def get_config(storage_dir: str | Path = "rag_storage") -> RAGConfig:
    llm_provider = (_env("RAG_LLM_PROVIDER", "ollama") or "ollama").lower()
    embed_provider = (_env("RAG_EMBED_PROVIDER", "ollama") or "ollama").lower()

    llm_model = _env("RAG_LLM_MODEL")
    if llm_model is None:
        if llm_provider == "ollama":
            llm_model = DEFAULT_LOCAL_LLM_MODEL
        elif llm_provider == "openai":
            llm_model = DEFAULT_OPENAI_LLM_MODEL
        elif llm_provider == "anthropic":
            llm_model = "claude-sonnet-4-20250514"
        else:
            raise ValueError(f"Unsupported RAG_LLM_PROVIDER={llm_provider!r}")

    embed_model = _env("RAG_EMBED_MODEL")
    if embed_model is None:
        if embed_provider == "ollama":
            embed_model = DEFAULT_LOCAL_EMBED_MODEL
        elif embed_provider == "openai":
            embed_model = DEFAULT_OPENAI_EMBED_MODEL
        else:
            raise ValueError(
                f"Unsupported RAG_EMBED_PROVIDER={embed_provider!r}. "
                "Anthropic/Claude does not provide embeddings in this LightRAG setup; "
                "use RAG_EMBED_PROVIDER=ollama or openai."
            )

    default_embed_dim = (
        DEFAULT_OPENAI_EMBED_DIM if embed_provider == "openai" else DEFAULT_LOCAL_EMBED_DIM
    )
    return RAGConfig(
        llm_provider=llm_provider,
        llm_model=llm_model,
        embed_provider=embed_provider,
        embed_model=embed_model,
        embed_dim=_int_env("RAG_EMBED_DIM", default_embed_dim),
        embed_max_tokens=_int_env("RAG_EMBED_MAX_TOKENS", DEFAULT_EMBED_MAX_TOKENS),
        storage_dir=str(storage_dir),
    )


def _api_kwargs(provider: str, kind: str) -> dict[str, Any]:
    prefix = f"RAG_{kind.upper()}"
    kwargs: dict[str, Any] = {}
    api_key = _env(f"{prefix}_API_KEY")
    base_url = _env(f"{prefix}_BASE_URL")

    if api_key:
        kwargs["api_key"] = api_key
    elif provider == "openai":
        kwargs["api_key"] = _env("OPENAI_API_KEY")
    elif provider == "anthropic":
        kwargs["api_key"] = _env("ANTHROPIC_API_KEY")

    if base_url:
        kwargs["base_url"] = base_url
    return {key: value for key, value in kwargs.items() if value is not None}


def _build_llm(config: RAGConfig) -> tuple[Any, dict[str, Any]]:
    if config.llm_provider == "ollama":
        from lightrag.llm.ollama import ollama_model_complete

        return ollama_model_complete, {"host": _env("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)}

    if config.llm_provider == "openai":
        from lightrag.llm.openai import openai_complete

        return openai_complete, _api_kwargs("openai", "llm")

    if config.llm_provider == "anthropic":
        from lightrag.llm.anthropic import anthropic_complete

        kwargs = _api_kwargs("anthropic", "llm")
        kwargs.setdefault("max_tokens", _int_env("RAG_LLM_MAX_TOKENS", 4096))
        return anthropic_complete, kwargs

    raise ValueError(f"Unsupported RAG_LLM_PROVIDER={config.llm_provider!r}")


def _build_embedding(config: RAGConfig) -> EmbeddingFunc:
    if config.embed_provider == "ollama":
        from lightrag.llm.ollama import ollama_embed

        return EmbeddingFunc(
            embedding_dim=config.embed_dim,
            max_token_size=config.embed_max_tokens,
            model_name=config.embed_model,
            func=partial(
                ollama_embed.func,
                embed_model=config.embed_model,
                host=_env("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
            ),
        )

    if config.embed_provider == "openai":
        from lightrag.llm.openai import openai_embed

        return EmbeddingFunc(
            embedding_dim=config.embed_dim,
            max_token_size=config.embed_max_tokens,
            model_name=config.embed_model,
            func=partial(
                openai_embed.func,
                model=config.embed_model,
                **_api_kwargs("openai", "embed"),
            ),
        )

    raise ValueError(
        f"Unsupported RAG_EMBED_PROVIDER={config.embed_provider!r}. "
        "Use ollama or openai embeddings."
    )


def build_lightrag(storage_dir: str | Path = "rag_storage") -> LightRAG:
    config = get_config(storage_dir)
    llm_model_func, llm_model_kwargs = _build_llm(config)
    return LightRAG(
        working_dir=config.storage_dir,
        embedding_func=_build_embedding(config),
        llm_model_func=llm_model_func,
        llm_model_name=config.llm_model,
        llm_model_kwargs=llm_model_kwargs,
    )


def describe_config(storage_dir: str | Path = "rag_storage") -> str:
    config = get_config(storage_dir)
    return (
        f"llm={config.llm_provider}:{config.llm_model} "
        f"embed={config.embed_provider}:{config.embed_model} "
        f"dim={config.embed_dim} storage={config.storage_dir}"
    )
