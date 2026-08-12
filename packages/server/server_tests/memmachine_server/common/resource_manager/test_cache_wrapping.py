"""Tests that managers wrap built resources when a cache store is supplied."""

from pathlib import Path

import pytest

from memmachine_server.common.cache.llm_cache_store import LLMCacheStore
from memmachine_server.common.configuration.embedder_conf import (
    EmbeddersConf,
    OpenAIEmbedderConf,
)
from memmachine_server.common.configuration.language_model_conf import (
    LanguageModelsConf,
    LiteLLMLanguageModelConf,
    OpenAIChatCompletionsLanguageModelConf,
)
from memmachine_server.common.embedder.caching_embedder import CachingEmbedder
from memmachine_server.common.embedder.openai_embedder import OpenAIEmbedder
from memmachine_server.common.language_model.caching_language_model import (
    CachingLanguageModel,
)
from memmachine_server.common.language_model.openai_chat_completions_language_model import (
    OpenAIChatCompletionsLanguageModel,
)
from memmachine_server.common.resource_manager.embedder_manager import EmbedderManager
from memmachine_server.common.resource_manager.language_model_manager import (
    LanguageModelManager,
)


def _embedders_conf() -> EmbeddersConf:
    return EmbeddersConf(
        openai={"e": OpenAIEmbedderConf(model="text-embedding-3-small")}
    )


def _language_models_conf() -> LanguageModelsConf:
    return LanguageModelsConf(
        openai_chat_completions_language_model_confs={
            "m": OpenAIChatCompletionsLanguageModelConf(model="gpt-5-nano")
        }
    )


@pytest.mark.asyncio
async def test_embedder_wrapped_with_cache(tmp_path: Path) -> None:
    """An embedder is wrapped in CachingEmbedder when a store is given."""
    store = LLMCacheStore(str(tmp_path / "cache.db"))
    mgr = EmbedderManager(_embedders_conf(), cache_store=store)
    embedder = await mgr.get_embedder("e")
    assert isinstance(embedder, CachingEmbedder)
    await store.close()


@pytest.mark.asyncio
async def test_embedder_not_wrapped_without_cache() -> None:
    """An embedder is returned unwrapped when no store is given."""
    mgr = EmbedderManager(_embedders_conf())
    embedder = await mgr.get_embedder("e")
    assert isinstance(embedder, OpenAIEmbedder)


@pytest.mark.asyncio
async def test_language_model_wrapped_with_cache(tmp_path: Path) -> None:
    """A language model is wrapped in CachingLanguageModel when a store is given."""
    store = LLMCacheStore(str(tmp_path / "cache.db"))
    mgr = LanguageModelManager(_language_models_conf(), cache_store=store)
    model = await mgr.get_language_model("m")
    assert isinstance(model, CachingLanguageModel)
    await store.close()


@pytest.mark.asyncio
async def test_language_model_not_wrapped_without_cache() -> None:
    """A language model is returned unwrapped when no store is given."""
    mgr = LanguageModelManager(_language_models_conf())
    model = await mgr.get_language_model("m")
    assert isinstance(model, OpenAIChatCompletionsLanguageModel)


def test_litellm_cache_signature(tmp_path: Path) -> None:
    """A litellm model yields a cache signature instead of a bedrock KeyError."""
    conf = LanguageModelsConf(
        litellm_language_model_confs={
            "l": LiteLLMLanguageModelConf(model="anthropic/claude-sonnet-5")
        }
    )
    store = LLMCacheStore(str(tmp_path / "cache.db"))
    mgr = LanguageModelManager(conf, cache_store=store)
    signature = mgr._language_model_signature("l")
    assert "litellm" in signature
    assert "anthropic/claude-sonnet-5" in signature
