"""Unit tests for the CachingLanguageModel decorator."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from memmachine_server.common.cache.llm_cache_store import LLMCacheStore
from memmachine_server.common.language_model.caching_language_model import (
    CachingLanguageModel,
)
from memmachine_server.common.language_model.language_model import LanguageModel


class _Parsed(BaseModel):
    value: str


class FakeLanguageModel(LanguageModel):
    """Counts calls and returns canned values; can sleep to simulate latency."""

    def __init__(self, sleep_s: float = 0.0, parsed_result: Any = None) -> None:
        self.response_calls = 0
        self.token_calls = 0
        self.parsed_calls = 0
        self._sleep_s = sleep_s
        self._parsed_result = parsed_result

    async def generate_response(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any]:
        self.response_calls += 1
        if self._sleep_s:
            await asyncio.sleep(self._sleep_s)
        return f"resp:{user_prompt}", [{"call_id": "1"}]

    async def generate_response_with_token_usage(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any, int, int]:
        self.token_calls += 1
        return f"resp:{user_prompt}", [{"call_id": "1"}], 11, 22

    async def generate_parsed_response(
        self,
        output_format: type,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        max_attempts: int = 1,
    ) -> Any:
        self.parsed_calls += 1
        return self._parsed_result


def _store(tmp_path: Path, emulate: bool = False) -> LLMCacheStore:
    return LLMCacheStore(str(tmp_path / "cache.db"), emulate_latency=emulate)


@pytest.mark.asyncio
async def test_generate_response_is_cached(tmp_path: Path) -> None:
    """Identical requests only hit the inner model once."""
    inner = FakeLanguageModel()
    model = CachingLanguageModel(inner, "sig", _store(tmp_path))

    first = await model.generate_response(user_prompt="hello")
    second = await model.generate_response(user_prompt="hello")

    assert first == second == ("resp:hello", [{"call_id": "1"}])
    assert inner.response_calls == 1
    await model._store.close()


@pytest.mark.asyncio
async def test_different_prompt_misses(tmp_path: Path) -> None:
    """A different prompt is a cache miss."""
    inner = FakeLanguageModel()
    model = CachingLanguageModel(inner, "sig", _store(tmp_path))

    await model.generate_response(user_prompt="a")
    await model.generate_response(user_prompt="b")

    assert inner.response_calls == 2
    await model._store.close()


@pytest.mark.asyncio
async def test_different_signature_misses(tmp_path: Path) -> None:
    """The same prompt under a different model signature is a miss."""
    store = _store(tmp_path)
    inner1 = FakeLanguageModel()
    inner2 = FakeLanguageModel()
    await CachingLanguageModel(inner1, "sigA", store).generate_response(user_prompt="x")
    await CachingLanguageModel(inner2, "sigB", store).generate_response(user_prompt="x")

    assert inner1.response_calls == 1
    assert inner2.response_calls == 1
    await store.close()


@pytest.mark.asyncio
async def test_token_usage_returns_stored_counts(tmp_path: Path) -> None:
    """The token method caches and returns the original token counts."""
    inner = FakeLanguageModel()
    model = CachingLanguageModel(inner, "sig", _store(tmp_path))

    first = await model.generate_response_with_token_usage(user_prompt="hello")
    second = await model.generate_response_with_token_usage(user_prompt="hello")

    assert first == ("resp:hello", [{"call_id": "1"}], 11, 22)
    assert second == first
    assert inner.token_calls == 1
    await model._store.close()


@pytest.mark.asyncio
async def test_parsed_response_round_trips(tmp_path: Path) -> None:
    """Parsed Pydantic results are cached and reconstructed."""
    inner = FakeLanguageModel(parsed_result=_Parsed(value="v"))
    model = CachingLanguageModel(inner, "sig", _store(tmp_path))

    first = await model.generate_parsed_response(_Parsed, user_prompt="p")
    second = await model.generate_parsed_response(_Parsed, user_prompt="p")

    assert first == second == _Parsed(value="v")
    assert inner.parsed_calls == 1
    await model._store.close()


@pytest.mark.asyncio
async def test_parsed_none_not_cached(tmp_path: Path) -> None:
    """A None parsed result is not cached and is retried."""
    inner = FakeLanguageModel(parsed_result=None)
    model = CachingLanguageModel(inner, "sig", _store(tmp_path))

    await model.generate_parsed_response(_Parsed, user_prompt="p")
    await model.generate_parsed_response(_Parsed, user_prompt="p")

    assert inner.parsed_calls == 2
    await model._store.close()


@pytest.mark.asyncio
async def test_emulate_latency_on_hit(tmp_path: Path) -> None:
    """With emulation on, a hit sleeps roughly the recorded latency."""
    inner = FakeLanguageModel(sleep_s=0.2)
    model = CachingLanguageModel(inner, "sig", _store(tmp_path, emulate=True))

    await model.generate_response(user_prompt="hello")  # miss, records ~200ms

    start = asyncio.get_event_loop().time()
    await model.generate_response(user_prompt="hello")  # hit, should sleep
    elapsed = asyncio.get_event_loop().time() - start

    assert inner.response_calls == 1
    assert elapsed >= 0.15
    await model._store.close()


@pytest.mark.asyncio
async def test_no_emulation_hit_is_fast(tmp_path: Path) -> None:
    """With emulation off, a hit returns near-instantly."""
    inner = FakeLanguageModel(sleep_s=0.2)
    model = CachingLanguageModel(inner, "sig", _store(tmp_path, emulate=False))

    await model.generate_response(user_prompt="hello")

    start = asyncio.get_event_loop().time()
    await model.generate_response(user_prompt="hello")
    elapsed = asyncio.get_event_loop().time() - start

    assert elapsed < 0.1
    await model._store.close()
