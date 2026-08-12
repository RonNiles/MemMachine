"""Unit tests for the LLMCacheStore SQLite cache."""

import asyncio
from pathlib import Path

import pytest

from memmachine_server.common.cache.llm_cache_store import LLMCacheStore


def _db_path(tmp_path: Path) -> str:
    return str(tmp_path / "cache.db")


@pytest.mark.asyncio
async def test_llm_round_trip(tmp_path: Path) -> None:
    """An LLM row can be written and read back including latency_ms."""
    store = LLMCacheStore(_db_path(tmp_path))
    await store.put_llm(
        "k1",
        sig="sig",
        method="generate_response",
        request_json="{}",
        response_json='{"text": "hi"}',
        input_tokens=3,
        output_tokens=5,
        latency_ms=42.0,
    )
    row = await store.get_llm("k1")
    assert row is not None
    assert row["response_json"] == '{"text": "hi"}'
    assert row["input_tokens"] == 3
    assert row["output_tokens"] == 5
    assert row["latency_ms"] == 42.0
    assert await store.get_llm("missing") is None
    await store.close()


@pytest.mark.asyncio
async def test_llm_insert_or_ignore(tmp_path: Path) -> None:
    """A second write to the same key does not overwrite the first."""
    store = LLMCacheStore(_db_path(tmp_path))
    await store.put_llm(
        "k1",
        sig="sig",
        method="generate_response",
        request_json="{}",
        response_json='{"text": "first"}',
        input_tokens=1,
        output_tokens=1,
        latency_ms=1.0,
    )
    await store.put_llm(
        "k1",
        sig="sig",
        method="generate_response",
        request_json="{}",
        response_json='{"text": "second"}',
        input_tokens=9,
        output_tokens=9,
        latency_ms=9.0,
    )
    row = await store.get_llm("k1")
    assert row is not None
    assert row["response_json"] == '{"text": "first"}'
    assert row["input_tokens"] == 1
    await store.close()


@pytest.mark.asyncio
async def test_embedding_round_trip_and_partial_hits(tmp_path: Path) -> None:
    """Embeddings round-trip as float32 and only present keys are returned."""
    store = LLMCacheStore(_db_path(tmp_path))
    vec = [0.1, 0.2, 0.3]
    await store.put_embeddings(
        [
            {
                "cache_key": "e1",
                "sig": "sig",
                "mode": "ingest",
                "input_text": "hello",
                "embedding": vec,
                "latency_ms": 7.5,
            }
        ]
    )
    result = await store.get_embeddings(["e1", "e2"])
    assert "e2" not in result
    assert result["e1"]["latency_ms"] == 7.5
    got = result["e1"]["embedding"]
    assert len(got) == 3
    for expected, actual in zip(vec, got, strict=True):
        assert abs(expected - actual) < 1e-6
    assert await store.get_embeddings([]) == {}
    await store.close()


@pytest.mark.asyncio
async def test_concurrent_writes(tmp_path: Path) -> None:
    """Many concurrent writers do not raise (WAL + busy_timeout)."""
    store = LLMCacheStore(_db_path(tmp_path))

    async def write(i: int) -> None:
        await store.put_llm(
            f"k{i}",
            sig="sig",
            method="generate_response",
            request_json="{}",
            response_json=f'{{"text": "{i}"}}',
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
        )

    await asyncio.gather(*[write(i) for i in range(50)])
    row = await store.get_llm("k49")
    assert row is not None
    await store.close()


@pytest.mark.asyncio
async def test_persistence_across_reopen(tmp_path: Path) -> None:
    """Reopening the same file sees previously written entries."""
    path = _db_path(tmp_path)
    store = LLMCacheStore(path)
    await store.put_llm(
        "k1",
        sig="sig",
        method="generate_response",
        request_json="{}",
        response_json='{"text": "persisted"}',
        input_tokens=0,
        output_tokens=0,
        latency_ms=0.0,
    )
    await store.close()

    reopened = LLMCacheStore(path)
    row = await reopened.get_llm("k1")
    assert row is not None
    assert row["response_json"] == '{"text": "persisted"}'
    await reopened.close()


def test_make_key_is_stable() -> None:
    """make_key is order-independent and deterministic."""
    k1 = LLMCacheStore.make_key({"a": 1, "b": 2})
    k2 = LLMCacheStore.make_key({"b": 2, "a": 1})
    assert k1 == k2
    assert k1 != LLMCacheStore.make_key({"a": 1, "b": 3})


def test_model_signature_drops_none() -> None:
    """model_signature omits None-valued fields."""
    sig1 = LLMCacheStore.model_signature("openai", model="m", base_url=None)
    sig2 = LLMCacheStore.model_signature("openai", model="m")
    assert sig1 == sig2
