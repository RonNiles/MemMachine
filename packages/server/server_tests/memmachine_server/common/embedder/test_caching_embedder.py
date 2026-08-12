"""Unit tests for the CachingEmbedder decorator."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from memmachine_server.common.cache.llm_cache_store import LLMCacheStore
from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.embedder.caching_embedder import CachingEmbedder
from memmachine_server.common.embedder.embedder import Embedder


class FakeEmbedder(Embedder):
    """Records the inputs sent to it and returns deterministic vectors."""

    def __init__(self, sleep_s: float = 0.0) -> None:
        super().__init__(batch_size=None)
        self.ingest_inputs: list[list[Any]] = []
        self.search_inputs: list[list[Any]] = []
        self._sleep_s = sleep_s

    @staticmethod
    def _vec(text: Any) -> list[float]:
        return [float(len(str(text))), 1.0]

    async def _ingest_embed(
        self, inputs: list[Any], max_attempts: int = 1
    ) -> list[list[float]]:
        self.ingest_inputs.append(inputs)
        if self._sleep_s:
            await asyncio.sleep(self._sleep_s)
        return [self._vec(t) for t in inputs]

    async def _search_embed(
        self, queries: list[Any], max_attempts: int = 1
    ) -> list[list[float]]:
        self.search_inputs.append(queries)
        return [self._vec(t) for t in queries]

    @property
    def model_id(self) -> str:
        return "fake-model"

    @property
    def dimensions(self) -> int:
        return 2

    @property
    def similarity_metric(self) -> SimilarityMetric:
        return SimilarityMetric.COSINE


def _store(tmp_path: Path, emulate: bool = False) -> LLMCacheStore:
    return LLMCacheStore(str(tmp_path / "cache.db"), emulate_latency=emulate)


@pytest.mark.asyncio
async def test_only_misses_sent_to_inner(tmp_path: Path) -> None:
    """A second call sends only previously-unseen inputs to the inner embedder."""
    inner = FakeEmbedder()
    emb = CachingEmbedder(inner, "sig", _store(tmp_path))

    first = await emb.ingest_embed(["a", "bb"])
    second = await emb.ingest_embed(["a", "bb", "ccc"])

    assert first == [[1.0, 1.0], [2.0, 1.0]]
    assert second == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    # First inner call embeds {a, bb}; second embeds only {ccc}.
    assert inner.ingest_inputs[0] == ["a", "bb"]
    assert inner.ingest_inputs[1] == ["ccc"]
    await emb._store.close()


@pytest.mark.asyncio
async def test_order_and_duplicates_preserved(tmp_path: Path) -> None:
    """Output order matches input order, including duplicates."""
    inner = FakeEmbedder()
    emb = CachingEmbedder(inner, "sig", _store(tmp_path))

    result = await emb.ingest_embed(["a", "bb", "a", "bb"])
    assert result == [[1.0, 1.0], [2.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
    # Duplicates dedupe to a single inner embedding per distinct text.
    assert inner.ingest_inputs[0] == ["a", "bb"]
    await emb._store.close()


@pytest.mark.asyncio
async def test_empty_input(tmp_path: Path) -> None:
    """An empty input list short-circuits to an empty result."""
    inner = FakeEmbedder()
    emb = CachingEmbedder(inner, "sig", _store(tmp_path))
    assert await emb.ingest_embed([]) == []
    assert inner.ingest_inputs == []
    await emb._store.close()


@pytest.mark.asyncio
async def test_ingest_and_search_keyed_separately(tmp_path: Path) -> None:
    """The same text under ingest vs search modes does not collide."""
    inner = FakeEmbedder()
    emb = CachingEmbedder(inner, "sig", _store(tmp_path))

    await emb.ingest_embed(["x"])
    await emb.search_embed(["x"])

    assert inner.ingest_inputs == [["x"]]
    assert inner.search_inputs == [["x"]]
    await emb._store.close()


@pytest.mark.asyncio
async def test_properties_delegate(tmp_path: Path) -> None:
    """Identity properties delegate to the inner embedder."""
    inner = FakeEmbedder()
    emb = CachingEmbedder(inner, "sig", _store(tmp_path))
    assert emb.model_id == "fake-model"
    assert emb.dimensions == 2
    assert emb.similarity_metric == SimilarityMetric.COSINE
    await emb._store.close()


@pytest.mark.asyncio
async def test_emulate_latency_on_all_hit_call(tmp_path: Path) -> None:
    """An all-hit call sleeps roughly the summed per-input latency shares."""
    inner = FakeEmbedder(sleep_s=0.2)
    emb = CachingEmbedder(inner, "sig", _store(tmp_path, emulate=True))

    await emb.ingest_embed(["a", "bb"])  # miss: one ~200ms inner call for 2 inputs

    start = asyncio.get_event_loop().time()
    await emb.ingest_embed(["a", "bb"])  # all hits -> emulate ~sum of shares (~200ms)
    elapsed = asyncio.get_event_loop().time() - start

    assert inner.ingest_inputs == [["a", "bb"]]  # no second inner call
    assert elapsed >= 0.12
    await emb._store.close()
