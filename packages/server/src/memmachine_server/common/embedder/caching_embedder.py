"""An Embedder decorator that caches embeddings in an LLMCacheStore."""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from typing import Any

from memmachine_server.common.cache.llm_cache_store import LLMCacheStore
from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.embedder.embedder import Embedder

logger = logging.getLogger(__name__)

# Cache-key schema version. Bump to intentionally invalidate all keys.
_KEY_VERSION = 1


class CachingEmbedder(Embedder):
    """
    Wrap an embedder so identical inputs are served from a cache.

    Caching is per input: a call's already-cached inputs are served from the
    store and only the misses are sent to the inner embedder (preserving its
    batching). New embeddings and the per-input share of the call's latency
    are persisted. When the whole call is served from cache, the recorded
    latency is optionally emulated.
    """

    def __init__(self, inner: Embedder, signature: str, store: LLMCacheStore) -> None:
        """Wrap ``inner`` with a model ``signature`` and a cache ``store``."""
        super().__init__(batch_size=None)
        self._inner = inner
        self._signature = signature
        self._store = store

    @property
    def model_id(self) -> str:
        """Delegate to the wrapped embedder."""
        return self._inner.model_id

    @property
    def dimensions(self) -> int:
        """Delegate to the wrapped embedder."""
        return self._inner.dimensions

    @property
    def similarity_metric(self) -> SimilarityMetric:
        """Delegate to the wrapped embedder."""
        return self._inner.similarity_metric

    async def ingest_embed(
        self, inputs: list[Any], max_attempts: int = 1
    ) -> list[list[float]]:
        """Embed ingestion inputs, serving cache hits and caching misses."""
        return await self._cached_embed("ingest", inputs, max_attempts)

    async def search_embed(
        self, queries: list[Any], max_attempts: int = 1
    ) -> list[list[float]]:
        """Embed search queries, serving cache hits and caching misses."""
        return await self._cached_embed("search", queries, max_attempts)

    @staticmethod
    def _to_text(value: object) -> str:
        """Coerce an embedder input into a stable cache-key text."""
        return value if isinstance(value, str) else LLMCacheStore.make_key({"x": value})

    def _key(self, mode: str, text: str) -> str:
        """Build the cache key for one input."""
        return LLMCacheStore.make_key(
            {"v": _KEY_VERSION, "mode": mode, "sig": self._signature, "text": text}
        )

    async def _cached_embed(
        self, mode: str, inputs: list[Any], max_attempts: int
    ) -> list[list[float]]:
        if not inputs:
            return []

        texts = [self._to_text(inp) for inp in inputs]
        keys = [self._key(mode, text) for text in texts]

        cached = await self._store.get_embeddings(list(dict.fromkeys(keys)))
        emb_by_key: dict[str, list[float]] = {
            k: v["embedding"] for k, v in cached.items()
        }
        lat_by_key: dict[str, float] = {k: v["latency_ms"] for k, v in cached.items()}

        # Distinct misses, preserving the first text seen for each key.
        miss_text_by_key: dict[str, str] = {}
        for text, key in zip(texts, keys, strict=True):
            if key not in emb_by_key and key not in miss_text_by_key:
                miss_text_by_key[key] = text

        logger.debug(
            "Embedding cache (%s): %d hits, %d misses of %d inputs.",
            mode,
            len(inputs) - sum(1 for key in keys if key in miss_text_by_key),
            sum(1 for key in keys if key in miss_text_by_key),
            len(inputs),
        )

        if miss_text_by_key:
            await self._fill_misses(mode, miss_text_by_key, emb_by_key, max_attempts)
        elif self._store.emulate_latency:
            total_ms = sum(lat_by_key.get(key, 0.0) for key in keys)
            if total_ms > 0:
                await asyncio.sleep(total_ms / 1000.0)

        return [emb_by_key[key] for key in keys]

    async def _fill_misses(
        self,
        mode: str,
        miss_text_by_key: dict[str, str],
        emb_by_key: dict[str, list[float]],
        max_attempts: int,
    ) -> None:
        """Embed the missing inputs via the inner embedder and persist them."""
        miss_keys = list(miss_text_by_key.keys())
        miss_texts = [miss_text_by_key[key] for key in miss_keys]

        inner_method = (
            self._inner.ingest_embed if mode == "ingest" else self._inner.search_embed
        )
        start = perf_counter()
        embeddings = await inner_method(miss_texts, max_attempts)
        share = (perf_counter() - start) * 1000.0 / len(miss_texts)

        rows: list[dict[str, Any]] = []
        for key, text, embedding in zip(miss_keys, miss_texts, embeddings, strict=True):
            emb_by_key[key] = embedding
            rows.append(
                {
                    "cache_key": key,
                    "sig": self._signature,
                    "mode": mode,
                    "input_text": text,
                    "embedding": embedding,
                    "latency_ms": share,
                }
            )
        await self._store.put_embeddings(rows)

    async def _ingest_embed(
        self, inputs: list[Any], max_attempts: int = 1
    ) -> list[list[float]]:
        """Satisfy the ABC; not used since the public method is overridden."""
        return await self._inner.ingest_embed(inputs, max_attempts)

    async def _search_embed(
        self, queries: list[Any], max_attempts: int = 1
    ) -> list[list[float]]:
        """Satisfy the ABC; not used since the public method is overridden."""
        return await self._inner.search_embed(queries, max_attempts)
