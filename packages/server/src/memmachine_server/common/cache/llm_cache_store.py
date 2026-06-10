"""
SQLite-backed persistent cache for LLM responses and embeddings.

A single portable ``.db`` file holds two tables: ``llm_cache`` (one row per
unique language-model request) and ``embedding_cache`` (one row per unique
embedded input). Entries are keyed by a SHA-256 of a canonical JSON request
payload that includes a model signature, so identical requests across runs
hit the same row. Both tables record ``latency_ms`` — the wall-clock duration
of the original API call — so reruns can optionally emulate the original
latency.

The store is reconstructible (it is a cache, not a source of truth), so it
runs in WAL mode with ``synchronous=NORMAL`` for throughput under the
concurrent writes of a large batch job, and writes are idempotent
(``INSERT OR IGNORE``).
"""

from __future__ import annotations

import hashlib
import json
import logging
from asyncio import Lock
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import (
    BigInteger,
    Column,
    Float,
    LargeBinary,
    MetaData,
    String,
    Table,
    event,
    select,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import ConnectionPoolEntry

logger = logging.getLogger(__name__)

# SQLite limits the number of bound parameters per statement; chunk IN-clauses
# well below the conservative historical default of 999.
_SELECT_CHUNK_SIZE = 500


def canonical_json(obj: object) -> str:
    """Serialize an object to a stable, compact canonical JSON string."""
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


class LLMCacheStore:
    """Persistent SQLite cache for language-model responses and embeddings."""

    def __init__(self, path: str, emulate_latency: bool = False) -> None:
        """
        Initialize the cache store.

        Args:
            path: Filesystem path to the SQLite cache file.
            emulate_latency: When true, callers should sleep for the recorded
                ``latency_ms`` before returning a cached value.

        """
        self.emulate_latency = emulate_latency
        self._resolved_path = str(Path(path).resolve())
        self._engine = create_async_engine(f"sqlite+aiosqlite:///{path}")

        @event.listens_for(self._engine.sync_engine, "connect")
        def _configure_sqlite(
            dbapi_connection: DBAPIConnection,
            _connection_record: ConnectionPoolEntry,
        ) -> None:
            cursor = dbapi_connection.cursor()
            # WAL allows concurrent readers with a single writer; busy_timeout
            # makes concurrent tasks block-and-retry instead of raising
            # SQLITE_BUSY; NORMAL is safe under WAL for a reconstructible cache.
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

        self._metadata = MetaData()
        self._llm_table = Table(
            "llm_cache",
            self._metadata,
            Column("cache_key", String, primary_key=True),
            Column("sig", String, nullable=False),
            Column("method", String, nullable=False),
            Column("request_json", String, nullable=False),
            Column("response_json", String, nullable=False),
            Column("input_tokens", BigInteger, nullable=False, default=0),
            Column("output_tokens", BigInteger, nullable=False, default=0),
            Column("latency_ms", Float, nullable=False, default=0.0),
            Column("created_at", String, nullable=False),
        )
        self._embedding_table = Table(
            "embedding_cache",
            self._metadata,
            Column("cache_key", String, primary_key=True),
            Column("sig", String, nullable=False),
            Column("mode", String, nullable=False),
            Column("input_text", String, nullable=False),
            Column("embedding", LargeBinary, nullable=False),
            Column("latency_ms", Float, nullable=False, default=0.0),
            Column("created_at", String, nullable=False),
        )

        self._started = False
        self._startup_lock = Lock()

    async def startup(self) -> None:
        """Create tables if they do not exist. Idempotent and concurrency-safe."""
        if self._started:
            return
        async with self._startup_lock:
            if self._started:
                return
            async with self._engine.begin() as connection:
                await connection.run_sync(self._metadata.create_all)
            self._started = True
            logger.info(
                "LLM cache ready at '%s' (emulate_latency=%s).",
                self._resolved_path,
                self.emulate_latency,
            )

    async def close(self) -> None:
        """Checkpoint the WAL into the main file and dispose the engine."""
        try:
            async with self._engine.begin() as connection:
                await connection.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            logger.warning(
                "Failed to checkpoint LLM cache WAL on close.", exc_info=True
            )
        await self._engine.dispose()

    # -- signature / key helpers ------------------------------------------

    @staticmethod
    def model_signature(provider: str, **fields: object) -> str:
        """
        Build a stable model signature string from output-affecting fields.

        ``None`` fields are dropped. Credentials must never be passed in.
        """
        data: dict[str, object] = {"provider": provider}
        data.update({k: v for k, v in fields.items() if v is not None})
        return canonical_json(data)

    @staticmethod
    def make_key(payload: dict[str, Any]) -> str:
        """Return a SHA-256 hex digest of a canonical request payload."""
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    # -- LLM cache --------------------------------------------------------

    async def get_llm(self, cache_key: str) -> dict[str, Any] | None:
        """Return the cached LLM row for a key, or None on a miss."""
        await self.startup()
        t = self._llm_table
        stmt = select(
            t.c.response_json,
            t.c.input_tokens,
            t.c.output_tokens,
            t.c.latency_ms,
        ).where(t.c.cache_key == cache_key)
        async with self._engine.connect() as connection:
            row = (await connection.execute(stmt)).first()
        if row is None:
            return None
        return {
            "response_json": row.response_json,
            "input_tokens": row.input_tokens,
            "output_tokens": row.output_tokens,
            "latency_ms": row.latency_ms,
        }

    async def put_llm(
        self,
        cache_key: str,
        *,
        sig: str,
        method: str,
        request_json: str,
        response_json: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> None:
        """Persist an LLM response. Existing rows are left untouched."""
        await self.startup()
        stmt = sqlite_insert(self._llm_table).values(
            cache_key=cache_key,
            sig=sig,
            method=method,
            request_json=request_json,
            response_json=response_json,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            created_at=datetime.now(UTC).isoformat(),
        )
        stmt = stmt.on_conflict_do_nothing(index_elements=["cache_key"])
        async with self._engine.begin() as connection:
            await connection.execute(stmt)

    # -- embedding cache --------------------------------------------------

    async def get_embeddings(self, cache_keys: list[str]) -> dict[str, dict[str, Any]]:
        """
        Return cached embeddings for the given keys.

        Result maps cache_key -> {"embedding": list[float], "latency_ms": float}.
        Missing keys are simply absent from the result.
        """
        if not cache_keys:
            return {}
        await self.startup()
        t = self._embedding_table
        results: dict[str, dict[str, Any]] = {}
        async with self._engine.connect() as connection:
            for i in range(0, len(cache_keys), _SELECT_CHUNK_SIZE):
                chunk = cache_keys[i : i + _SELECT_CHUNK_SIZE]
                stmt = select(t.c.cache_key, t.c.embedding, t.c.latency_ms).where(
                    t.c.cache_key.in_(chunk)
                )
                for row in (await connection.execute(stmt)).all():
                    results[row.cache_key] = {
                        "embedding": np.frombuffer(
                            row.embedding, dtype=np.float32
                        ).tolist(),
                        "latency_ms": row.latency_ms,
                    }
        return results

    async def put_embeddings(self, rows: list[dict[str, Any]]) -> None:
        """
        Persist embeddings. Existing rows are left untouched.

        Each row must contain: cache_key, sig, mode, input_text, embedding
        (list[float]), latency_ms.
        """
        if not rows:
            return
        await self.startup()
        created_at = datetime.now(UTC).isoformat()
        values = [
            {
                "cache_key": row["cache_key"],
                "sig": row["sig"],
                "mode": row["mode"],
                "input_text": row["input_text"],
                "embedding": np.asarray(row["embedding"], dtype=np.float32).tobytes(),
                "latency_ms": row["latency_ms"],
                "created_at": created_at,
            }
            for row in rows
        ]
        stmt = sqlite_insert(self._embedding_table).on_conflict_do_nothing(
            index_elements=["cache_key"]
        )
        async with self._engine.begin() as connection:
            await connection.execute(stmt, values)
