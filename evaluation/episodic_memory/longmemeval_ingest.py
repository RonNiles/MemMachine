import argparse
import asyncio
import os
from datetime import datetime
from uuid import uuid4

import neo4j
import openai
from dotenv import load_dotenv
from longmemeval_models import (
    LongMemEvalItem,
    iter_longmemeval_dataset,
)
from memmachine_server.common.cache.llm_cache_store import LLMCacheStore
from memmachine_server.common.embedder.caching_embedder import CachingEmbedder
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.reranker.identity_reranker import IdentityReranker
from memmachine_server.common.utils import async_with
from memmachine_server.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
)
from memmachine_server.episodic_memory.declarative_memory import (
    ContentType,
    DeclarativeMemory,
    DeclarativeMemoryParams,
    Episode,
)


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", required=True, help="Path to the data file")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ingest only the first N questions (subset for faster A/B runs). "
        "Use the same --limit for the search step.",
    )
    parser.add_argument(
        "--session-concurrency",
        type=int,
        default=4,
        help="Max sessions embedded/written concurrently per question "
        "(lower = less peak memory)",
    )
    parser.add_argument(
        "--no-sentence-chunking",
        action="store_true",
        help="Disable per-sentence chunking (~5x less memory/disk, ~1-2%% lower scores)",
    )
    parser.add_argument(
        "--embedding-cache",
        default=None,
        help="Path to an LLM cache SQLite file (e.g. ../../llm_cache.db) to cache "
        "embeddings across runs. Omit to disable. Re-ingesting the same content "
        "then skips the embedding provider calls (cache hits).",
    )

    args = parser.parse_args()

    data_path = args.data_path

    neo4j_driver = neo4j.AsyncGraphDatabase.driver(
        uri=os.getenv("NEO4J_URI"),
        auth=(
            os.getenv("NEO4J_USERNAME"),
            os.getenv("NEO4J_PASSWORD"),
        ),
    )

    vector_graph_store = Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=neo4j_driver,
            range_index_creation_threshold=10000,
            vector_index_creation_threshold=10000,
        )
    )

    openai_client = openai.AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client,
            model="text-embedding-3-small",
            dimensions=1536,
            max_input_length=2048,
        )
    )

    # Optionally wrap the embedder with the persistent LLM cache so identical
    # inputs are served from disk instead of re-calling the provider. The
    # signature mirrors this embedder's config, so cache hits return vectors
    # computed the same way; to share with a running server's cache, its
    # `openai_embedder` config must match these fields exactly.
    cache_store: LLMCacheStore | None = None
    if args.embedding_cache:
        cache_store = LLMCacheStore(args.embedding_cache)
        await cache_store.startup()
        signature = LLMCacheStore.model_signature(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=1536,
            max_input_length=2048,
        )
        embedder = CachingEmbedder(embedder, signature, cache_store)
        print(f"Embedding cache enabled: {args.embedding_cache}", flush=True)

    # "No reranker": IdentityReranker preserves order (declarative memory still
    # requires a reranker object; this one does no reordering).
    reranker = IdentityReranker()

    async def process_conversation(question: LongMemEvalItem):
        group_id = question.question_id
        session_ids = list(question.session_id_map.keys())

        memory = DeclarativeMemory(
            DeclarativeMemoryParams(
                session_id=group_id,
                vector_graph_store=vector_graph_store,
                embedder=embedder,
                reranker=reranker,
                message_sentence_chunking=not args.no_sentence_chunking,
            )
        )

        session_tasks = []
        for session_id in session_ids:
            session = question.get_session(session_id)

            episodes = []
            for turn in session:
                timestamp = datetime.fromisoformat(turn.timestamp)
                episodes.append(
                    Episode(
                        uid=str(uuid4()),
                        timestamp=timestamp,
                        source="Assistant" if turn.role == "assistant" else "User",
                        content_type=ContentType.MESSAGE,
                        content=turn.content.strip(),
                        user_metadata={
                            "longmemeval_session_id": session_id,
                            "has_answer": turn.has_answer,
                            "turn_id": turn.index,
                        },
                    )
                )

            # Bound concurrency so only N sessions' chunks+embeddings are in
            # flight at once, instead of the whole haystack simultaneously.
            session_tasks.append(
                async_with(session_semaphore, memory.add_episodes(episodes=episodes))
            )

        await asyncio.gather(*session_tasks)

    session_semaphore = asyncio.Semaphore(args.session_concurrency)

    # Stream one question at a time so the multi-GB dataset is never fully
    # resident (json.load on the ~2.6 GB M split alone exhausts RAM). Questions
    # are ingested sequentially; the session semaphore bounds concurrency
    # within each question.
    count = 0
    for question in iter_longmemeval_dataset(data_path, limit=args.limit):
        await process_conversation(question)
        count += 1
        print(f"ingested {count} questions (last: {question.question_id})", flush=True)
    print(f"Done: {count} questions ingested")

    if cache_store is not None:
        await cache_store.close()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
