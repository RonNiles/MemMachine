import argparse
import asyncio
import json
import os
import time

import neo4j
from dotenv import load_dotenv
from llm_cache_util import cached_chat_completion
from longmemeval_models import (
    LongMemEvalItem,
    get_datetime_from_timestamp,
    iter_longmemeval_dataset,
)
from memmachine_server.common.cache.llm_cache_store import LLMCacheStore
from memmachine_server.common.embedder.caching_embedder import CachingEmbedder
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.episode_store.episode_model import episodes_to_string
from memmachine_server.common.reranker.identity_reranker import IdentityReranker
from memmachine_server.common.utils import async_with
from memmachine_server.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
)
from memmachine_server.episodic_memory.long_term_memory.long_term_memory import (
    DeclarativeBackendParams,
    LongTermMemory,
)
from openai import AsyncOpenAI

# Parts of prompt borrowed from Mastra's OM.
# https://github.com/mastra-ai/mastra/blob/977b49e23d8b050a2c6a6a91c0aa38b28d6388ee/packages/memory/src/processors/observational-memory/observational-memory.ts#L312-L318
ANSWER_PROMPT = """
You are a helpful assistant with access to extensive conversation history.
When answering questions, carefully review the conversation history to identify and use any relevant user preferences, interests, or specific details they have mentioned.

<history>
{memories}
</history>

IMPORTANT: When responding, reference specific details from these observations. Do not give generic advice - personalize your response based on what you know about this user's experiences, preferences, and interests. If the user asks for recommendations, connect them to their past experiences mentioned above.

KNOWLEDGE UPDATES: When asked about current state (e.g., "where do I currently...", "what is my current..."), always prefer the MOST RECENT information. Observations include dates - if you see conflicting information, the newer observation supersedes the older one. Look for phrases like "will start", "is switching", "changed to", "moved to" as indicators that previous information has been updated.

PLANNED ACTIONS: If the user stated they planned to do something (e.g., "I'm going to...", "I'm looking forward to...", "I will...") and the date they planned to do it is now in the past (check the relative time like "3 weeks ago"), assume they completed the action unless there's evidence they didn't. For example, if someone said "I'll start my new diet on Monday" and that was 2 weeks ago, assume they started the diet.

Current date: {question_timestamp}
Question: {question}
"""


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path", required=True, help="Path to the source data file"
    )
    parser.add_argument(
        "--target-path", required=True, help="Path to the target data file"
    )
    parser.add_argument(
        "--use-fts",
        action="store_true",
        help="Enable hybrid Vector + FTS (RRF) retrieval instead of vector-only",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Query only the FIRST N questions (all one question_type; smoke "
        "tests only). Must match the --limit used at ingest time.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Query N evenly-spaced questions spanning all question types. Must "
        "match the --sample used at ingest time.",
    )
    parser.add_argument(
        "--embedding-cache",
        default=None,
        help="Path to an LLM cache SQLite file (e.g. ./llm_cache.db) to cache the "
        "per-question query embeddings across runs. Omit to disable.",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Embedding model id (e.g. Qwen/Qwen3-Embedding-4B). Must match the "
        "value used at ingest time.",
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=1536,
        help="Embedding dimensionality (e.g. 2560 for Qwen3-Embedding-4B). Must "
        "match the value used at ingest time.",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=None,
        help="OpenAI-compatible base URL for the embedding provider (e.g. "
        "https://api.deepinfra.com/v1/openai). Omit to use OpenAI. The API key "
        "comes from EMBEDDING_API_KEY, falling back to OPENAI_API_KEY.",
    )
    args = parser.parse_args()

    data_path = args.data_path
    target_path = args.target_path

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
            max_concurrent_transactions=1000,
        )
    )

    openai_client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        # Answer generation over 100-episode prompts is token-heavy and can
        # burst past the model's TPM limit; let the SDK back off and retry
        # (it honors the 429 Retry-After) instead of crashing the whole run,
        # which would discard all results (they are only written at the end).
        max_retries=10,
    )

    # Embeddings may come from a different OpenAI-compatible provider than the
    # answer-generation model (e.g. a hosted Qwen embedder), so give it its own
    # client and key (EMBEDDING_API_KEY, falling back to OPENAI_API_KEY). It
    # must match the embedder used at ingest so queries land in the same space.
    embedding_client = AsyncOpenAI(
        api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=args.embedding_base_url,
    )

    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=embedding_client,
            model=args.embedding_model,
            dimensions=args.embedding_dimensions,
        )
    )

    # Optionally cache query embeddings so re-runs (and the vector→FTS pair)
    # reuse them instead of re-calling the provider. Query embeddings are keyed
    # under a distinct "search" mode, so they never collide with ingest's
    # cached derivative embeddings even in the same cache file.
    cache_store: LLMCacheStore | None = None
    if args.embedding_cache:
        cache_store = LLMCacheStore(args.embedding_cache)
        await cache_store.startup()
        signature = LLMCacheStore.model_signature(
            provider="openai",
            model=args.embedding_model,
            dimensions=args.embedding_dimensions,
        )
        embedder = CachingEmbedder(embedder, signature, cache_store)
        print(f"Embedding cache enabled: {args.embedding_cache}", flush=True)

    # "No reranker": IdentityReranker preserves retrieval order without reordering.
    reranker = IdentityReranker()

    async def qa_eval(
        memories,
        question_timestamp,
        question: str,
        model: str = "gpt-5-mini",
    ):
        messages = [
            {
                "role": "user",
                "content": ANSWER_PROMPT.format(
                    memories=memories,
                    question_timestamp=question_timestamp,
                    question=question,
                ),
            },
        ]
        # Served from the shared cache (same --embedding-cache .db) when set, so
        # a re-run or crash-resume reuses answers instead of re-billing.
        result = await cached_chat_completion(
            cache_store, openai_client, model=model, messages=messages
        )
        return {
            "response": result["content"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "total_tokens": result["input_tokens"] + result["output_tokens"],
            "latency": result["latency"],
        }

    async def process_question(
        question: LongMemEvalItem,
    ):
        group_id = question.question_id

        long_term_memory = LongTermMemory(
            DeclarativeBackendParams(
                session_id=group_id,
                vector_graph_store=vector_graph_store,
                embedder=embedder,
                reranker=reranker,
                message_sentence_chunking=True,
            )
        )

        search_query = f"User: {question.question}"

        total_start = time.monotonic()
        memory_start = time.monotonic()
        # use_fts=False -> vector-only; use_fts=True -> Vector + FTS fused via RRF.
        scored = await long_term_memory.search_scored(
            query=search_query,
            num_episodes_limit=100,
            expand_context=0,
            use_fts=args.use_fts,
        )
        memory_end = time.monotonic()
        memory_latency = memory_end - memory_start

        formatted_context = episodes_to_string([episode for _, episode in scored])

        response = await qa_eval(
            formatted_context,
            get_datetime_from_timestamp(question.question_date).strftime(
                "%A, %B %d, %Y at %I:%M %p"
            ),
            question.question,
        )
        total_end = time.monotonic()
        total_latency = total_end - total_start

        print(
            f"Question ID: {question.question_id}\n"
            f"Question: {question.question}\n"
            f"Question Date: {question.question_date}\n"
            f"Question Type: {question.question_type}\n"
            f"Answer: {question.answer}\n"
            f"Response: {response['response']}\n"
            f"Memory retrieval time: {memory_latency:.2f} seconds\n"
            f"LLM response time: {response['latency']:.2f} seconds\n"
            f"Total processing time: {total_latency:.2f} seconds\n"
            f"MEMORIES_START\n{formatted_context}MEMORIES_END\n"
        )

        return {
            "question_id": question.question_id,
            "question_date": question.question_date,
            "question": question.question,
            "answer": question.answer,
            "response": response["response"],
            "question_type": question.question_type.value,
            "abstention": question.abstention_question,
            "total_latency": total_latency,
            "memory_latency": memory_latency,
            "llm_latency": response["latency"],
            "episodes_text": formatted_context,
        }

    semaphore = asyncio.Semaphore(5)

    # Stream questions so the multi-GB dataset is never fully resident. Keep a
    # bounded window of in-flight searches; collect results as they finish.
    # Order-independent: each result carries its own question_id/answer, and
    # evaluation scores results item-by-item.
    max_outstanding = 10
    results = []
    in_flight: set = set()
    for question in iter_longmemeval_dataset(
        data_path, limit=args.limit, sample=args.sample
    ):
        in_flight.add(
            asyncio.create_task(async_with(semaphore, process_question(question)))
        )
        if len(in_flight) >= max_outstanding:
            done, in_flight = await asyncio.wait(
                in_flight, return_when=asyncio.FIRST_COMPLETED
            )
            results.extend(task.result() for task in done)
    if in_flight:
        done, _ = await asyncio.wait(in_flight)
        results.extend(task.result() for task in done)

    with open(target_path, "w") as f:
        json.dump(results, f, indent=4)

    if cache_store is not None:
        await cache_store.close()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
