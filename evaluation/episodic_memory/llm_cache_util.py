"""Chat-completion caching backed by the shared LLMCacheStore.

Lets the LongMemEval search (answer generation) and evaluate (LLM-judge)
harnesses reuse the same on-disk cache that already holds the embeddings, so a
re-run — or resuming after a crash — replays stored generations instead of
re-billing the provider. Completions live in the store's ``llm_cache`` table,
separate from the ``embedding_cache`` table, so both share one ``.db`` file
without colliding.
"""

import json
import time

from memmachine_server.common.cache.llm_cache_store import LLMCacheStore


async def cached_chat_completion(store, client, *, model, messages, **create_kwargs):
    """Return a chat completion, served from ``store`` when present.

    The cache key covers the model signature, the messages, and any extra
    create kwargs (e.g. ``temperature``), so different prompts or params never
    collide and the vector vs. FTS runs — which build different retrieved
    context — stay distinct. ``store`` may be ``None`` to disable caching.

    Returns a normalized dict: ``content``, ``input_tokens``,
    ``output_tokens``, ``latency`` (seconds).
    """
    sig = LLMCacheStore.model_signature(provider="openai", model=model)
    payload = {
        "method": "chat.completions",
        "sig": sig,
        "messages": messages,
        "kwargs": create_kwargs,
    }
    key = LLMCacheStore.make_key(payload)

    if store is not None:
        row = await store.get_llm(key)
        if row is not None:
            return {
                "content": json.loads(row["response_json"])["content"],
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "latency": row["latency_ms"] / 1000.0,
            }

    start = time.monotonic()
    response = await client.chat.completions.create(
        model=model, messages=messages, **create_kwargs
    )
    latency = time.monotonic() - start

    content = response.choices[0].message.content.strip()
    usage = response.usage
    result = {
        "content": content,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "latency": latency,
    }

    if store is not None:
        await store.put_llm(
            key,
            sig=sig,
            method="chat.completions",
            request_json=json.dumps(payload),
            response_json=json.dumps({"content": content}),
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            latency_ms=latency * 1000.0,
        )
    return result
