"""A LanguageModel decorator that caches responses in an LLMCacheStore."""

from __future__ import annotations

import asyncio
import json
import logging
from time import perf_counter
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from memmachine_server.common.cache.llm_cache_store import (
    LLMCacheStore,
    canonical_json,
)
from memmachine_server.common.language_model.language_model import LanguageModel

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Cache-key schema version. Bump to intentionally invalidate all keys.
_KEY_VERSION = 1


class CachingLanguageModel(LanguageModel):
    """
    Wrap a language model so identical requests are served from a cache.

    On a miss the inner model is called, its response and the wall-clock
    latency are persisted, and the response is returned. On a hit the stored
    response is returned (optionally after sleeping for the recorded latency).
    """

    def __init__(
        self, inner: LanguageModel, signature: str, store: LLMCacheStore
    ) -> None:
        """Wrap ``inner`` with a model ``signature`` and a cache ``store``."""
        self._inner = inner
        self._signature = signature
        self._store = store

    async def _maybe_emulate(self, latency_ms: float) -> None:
        """Sleep for the recorded latency when emulation is enabled."""
        if self._store.emulate_latency and latency_ms > 0:
            await asyncio.sleep(latency_ms / 1000.0)

    @staticmethod
    def _decode_text(response_json: str) -> tuple[str, Any]:
        """Decode a stored text response into (text, tool_calls)."""
        data = json.loads(response_json)
        return data["text"], data["tool_calls"]

    async def _get_or_generate_text(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, str] | None,
        max_attempts: int,
        want_tokens: bool,
    ) -> tuple[str, Any, int, int]:
        """Shared cache logic for both text-generation methods."""
        payload = {
            "v": _KEY_VERSION,
            "method": "generate_response",
            "sig": self._signature,
            "system": system_prompt,
            "user": user_prompt,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        key = LLMCacheStore.make_key(payload)

        row = await self._store.get_llm(key)
        if row is not None:
            logger.debug("LLM cache hit (generate_response, key=%.12s).", key)
            await self._maybe_emulate(row["latency_ms"])
            text, tool_calls = self._decode_text(row["response_json"])
            return text, tool_calls, row["input_tokens"], row["output_tokens"]

        logger.debug("LLM cache miss (generate_response, key=%.12s).", key)
        start = perf_counter()
        if want_tokens:
            (
                text,
                tool_calls,
                input_tokens,
                output_tokens,
            ) = await self._inner.generate_response_with_token_usage(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=tools,
                tool_choice=tool_choice,
                max_attempts=max_attempts,
            )
        else:
            text, tool_calls = await self._inner.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=tools,
                tool_choice=tool_choice,
                max_attempts=max_attempts,
            )
            input_tokens, output_tokens = 0, 0
        latency_ms = (perf_counter() - start) * 1000.0

        await self._store.put_llm(
            key,
            sig=self._signature,
            method="generate_response",
            request_json=canonical_json(payload),
            response_json=json.dumps({"text": text, "tool_calls": tool_calls}),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )
        return text, tool_calls, input_tokens, output_tokens

    async def generate_response(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any]:
        """Return a cached response or generate and cache a new one."""
        text, tool_calls, _, _ = await self._get_or_generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=tools,
            tool_choice=tool_choice,
            max_attempts=max_attempts,
            want_tokens=False,
        )
        return text, tool_calls

    async def generate_response_with_token_usage(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any, int, int]:
        """Return a cached response with token usage, or generate and cache."""
        return await self._get_or_generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=tools,
            tool_choice=tool_choice,
            max_attempts=max_attempts,
            want_tokens=True,
        )

    async def generate_parsed_response(
        self,
        output_format: type[T],
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        max_attempts: int = 1,
    ) -> T | None:
        """Return a cached parsed response or generate and cache a new one."""
        # Only Pydantic models can be (de)serialized for the cache; bypass
        # otherwise so behavior is identical to the inner model.
        if not (
            isinstance(output_format, type) and issubclass(output_format, BaseModel)
        ):
            return await self._inner.generate_parsed_response(
                output_format=output_format,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_attempts=max_attempts,
            )

        payload = {
            "v": _KEY_VERSION,
            "method": "generate_parsed_response",
            "sig": self._signature,
            "system": system_prompt,
            "user": user_prompt,
            "output_format": {
                "name": output_format.__name__,
                "schema": output_format.model_json_schema(),
            },
        }
        key = LLMCacheStore.make_key(payload)

        row = await self._store.get_llm(key)
        if row is not None:
            logger.debug("LLM cache hit (generate_parsed_response, key=%.12s).", key)
            await self._maybe_emulate(row["latency_ms"])
            return cast("T", output_format.model_validate_json(row["response_json"]))

        logger.debug("LLM cache miss (generate_parsed_response, key=%.12s).", key)
        start = perf_counter()
        (
            result,
            input_tokens,
            output_tokens,
        ) = await self._inner.generate_parsed_response_with_token_usage(
            output_format=output_format,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_attempts=max_attempts,
        )
        latency_ms = (perf_counter() - start) * 1000.0

        # Do not cache None — it signals a parse/refusal failure to retry.
        if result is None:
            return None

        # output_format is guaranteed a BaseModel subclass here (guarded above),
        # so a non-None result is always a BaseModel instance.
        await self._store.put_llm(
            key,
            sig=self._signature,
            method="generate_parsed_response",
            request_json=canonical_json(payload),
            response_json=cast("BaseModel", result).model_dump_json(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )
        return cast("T", result)
