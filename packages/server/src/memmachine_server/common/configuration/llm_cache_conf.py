"""Configuration for the persistent LLM/embedding cache."""

from __future__ import annotations

from pydantic import Field, model_validator

from memmachine_server.common.configuration.mixin_confs import YamlSerializableMixin


class LLMCacheConf(YamlSerializableMixin):
    """
    Configuration for the persistent LLM/embedding response cache.

    When enabled, LLM completions and embeddings are recorded to a single
    portable SQLite file keyed by a hash of the request and model signature.
    Duplicate requests (same model + same inputs) are served from the cache
    instead of calling the provider, reducing spend and latency on reruns.

    The cache is opt-in (disabled by default) and intended for repeatable
    batch jobs. Entries are permanent (no TTL); the data is static.
    """

    enabled: bool = Field(
        default=False,
        description="Whether the persistent LLM/embedding cache is enabled.",
    )
    path: str = Field(
        default="",
        description=(
            "Filesystem path to the SQLite cache file. Required when enabled. "
            "Back up or port the cache by copying this single file."
        ),
    )
    cache_llm: bool = Field(
        default=True,
        description="Whether to cache language model responses.",
    )
    cache_embeddings: bool = Field(
        default=True,
        description="Whether to cache embeddings.",
    )
    emulate_latency: bool = Field(
        default=False,
        description=(
            "When true, sleep for the recorded duration of the original API "
            "call before returning a cached value, reproducing the original "
            "timing profile. Leave false for fast warm reruns."
        ),
    )
    deterministic_ingestion: bool = Field(
        default=False,
        description=(
            "When true, ingestion-side LLM calls are made deterministic so "
            "identical reruns produce identical requests (and thus cache "
            "hits): short-term memory waits for in-flight summarization "
            "before deciding eviction batches, and semantic consolidation is "
            "checked after each message instead of per polling cycle. Trades "
            "some ingestion parallelism within a session on cold runs."
        ),
    )

    @model_validator(mode="after")
    def _require_path_when_enabled(self) -> LLMCacheConf:
        """Ensure a cache file path is configured when the cache is enabled."""
        if self.enabled and not self.path:
            raise ValueError(
                "llm_cache.path must be set when llm_cache.enabled is true"
            )
        return self
