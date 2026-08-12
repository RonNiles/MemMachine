# LLM + Embedding Cache

An opt-in, persistent cache for LLM responses and embeddings, backed by a single
portable SQLite file. On a rerun, identical requests (same model + inputs) are
served from the cache instead of calling the provider — cutting LLM spend and
latency. The cache records each original call's latency so reruns can optionally
emulate the original timing.

This was built for batch jobs that ingest 100k+ episodic memories where the
databases are re-initialized and the same job is re-run many times.

## How it works

A **decorator pattern** at the two construction chokepoints. `LanguageModelManager`
and `EmbedderManager` are the only places models/embedders are built; when caching
is enabled they wrap each built instance (after validation, so validation probes
still hit the real provider). All LLM/embedder usage — including embedder-backed
rerankers — flows through these wrappers.

Cache keys are a SHA-256 of a versioned canonical JSON request payload that includes
a **model signature** (provider + model + output-affecting params like dimensions /
base_url / inference config). Keys contain no timestamps or run state, so identical
requests across runs hit the same row. Changing the model or any output-affecting
parameter changes the signature and naturally invalidates affected entries.

## Files created

- `packages/server/src/memmachine_server/common/configuration/llm_cache_conf.py` —
  `LLMCacheConf` (`enabled`, `path`, `cache_llm`, `cache_embeddings`,
  `emulate_latency`; validator requires `path` when enabled).
- `packages/server/src/memmachine_server/common/cache/llm_cache_store.py` —
  `LLMCacheStore`: async SQLite (aiosqlite) with WAL + `busy_timeout` +
  `synchronous=NORMAL` for concurrent writes during the batch job; `llm_cache` +
  `embedding_cache` tables (each with a `latency_ms` column); idempotent
  `INSERT OR IGNORE`; embeddings stored as compact float32 blobs;
  `wal_checkpoint(TRUNCATE)` on close so the single file is self-contained for copying.
- `packages/server/src/memmachine_server/common/language_model/caching_language_model.py` —
  caches all 3 LM methods; returns **stored original token counts** on hits; never
  caches `None` parsed results; bypasses non-Pydantic output formats.
- `packages/server/src/memmachine_server/common/embedder/caching_embedder.py` —
  **per-input** caching; only misses go to the inner embedder; preserves
  order/duplicates; records per-input latency share and emulates summed latency on
  all-hit calls.

## Files edited

- `common/configuration/__init__.py` — `Configuration.llm_cache` field + `to_yaml()`.
- `common/resource_manager/resource_manager.py` — builds the store when enabled,
  `startup()` in `build()`, `close()` in `close()`.
- `common/resource_manager/embedder_manager.py` and `language_model_manager.py` —
  `cache_store` ctor param + signature helpers + wrapping.
- `configuration.yml` and `deployments/helm/templates/memmachine-configmaps.yaml` —
  documented `llm_cache:` blocks.

## Verification

- **25 new tests** pass (store round-trip/persistence/concurrency, LM caching incl.
  latency emulation on/off, embedder partial-hit/order/mode-separation, manager
  wrapping), plus the existing configuration suite confirming the YAML round-trip.
- `ruff check` clean, `ruff format` applied, `ty check` clean for the new files,
  complexipy under the limit of 10.

## Using it

Set in `configuration.yml`:

```yaml
llm_cache:
  enabled: true
  path: ./llm_cache.db
  cache_llm: true
  cache_embeddings: true
  emulate_latency: false   # true to reproduce original call timing on reruns
```

To back up / port: stop the server (checkpoints the WAL), copy the one `.db` file,
and point `llm_cache.path` at it on the new machine. The file is credential-free
(only request payloads, responses, and embeddings). Entries are permanent (no TTL).

### Path resolution

A relative `path` resolves against the **server's working directory**. Under
docker-compose that is `/app` inside the container — use
`path: /app/cache/llm_cache.db` and uncomment the `./cache` volume mount in
`docker-compose.yml` so the file persists on the host.

### Verifying it's active

The `.db` file is created eagerly at server startup (`MemMachine.start()`), and the
log shows:

- `LLM cache enabled: path='...' cache_llm=... cache_embeddings=... emulate_latency=...`
  — at resource-manager construction.
- `LLM cache ready at '/abs/path/llm_cache.db' (...)` — when the SQLite file is
  created/opened at startup.
- `Language model '<name>' wrapped with persistent LLM cache.` /
  `Embedder '<name>' wrapped with persistent LLM cache.` — when each resource is
  first built (lazily, on first use).
- At `debug` level: per-call `LLM cache hit/miss (...)` and
  `Embedding cache (<mode>): N hits, M misses of K inputs.` lines.

If none of these appear, the loaded config doesn't have `llm_cache.enabled: true` —
check the `Configuration file '<path>' loaded.` log line to see which file the
server actually read (`MEMORY_CONFIG` env var, falling back to
`~/.config/memmachine/cfg.yml`, then `./cfg.yml`).

## Determinism requirements (getting hits on reruns)

Cache hits require byte-identical prompts. The main pitfall: if a message is
added **without an explicit `timestamp`**, the server stamps it with the current
time (`MemoryMessage.timestamp` default, `memmachine_common/api/spec.py`), and
that timestamp is rendered into LLM prompts (`[Wednesday, June 03, 2026 at
06:21 PM] user: ...` via `episodes_to_string`) — so reruns of the same job
produce different cache keys and miss.

For deterministic reruns:

1. **Always pass an explicit `timestamp` per message** — the REST API accepts
   ISO-8601 / unix seconds / unix ms; the Python client's `add_memory` takes a
   `datetime`. Use the source data's real date, or derive a stable synthetic
   one (e.g. fixed epoch + record sequence number). Note: the MCP `add_memory`
   tool hardcodes `datetime.now()` server-side and cannot currently supply one.
2. **Preserve ingestion order per session** — short-term memory's rolling
   summary prompt embeds the previous summary, so the chain only replays from
   cache if episodes arrive in the same order.
3. **Set `llm_cache.deterministic_ingestion: true`** (see below) — without it,
   server-side timing races change the LLM requests across otherwise-identical
   runs even when 1 and 2 hold.

Verify a rerun: `sqlite3 llm_cache.db "SELECT COUNT(*) FROM llm_cache;"` should
barely grow, and debug logs show `LLM cache hit` lines.

### `deterministic_ingestion`

Two server-side timing races were found to break rerun determinism (confirmed
by diffing cached requests across two identical 10-message runs):

- **Short-term memory dynamic batching**: the summary worker drains whatever
  episodes accumulated while the previous LLM call was in flight, so batch
  boundaries depend on provider latency vs. add rate. With the flag on,
  `add_episodes` settles in-flight summarization before deciding eviction, so
  batches are a pure function of the add sequence.
- **Semantic consolidation cycle boundaries**: consolidation/dedup ran once
  per background polling cycle, so its LLM calls fired at points depending on
  where the poller's 5-message windows landed. With the flag on, the
  consolidation check runs after each message instead (the check itself is a
  cheap DB count; dedup LLM calls fire under the same threshold).
- **Semantic feature identity & render order**: the profile extraction
  (`<OLD_PROFILE>`) and consolidation prompts rendered features in DB order
  (`ORDER BY created_at, id`) and embedded the ephemeral DB row id
  (`metadata.id`) in the consolidation payload. Neither survives a database
  wipe — ids restart and are reassigned in a non-deterministic order by the
  concurrent re-add path (`asyncio.gather`) — so the prompts (and thus cache
  keys) differed across reruns. A single divergent consolidation produces a
  different surviving feature set, which becomes the next message's
  `<OLD_PROFILE>` and cascades a miss through the rest of the run. With the
  flag on, both prompts are canonicalized to depend only on feature *content*
  `(tag, feature, value)`: `<OLD_PROFILE>` is content-sorted, and the
  consolidation prompt presents each feature by its **position** in that sorted
  order instead of its DB id (positions are mapped back to real ids when
  applying `keep_memories`). This makes consolidation outcomes — and the whole
  profile — reproducible without serializing any ingestion work.

A third issue was a genuine bug fixed unconditionally: un-ingested messages
were ordered by `history_id` as a *string*, so message "10" was extracted
before message "6", scrambling extraction order differently per run (and out
of insertion order generally). Ordering is now (length, value) — numeric order
for the stringified integer ids.

Trade-off: with `deterministic_ingestion` on, an add that triggers
summarization waits for the previous summary LLM call (per session), reducing
ingestion parallelism within a session on cold (uncached) runs. Warm reruns
are fast since hits return immediately.

## Design note

Because caching reuses the first response for every duplicate prompt, reruns become
deterministic even at `temperature > 0` — the intended behavior here, but it means
sampling diversity across duplicate prompts within the cached corpus is lost.
