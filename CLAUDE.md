# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MemMachine is an open-source long-term memory layer for AI agents — a Python-first monorepo (Python 3.12+) with a TypeScript REST client. The server exposes a FastAPI REST API (`/api/v2`) and a native MCP server (stdio + HTTP) backed by Neo4j (episodic / graph memory) and PostgreSQL+pgvector (semantic / profile memory).

For richer context on user-facing behavior see `README.md`, `USAGE.md`, and `DOCKER_COMPOSE_README.md`. `AGENTS.md` is the upstream agent guide; this file extends and supersedes it where they conflict.

## Workspace layout

uv workspace defined in the root `pyproject.toml`. Members:

- `packages/common/` — `memmachine-common`: shared Pydantic API models / DTOs used by both server and client. No runtime dependencies on the others.
- `packages/server/` — `memmachine-server`: the FastAPI server, memory engines, retrieval agents, MCP entrypoints, and Alembic migrations. Source under `src/memmachine_server/`.
- `packages/client/` — `memmachine-client`: lightweight Python REST client (`MemMachineClient`).
- `packages/meta/` — `memmachine`: meta-package that simply pins `memmachine-client` + `memmachine-server` for the `pip install memmachine` convenience install. **Not** a workspace member — it lives outside the uv workspace and is published separately.
- `packages/ts-client/` — `@memmachine/client`: TypeScript REST client (tsup build, Jest tests, ESLint + Prettier).
- `integrations/` — adapters for LangChain, LangGraph, CrewAI, LlamaIndex, AWS Strands, n8n, Dify, FastGPT, openclaw.
- `examples/`, `evaluation/`, `tools/` — demos, benchmarks, helper scripts. These are lint-relaxed via `per-file-ignores` in the root `pyproject.toml` — don't tighten them without reason.
- `sample_configs/` — example `configuration.yml` files (cpu, gpu, nebula) and `env.dockercompose`. Users copy these to repo root as `configuration.yml` / `.env`.
- `deployments/helm/` — Helm charts.

### Server package internal structure (`packages/server/src/memmachine_server/`)

- `server/` — FastAPI app (`app.py`), HTTP middleware, MCP entrypoints (`mcp_stdio.py`, `mcp_http.py`), and the v2 router under `server/api_v2/` (split into `router.py`, `service.py`, `config_router.py`, `config_service.py`, `mcp.py`, `exceptions.py`).
- `episodic_memory/` — episodic memory engine: short-term, long-term, declarative, and event sub-stores plus `episodic_memory_manager.py` and `service_locator.py`. Backed by Neo4j.
- `semantic_memory/` — semantic/profile memory: ingestion pipeline, LLM extraction, clustering (`cluster_manager.py`, `cluster_splitter.py`), session manager, and Postgres storage under `semantic_memory/storage/` (includes the Alembic migration env).
- `retrieval_agent/` — retrieval agents and orchestration layer used by search endpoints.
- `common/` — shared infrastructure: embedders, language models, vector/graph stores, rerankers, episode store, filter/payload codecs, metrics, resource manager, RW locks, Neo4j utilities, error types.
- `installation/` — `memmachine-configure` CLI wizard for bootstrapping a config file.
- `main/` — small top-level entry shim.

Console scripts (declared in `packages/server/pyproject.toml`): `memmachine-server`, `memmachine-mcp-stdio`, `memmachine-mcp-http`, `memmachine-configure`, `memmachine-nltk-setup`.

## Environment setup

```bash
uv sync                 # install workspace + dev deps
uv sync --all-extras    # include optional extras (gpu, hnswlib, nebula, qdrant)
```

Python 3.12+ required. If not using uv, `pip install -e ".[gpu]"` works against any of the package dirs.

## Build / lint / test (Python)

Run from repo root:

```bash
uv run ruff check              # lint
uv run ruff format             # auto-format
uv run ty check packages       # type check (Astral's ty, not mypy)
uv run pytest                  # run tests (excludes 'integration' by default — see pytest addopts)
uv run pytest -m integration   # run integration tests
uv run pytest -m slow          # run slow tests
uv run pytest -k "create_memory"                                            # by keyword
uv run pytest packages/server/server_tests/.../test_utils.py::test_chunk_text  # single test
```

Pytest defaults (root `pyproject.toml`): asyncio session-scoped loops; `addopts = ["-m", "not integration"]`, so integration tests must be opted in. Mark new tests with `@pytest.mark.integration` or `@pytest.mark.slow` when appropriate. Integration tests rely on `testcontainers[neo4j,postgres,qdrant]` and need Docker.

Ruff is pinned (`ruff==0.15.12`) and uses a broad rule set (see `[tool.ruff.lint] select`). Per-directory relaxations exist for `tests/`, `tools/`, `examples/`, `integrations/`, `evaluation/` — when editing files there, the relaxed rules are intentional.

`complexipy` enforces a max cyclomatic complexity of 10 across `packages/{server,client,common}/src` — surfaces as CI failures, not Ruff warnings.

## TypeScript client

From `packages/ts-client/`:

```bash
npm install
npm run build              # eslint --fix + prettier + tsup
npm run lint               # eslint
npm run format             # prettier --write
npm run test               # jest
npm run test -- -t "name"  # single test by name
```

## Database migrations (semantic memory)

Alembic is configured at the repo root via `alembic.ini`. Script location: `packages/server/src/memmachine_server/semantic_memory/storage/alembic_pg`. Run `uv run alembic ...` from the repo root.

## Docker / runtime

- Local stack: `./memmachine-compose.sh` (preferred — validates `.env`, `configuration.yml`, waits for health) or `docker-compose up -d`.
- Image build: `./build-docker.sh` (interactive; flags documented in the script header).
- Required env var: `OPENAI_API_KEY` (or whichever provider is configured in `configuration.yml`).
- The compose stack runs PostgreSQL (5432), Neo4j (7474/7687), and MemMachine (8080). When editing `configuration.yml` for Docker, use the service names `postgres` and `neo4j` as hosts — not `localhost`.

## Conventions worth knowing

- **Async-first**: server code uses `async def` end-to-end. Don't introduce blocking I/O inside async paths — wrap with `asyncio.to_thread` or use an async client.
- **Pydantic everywhere** for request/response and structured data; prefer `pydantic` models over dicts at API boundaries.
- **Public APIs are versioned** under `/api/v2`. Treat `memmachine_common.api` as the contract between server and clients — changes there are user-visible.
- **Commits must be signed** (`git commit -sS`). Unsigned commits fail CI.
- **Don't reformat unrelated files.** Ruff's auto-format is aggressive; restrict edits to files relevant to your change.
- The MCP layer (`server/api_v2/mcp.py`) mounts at `/mcp` and shares lifespan with the FastAPI app — config and resource initialization flow through `mcp.initialize_resource` / `load_configuration`.
