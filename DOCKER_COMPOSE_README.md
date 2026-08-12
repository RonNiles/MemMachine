# MemMachine Docker Setup Guide

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- OpenAI API key configured

### 1. Configure Environment
Copy the example environment file and add your OpenAI API key:
```bash
cp sample_configs/env.dockercompose .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Configure MemMachine
Copy the sample configuration file and update it with your settings:
```bash
cp sample_configs/episodic_memory_config.sample configuration.yml
# Edit configuration.yml and update:
# - Replace <YOUR_API_KEY> with your OpenAI API key
# - Replace <YOUR_PASSWORD_HERE> with your Neo4j password
# - Update host from 'localhost' to 'neo4j' for Docker environment
```

### 3. Start Services

#### Option A: Using the MemMachine Compose Script (Recommended)
Run the startup script:
```bash
./memmachine-compose.sh
```

This will:
- ✅ Check Docker and Docker Compose availability
- ✅ Verify .env file and OpenAI API key
- ✅ Check and create configuration.yml if needed
- ✅ Validate configuration settings
- ✅ Pull and start all services (PostgreSQL, Neo4j 5.23, MemMachine)
- ✅ Wait for all services to be healthy
- ✅ Display service URLs and connection info

#### Option B: Using Docker Compose Directly
```bash
docker-compose up -d
```

### 4. Access Services
Once started, you can access:

- **MemMachine API**: http://localhost:8080
- **Neo4j Browser**: http://localhost:7474
- **Health Check**: http://localhost:8080/health
- **Metrics**: http://localhost:8080/metrics

### 5. Test the Setup
```bash
# Test health endpoint
curl http://localhost:8080/health

# Test memory storage
curl -X POST "http://localhost:8080/v1/memories" \
  -H "Content-Type: application/json" \
  -d '{
    "session": {
      "group_id": "test-group",
      "agent_id": ["test-agent"],
      "user_id": ["test-user"],
      "session_id": "test-session-123"
    },
    "producer": "test-user",
    "produced_for": "test-user",
    "episode_content": "Hello, this is a test message",
    "episode_type": "text",
    "metadata": {"test": true}
  }'
```

## Useful Commands

### Using the MemMachine Compose Script (Recommended)

#### View Logs
```bash
./memmachine-compose.sh logs
```

#### Stop Services
```bash
./memmachine-compose.sh stop
```

#### Restart Services
```bash
./memmachine-compose.sh restart
```

#### Clean Up (Remove All Data)
```bash
./memmachine-compose.sh clean
```

#### Show Help
```bash
./memmachine-compose.sh help
```

### Using Docker Compose Directly

#### View Logs
```bash
docker-compose logs -f
```

#### Stop Services
```bash
docker-compose down
```

#### Restart Services
```bash
docker-compose restart
```

#### Clean Up (Remove All Data)
```bash
docker-compose down -v
```

## Backup, Restore & Replication

Three helper scripts under `scripts/` manage the databases (PostgreSQL + Neo4j,
plus the optional `llm_cache.db`):

- `db-backup.sh` — full backup / restore (binary dumps; fast, reliable).
- `db-sync-export.sh` / `db-sync-apply.sh` — incremental replication to keep a
  remote read-only mirror up to date without full restores.

**Compose container names.** The scripts default to the `dev-db.sh` names
(`memmachine-*-dev`). Under Docker Compose the containers are `memmachine-postgres`
and `memmachine-neo4j`, so export those overrides first:

```bash
export PG_CONTAINER=memmachine-postgres NEO_CONTAINER=memmachine-neo4j
# To include the LLM cache, enable the ./cache volume in docker-compose.yml
# (see the memmachine service) and set llm_cache.path: /app/cache/llm_cache.db,
# then also: export LLM_CACHE=./cache/llm_cache.db
```

### Full backup / restore

Stop the app first so the dumps are consistent (the DB containers stay up):

```bash
# Backup
docker compose stop memmachine
./scripts/db-backup.sh backup ./backups/$(date +%F)
docker compose start memmachine

# Restore (OVERWRITES the live databases)
docker compose stop memmachine
./scripts/db-backup.sh restore ./backups/2026-07-02
docker compose start memmachine
```

The backup dir contains `postgres.dump` + `neo4j.dump` (binary, used by restore),
`neo4j.cypher` (portable text export, not used by restore), `llm_cache.db`, and a
copy of your config. Restore uses the fast binary Neo4j load (~seconds); replaying
the large `neo4j.cypher` would take far longer, so it is intentionally not the
restore path.

### Incremental replication to a remote mirror

The remote is seeded once from a full backup, then kept current by shipping small
increment bundles. Adds, updates, and deletes all propagate; applies are
idempotent and must be applied in order.

**1. Establish the baseline (run together, app stopped, so they are consistent):**

```bash
# SOURCE
docker compose stop memmachine
./scripts/db-backup.sh backup ./baseline
./scripts/db-sync-export.sh baseline      # seeds sync cursors/manifest to match
docker compose start memmachine
# ship ./baseline to the remote host, then on the REMOTE:
docker compose stop memmachine
./scripts/db-backup.sh restore ./baseline
# leave the remote app stopped — it is a read-only mirror
```

**2. Sync periodically:**

```bash
# SOURCE (online — no need to stop the app)
./scripts/db-sync-export.sh export ./outgoing
# -> writes ./outgoing/bundle-NNNNNN ; ship it to the remote (rsync/scp/etc.)

# REMOTE — apply bundles strictly in order (already-applied bundles are skipped)
./scripts/db-sync-apply.sh ./incoming/bundle-000001
```

How each store is handled: `llm_cache` ships new rows (`INSERT OR IGNORE`); Neo4j
diffs a node-uid manifest (new subgraph added, removed uids `DETACH DELETE`d);
Postgres ships a full dump each run (it is small, and a snapshot inherently
carries updates + deletes). Runtime state lives in `.sync-state/` (source) and
`.sync-state-remote/` (remote) — both git-ignored. Bundle transport between hosts
is up to you.

## Services

- **PostgreSQL** (port 5432): Profile memory storage with pgvector
- **Neo4j** (ports 7474, 7687): Episodic memory with vector similarity
- **MemMachine** (port 8080): Main API server (uses pre-built `memmachine/memmachine` image)

## Configuration

Key files:
- `.env` - Environment variables
- `configuration.yml` - MemMachine configuration
- `docker-compose.yml` - Service definitions
- `memmachine-compose.sh` - Startup script with validation and health checks

### ⚠️ Important Configuration Notes

**1. Database Configuration Consistency**
Make sure the database configuration details in `configuration.yml` match the database configuration details in `.env`

Both files must have consistent:
- Database hostnames (use service names: `postgres`, `neo4j`)
- Database ports (5432 for PostgreSQL, 7687 for Neo4j)
- Database credentials (usernames and passwords)
- Database names

**2. Configuration.yml Setup**
The `configuration.yml` file contains MemMachine-specific settings:
- **Model configuration**: OpenAI API settings for LLM and embeddings
- **Storage configuration**: Neo4j connection details
- **Memory settings**: Session memory capacity and limits
- **Reranker configuration**: Search and ranking algorithms

**Key settings to update in configuration.yml:**
- Replace `<YOUR_API_KEY>` with your OpenAI API key (appears in both Model and embedder sections)
- Replace `<YOUR_PASSWORD_HERE>` with your Neo4j password
- Ensure the Neo4j host is set to `neo4j` (not `localhost`) for Docker environment

This ensures MemMachine can properly connect to the Docker services and use your OpenAI API key for embeddings and LLM operations.
