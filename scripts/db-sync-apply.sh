#!/usr/bin/env bash
# Incremental replication — REMOTE side. Applies an increment bundle produced by
# scripts/db-sync-export.sh to keep a read-only mirror current.
#
# The remote must first be seeded from a binary full backup
# (scripts/db-backup.sh restore) so schemas/baseline data exist. Bundles are
# then applied strictly in order (seq = previous + 1); a bundle already applied
# is skipped, and every operation is itself idempotent, so a re-run is a no-op:
#   Postgres  : full replace (drop schemas + pg_restore) — reflects updates/deletes.
#   Neo4j     : delete-then-create the added uids (so re-apply can't duplicate;
#               Neo4j has no uid uniqueness constraint), then run the deletions.
#   llm_cache : INSERT OR IGNORE (append-only; dupes ignored).
#
# Run with NO memmachine-server writing to the remote (it is a read-only mirror).
# Container names / credentials mirror scripts/dev-db.sh; override via env — point
# them at the REMOTE stack.
set -euo pipefail

PG_CONTAINER="${PG_CONTAINER:-memmachine-postgres-dev}"
NEO_CONTAINER="${NEO_CONTAINER:-memmachine-neo4j-dev}"
PG_USER="${PG_USER:-memmachine}"
PG_DB="${PG_DB:-memmachine}"
NEO_USER="${NEO_USER:-neo4j}"
NEO_PASS="${NEO_PASS:-neo4j_password}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLM_CACHE="${LLM_CACHE:-$REPO_ROOT/llm_cache.db}"
# Tracks the last successfully applied bundle seq (guards order + re-apply).
SYNC_STATE="${SYNC_STATE:-$REPO_ROOT/.sync-state-remote}"

die() { echo "[!] $*" >&2; exit 1; }
neo_cypher() { docker exec "$NEO_CONTAINER" cypher-shell -u "$NEO_USER" -p "$NEO_PASS" "$@"; }

usage() {
  cat <<EOF
Usage: $0 <bundle-dir>

Applies one increment bundle to the remote (Postgres + Neo4j + llm_cache),
in strict seq order. Idempotent: a bundle already applied is skipped.

State dir: $SYNC_STATE   (override with SYNC_STATE=)
Env overrides: PG_CONTAINER NEO_CONTAINER PG_USER PG_DB NEO_USER NEO_PASS
               LLM_CACHE SYNC_STATE
EOF
}

apply() {
  local bundle="${1:-}"
  [ -n "$bundle" ] || { usage; exit 1; }
  [ -d "$bundle" ] || die "not a directory: $bundle"
  [ -f "$bundle/MANIFEST.txt" ] || die "missing $bundle/MANIFEST.txt"
  [ -f "$bundle/postgres.dump" ] || die "missing $bundle/postgres.dump"

  local seq applied
  seq="$(awk -F': ' '/^seq:/{print $2}' "$bundle/MANIFEST.txt")"
  [ -n "$seq" ] || die "no seq in $bundle/MANIFEST.txt"
  mkdir -p "$SYNC_STATE"
  applied="$(cat "$SYNC_STATE/applied.seq" 2>/dev/null || echo 0)"

  if [ "$seq" -le "$applied" ]; then
    echo "[=] bundle seq $seq already applied (applied=$applied) — skipping."
    return 0
  fi
  [ "$seq" -eq "$((applied + 1))" ] \
    || die "out-of-order: expected seq $((applied + 1)), got $seq — apply intermediate bundles first."

  # --- Postgres: full replace (reflects updates AND deletes) ----------------
  echo "[*] Postgres: full replace from $bundle/postgres.dump"
  docker cp "$bundle/postgres.dump" "$PG_CONTAINER:/tmp/sync-postgres.dump"
  docker exec "$PG_CONTAINER" psql -U "$PG_USER" -d "$PG_DB" -v ON_ERROR_STOP=1 \
    -c "DROP SCHEMA IF EXISTS public CASCADE; DROP SCHEMA IF EXISTS metadata CASCADE; CREATE SCHEMA public;"
  docker exec "$PG_CONTAINER" pg_restore --no-owner -U "$PG_USER" -d "$PG_DB" /tmp/sync-postgres.dump
  docker exec "$PG_CONTAINER" rm -f /tmp/sync-postgres.dump

  # --- Neo4j: adds (delete-then-create = idempotent), then deletes ----------
  if [ -f "$bundle/neo4j-add.cypher" ]; then
    echo "[*] Neo4j: apply $(wc -l < "$bundle/neo4j-add.uids" 2>/dev/null || echo 0) added node(s)"
    if [ -s "$bundle/neo4j-add.uids" ]; then
      local uidlist; uidlist="$(sed "s/.*/'&'/" "$bundle/neo4j-add.uids" | paste -sd,)"
      neo_cypher "MATCH (n) WHERE n.uid IN [$uidlist] DETACH DELETE n;" >/dev/null
    fi
    # A prior interrupted apply could leave the temp import constraint behind.
    neo_cypher "DROP CONSTRAINT UNIQUE_IMPORT_NAME IF EXISTS;" >/dev/null
    docker exec -i "$NEO_CONTAINER" cypher-shell -u "$NEO_USER" -p "$NEO_PASS" < "$bundle/neo4j-add.cypher"
  fi
  if [ -f "$bundle/neo4j-del.cypher" ]; then
    echo "[*] Neo4j: apply $(wc -l < "$bundle/neo4j-del.cypher") deletion(s)"
    docker exec -i "$NEO_CONTAINER" cypher-shell -u "$NEO_USER" -p "$NEO_PASS" < "$bundle/neo4j-del.cypher"
  fi

  # --- llm_cache: INSERT OR IGNORE (append-only) ---------------------------
  if [ -f "$bundle/llm_cache.sql" ] && [ -s "$bundle/llm_cache.sql" ]; then
    [ -f "$LLM_CACHE" ] || die "remote $LLM_CACHE missing — seed it from the baseline backup first."
    echo "[*] llm_cache: $(grep -c '^INSERT OR IGNORE' "$bundle/llm_cache.sql") row(s) -> $LLM_CACHE"
    sqlite3 "$LLM_CACHE" < "$bundle/llm_cache.sql"
  fi

  echo "$seq" > "$SYNC_STATE/applied.seq"
  echo "[✓] applied bundle seq $seq"
}

apply "${1:-}"
