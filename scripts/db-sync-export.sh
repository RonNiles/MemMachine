#!/usr/bin/env bash
# Incremental replication — SOURCE side.
#
# Produces per-run "increment bundles" that a remote read-only mirror applies
# with scripts/db-sync-apply.sh to stay current, without a full restore. The
# remote is first seeded from a binary full backup (scripts/db-backup.sh
# restore); then `baseline` seeds this tool's state to match, and each `export`
# ships only what changed since the previous run.
#
# Per store (see docs in db-backup.sh for the data-model rationale):
#   llm_cache : append-only SQLite -> ship rows with rowid > cursor as
#               INSERT OR IGNORE. Cursor advances to MAX(rowid).
#   Neo4j     : append-only graph + explicit deletes -> diff a persisted node-uid
#               manifest. Added uids -> apoc cypher export of that subgraph
#               (app index DDL stripped; the remote already has those indexes).
#               Deleted uids -> MATCH (n {uid}) DETACH DELETE (cascades edges).
#   Postgres  : mutable but tiny (~30 MB) -> full pg_dump every run (a snapshot
#               is always correct and inherently reflects updates + deletes).
#
# Reads are online-safe (SQLite --readonly on WAL, apoc read, pg_dump MVCC), so
# this can run against a live source without stopping the server.
#
# Container names / credentials mirror scripts/dev-db.sh; override via env.
set -euo pipefail

PG_CONTAINER="${PG_CONTAINER:-memmachine-postgres-dev}"
NEO_CONTAINER="${NEO_CONTAINER:-memmachine-neo4j-dev}"
PG_USER="${PG_USER:-memmachine}"
PG_DB="${PG_DB:-memmachine}"
NEO_USER="${NEO_USER:-neo4j}"
NEO_PASS="${NEO_PASS:-neo4j_password}"
NEO_IMPORT_DIR="${NEO_IMPORT_DIR:-/var/lib/neo4j/import}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLM_CACHE="${LLM_CACHE:-$REPO_ROOT/llm_cache.db}"
# Persisted replication state (cursors + neo4j uid manifest + bundle counter).
SYNC_STATE="${SYNC_STATE:-$REPO_ROOT/.sync-state}"

die() { echo "[!] $*" >&2; exit 1; }
neo_cypher() { docker exec "$NEO_CONTAINER" cypher-shell -u "$NEO_USER" -p "$NEO_PASS" "$@"; }

usage() {
  cat <<EOF
Usage: $0 baseline            Seed state to match the current source (no data emitted).
       $0 export <out-dir>    Emit an increment bundle of changes since last run.

Run 'baseline' once, right after the remote has been seeded from a binary full
backup, so the first 'export' ships only subsequent changes.

State dir: $SYNC_STATE   (override with SYNC_STATE=)
Env overrides: PG_CONTAINER NEO_CONTAINER PG_USER PG_DB NEO_USER NEO_PASS
               NEO_IMPORT_DIR LLM_CACHE SYNC_STATE
EOF
}

# Sorted (C-collation) list of all node uids currently in the graph -> stdout.
neo_node_uids() {
  neo_cypher --format plain "MATCH (n) RETURN n.uid;" \
    | tail -n +2 | tr -d '"' | sed '/^$/d' | LC_ALL=C sort -u
}

sqlite_ro() { sqlite3 --readonly "$LLM_CACHE" "$@"; }

baseline() {
  mkdir -p "$SYNC_STATE"
  if [ -f "$LLM_CACHE" ]; then
    sqlite_ro "SELECT COALESCE(MAX(rowid),0) FROM llm_cache;"       > "$SYNC_STATE/llm_cache.rowid"
    sqlite_ro "SELECT COALESCE(MAX(rowid),0) FROM embedding_cache;" > "$SYNC_STATE/embedding_cache.rowid"
  else
    echo 0 > "$SYNC_STATE/llm_cache.rowid"; echo 0 > "$SYNC_STATE/embedding_cache.rowid"
  fi
  neo_node_uids > "$SYNC_STATE/neo4j.manifest"
  echo 0 > "$SYNC_STATE/bundle.seq"
  echo "[✓] baseline seeded in $SYNC_STATE"
  echo "    llm_cache rowid=$(cat "$SYNC_STATE/llm_cache.rowid") embedding_cache rowid=$(cat "$SYNC_STATE/embedding_cache.rowid") neo4j uids=$(wc -l < "$SYNC_STATE/neo4j.manifest")"
}

export_bundle() {
  local out="${1:-}"
  [ -n "$out" ] || { usage; exit 1; }
  [ -f "$SYNC_STATE/bundle.seq" ] || die "no state in $SYNC_STATE — run '$0 baseline' first."

  local seq; seq=$(( $(cat "$SYNC_STATE/bundle.seq") + 1 ))
  local bundle; bundle="$out/bundle-$(printf '%06d' "$seq")"
  mkdir -p "$bundle"
  local ts; ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local n_cache=0 n_add=0 n_del=0

  # --- llm_cache: rows with rowid > cursor, as INSERT OR IGNORE -------------
  if [ -f "$LLM_CACHE" ]; then
    local curL curE maxL maxE
    curL="$(cat "$SYNC_STATE/llm_cache.rowid")"
    curE="$(cat "$SYNC_STATE/embedding_cache.rowid")"
    maxL="$(sqlite_ro 'SELECT COALESCE(MAX(rowid),0) FROM llm_cache;')"
    maxE="$(sqlite_ro 'SELECT COALESCE(MAX(rowid),0) FROM embedding_cache;')"
    sqlite3 <<SQL | sed 's/^INSERT INTO/INSERT OR IGNORE INTO/' > "$bundle/llm_cache.sql"
.open --readonly "$LLM_CACHE"
.mode insert llm_cache
SELECT * FROM llm_cache WHERE rowid > $curL AND rowid <= $maxL;
.mode insert embedding_cache
SELECT * FROM embedding_cache WHERE rowid > $curE AND rowid <= $maxE;
SQL
    n_cache="$(grep -c '^INSERT OR IGNORE' "$bundle/llm_cache.sql" || true)"
    printf '%s' "$maxL" > "$SYNC_STATE/llm_cache.rowid.next"
    printf '%s' "$maxE" > "$SYNC_STATE/embedding_cache.rowid.next"
  fi

  # --- neo4j: uid-manifest diff -> add subgraph + del statements ------------
  local cur added deleted
  cur="$(mktemp)"; added="$(mktemp)"; deleted="$(mktemp)"
  neo_node_uids > "$cur"
  LC_ALL=C comm -13 "$SYNC_STATE/neo4j.manifest" "$cur" > "$added"    # in current, not manifest
  LC_ALL=C comm -23 "$SYNC_STATE/neo4j.manifest" "$cur" > "$deleted"  # in manifest, not current
  n_add="$(wc -l < "$added")"; n_del="$(wc -l < "$deleted")"

  if [ "$n_add" -gt 0 ]; then
    cp "$added" "$bundle/neo4j-add.uids"
    local uidlist; uidlist="$(sed "s/.*/'&'/" "$added" | paste -sd,)"
    # Restrict both endpoints to the added set so pre-existing neighbour nodes are
    # NOT re-CREATEd on the remote (which would duplicate them). Safe because the
    # source adds an episode + its derivatives + their edges atomically.
    local q="MATCH (n) WHERE n.uid IN [$uidlist] OPTIONAL MATCH (n)-[r]-(m) WHERE m.uid IN [$uidlist] RETURN n, r, m"
    neo_cypher \
      "CALL apoc.export.cypher.query(\"$q\", 'mm-inc.cypher', {format:'cypher-shell', useOptimizations:{type:'NONE'}}) YIELD nodes RETURN nodes;" \
      >/dev/null || die "neo4j incremental export failed (apoc.export.file.enabled=true?)"
    # Strip app index DDL — the remote already has those indexes (baseline);
    # re-creating them would error. Keep the temp UNIQUE_IMPORT_NAME constraint,
    # the node/edge data, and the cleanup that the import mechanism needs.
    docker exec "$NEO_CONTAINER" sh -c "cat '$NEO_IMPORT_DIR/mm-inc.cypher'" \
      | grep -vE '^CREATE (RANGE|VECTOR|POINT|TEXT|FULLTEXT|LOOKUP) INDEX' > "$bundle/neo4j-add.cypher"
    docker exec "$NEO_CONTAINER" rm -f "$NEO_IMPORT_DIR/mm-inc.cypher"
  fi
  if [ "$n_del" -gt 0 ]; then
    # A node delete cascades its edges (DETACH DELETE), matching the source.
    sed "s/.*/MATCH (n {uid:'&'}) DETACH DELETE n;/" "$deleted" > "$bundle/neo4j-del.cypher"
  fi

  # --- postgres: full snapshot ---------------------------------------------
  docker exec "$PG_CONTAINER" pg_dump -U "$PG_USER" -d "$PG_DB" -Fc > "$bundle/postgres.dump"

  # --- manifest + commit state (only after everything above succeeded) ------
  {
    echo "seq: $seq"
    echo "created: $ts"
    echo "llm_cache_rows: $n_cache"
    echo "neo4j_added: $n_add  neo4j_deleted: $n_del"
    echo "postgres: full snapshot (postgres.dump)"
  } > "$bundle/MANIFEST.txt"

  [ -f "$SYNC_STATE/llm_cache.rowid.next" ] && mv "$SYNC_STATE/llm_cache.rowid.next" "$SYNC_STATE/llm_cache.rowid"
  [ -f "$SYNC_STATE/embedding_cache.rowid.next" ] && mv "$SYNC_STATE/embedding_cache.rowid.next" "$SYNC_STATE/embedding_cache.rowid"
  mv "$cur" "$SYNC_STATE/neo4j.manifest"
  echo "$seq" > "$SYNC_STATE/bundle.seq"
  rm -f "$added" "$deleted"

  echo "[✓] bundle $seq -> $bundle"
  echo "    llm_cache_rows=$n_cache neo4j_added=$n_add neo4j_deleted=$n_del postgres=full"
}

cmd="${1:-}"; shift || true
case "$cmd" in
  baseline) baseline ;;
  export)   export_bundle "${1:-}" ;;
  *)        usage; exit 1 ;;
esac
