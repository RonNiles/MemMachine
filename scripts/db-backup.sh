#!/usr/bin/env bash
# Backup / restore the MemMachine databases (Postgres + Neo4j).
#
# Run with the DB containers UP but memmachine-server DOWN, so there is no
# write activity and the dumps are consistent. The script refuses to run if it
# detects a running server (override with FORCE=1 / --force).
#
# Postgres : pg_dump custom-format archive (postgres.dump).
# Neo4j    : STOP DATABASE -> neo4j-admin database dump -> START DATABASE
#            (offline dump of just the `neo4j` database; the container/DBMS
#            stays up, only the one database is briefly stopped).
# Config   : copies configuration.yml / .env (whichever exist) for reference.
#
# Container names and credentials match scripts/dev-db.sh; override via env.
set -euo pipefail

PG_CONTAINER="${PG_CONTAINER:-memmachine-postgres-dev}"
NEO_CONTAINER="${NEO_CONTAINER:-memmachine-neo4j-dev}"
PG_USER="${PG_USER:-memmachine}"
PG_DB="${PG_DB:-memmachine}"
NEO_USER="${NEO_USER:-neo4j}"
NEO_PASS="${NEO_PASS:-neo4j_password}"
NEO_DB="${NEO_DB:-neo4j}"
# Space-separated, resolved relative to the repo root.
CONFIG_FILES="${CONFIG_FILES:-configuration.yml .env}"
FORCE="${FORCE:-0}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
  cat <<EOF
Usage: $0 backup  <target-dir> [--force]
       $0 restore <source-dir> [--force]

  backup   Dump Postgres + Neo4j and copy config files into <target-dir>.
  restore  OVERWRITE the live Postgres + Neo4j data from <source-dir>.

Both require the two DB containers running and memmachine-server stopped.
Env overrides: PG_CONTAINER NEO_CONTAINER PG_USER PG_DB NEO_USER NEO_PASS
               NEO_DB CONFIG_FILES FORCE
EOF
}

die() { echo "[!] $*" >&2; exit 1; }

require_running() {
  docker ps --format '{{.Names}}' | grep -qx "$1" \
    || die "container '$1' is not running (start it: scripts/dev-db.sh up)"
}

ensure_server_down() {
  [ "$FORCE" = "1" ] && { echo "[!] FORCE set — skipping server-down check"; return 0; }
  if pgrep -f 'memmachine-server' >/dev/null 2>&1; then
    die "memmachine-server appears to be running — stop it first (or FORCE=1)."
  fi
  local others
  others="$(docker ps --format '{{.Names}}' | grep -i memmachine \
            | grep -vxE "${PG_CONTAINER}|${NEO_CONTAINER}" || true)"
  [ -z "$others" ] || die "other memmachine container(s) running: ${others//$'\n'/ } (stop them, or FORCE=1)."
}

neo4j_sys() { docker exec "$NEO_CONTAINER" cypher-shell -u "$NEO_USER" -p "$NEO_PASS" -d system "$1" >/dev/null; }
neo4j_start_safety() { neo4j_sys "START DATABASE $NEO_DB WAIT" 2>/dev/null || true; }

backup() {
  local target="${1:-}"
  [ -n "$target" ] || { usage; exit 1; }
  require_running "$PG_CONTAINER"
  require_running "$NEO_CONTAINER"
  ensure_server_down

  mkdir -p "$target/config"
  local ts; ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  echo "[*] Postgres: pg_dump $PG_DB -> $target/postgres.dump"
  docker exec "$PG_CONTAINER" pg_dump -U "$PG_USER" -d "$PG_DB" -Fc > "$target/postgres.dump"

  echo "[*] Neo4j: stop '$NEO_DB', dump, start"
  docker exec "$NEO_CONTAINER" sh -c 'rm -rf /tmp/mm-backup && mkdir -p /tmp/mm-backup'
  neo4j_sys "STOP DATABASE $NEO_DB WAIT"
  trap neo4j_start_safety EXIT
  docker exec "$NEO_CONTAINER" neo4j-admin database dump "$NEO_DB" \
    --to-path=/tmp/mm-backup --overwrite-destination=true
  neo4j_sys "START DATABASE $NEO_DB WAIT"
  trap - EXIT
  docker cp "$NEO_CONTAINER:/tmp/mm-backup/$NEO_DB.dump" "$target/neo4j.dump"

  for f in $CONFIG_FILES; do
    if [ -f "$REPO_ROOT/$f" ]; then
      cp "$REPO_ROOT/$f" "$target/config/"
      echo "[*] config: $f"
    fi
  done

  {
    echo "created: $ts"
    echo "pg_container: $PG_CONTAINER  pg_db: $PG_DB"
    echo "neo_container: $NEO_CONTAINER  neo_db: $NEO_DB"
    echo "files: postgres.dump neo4j.dump config/"
  } > "$target/MANIFEST.txt"

  echo "[✓] backup complete -> $target"
  ls -lh "$target"
}

restore() {
  local src="${1:-}"
  [ -n "$src" ] || { usage; exit 1; }
  [ -f "$src/postgres.dump" ] || die "missing $src/postgres.dump"
  [ -f "$src/neo4j.dump" ]    || die "missing $src/neo4j.dump"
  require_running "$PG_CONTAINER"
  require_running "$NEO_CONTAINER"
  ensure_server_down

  if [ "$FORCE" != "1" ]; then
    echo "This OVERWRITES the live Postgres ($PG_DB) and Neo4j ($NEO_DB) data from $src."
    read -r -p "Continue? [y/N] " a
    case "$a" in [yY]|[yY][eE][sS]) ;; *) echo "Aborted."; exit 1 ;; esac
  fi

  echo "[*] Postgres: drop schema + pg_restore"
  docker cp "$src/postgres.dump" "$PG_CONTAINER:/tmp/postgres.dump"
  docker exec "$PG_CONTAINER" psql -U "$PG_USER" -d "$PG_DB" -v ON_ERROR_STOP=1 \
    -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
  docker exec "$PG_CONTAINER" pg_restore --no-owner -U "$PG_USER" -d "$PG_DB" /tmp/postgres.dump
  docker exec "$PG_CONTAINER" rm -f /tmp/postgres.dump

  echo "[*] Neo4j: stop '$NEO_DB', load, start"
  docker exec "$NEO_CONTAINER" sh -c 'rm -rf /tmp/mm-backup && mkdir -p /tmp/mm-backup'
  docker cp "$src/neo4j.dump" "$NEO_CONTAINER:/tmp/mm-backup/$NEO_DB.dump"
  neo4j_sys "STOP DATABASE $NEO_DB WAIT"
  trap neo4j_start_safety EXIT
  docker exec "$NEO_CONTAINER" neo4j-admin database load "$NEO_DB" \
    --from-path=/tmp/mm-backup --overwrite-destination=true
  neo4j_sys "START DATABASE $NEO_DB WAIT"
  trap - EXIT

  echo "[✓] restore complete."
  echo "    Note: config files in $src/config are NOT auto-applied — copy them"
  echo "    back into place manually if you intend to use them."
}

cmd="${1:-}"; shift || true
# strip a trailing --force into FORCE
args=(); for a in "$@"; do [ "$a" = "--force" ] && FORCE=1 || args+=("$a"); done

case "$cmd" in
  backup)  backup "${args[0]:-}" ;;
  restore) restore "${args[0]:-}" ;;
  *)       usage; exit 1 ;;
esac
