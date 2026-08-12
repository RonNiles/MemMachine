#!/usr/bin/env bash
# Backup / restore the MemMachine databases (Postgres + Neo4j + llm_cache).
#
# Run with the DB containers UP but memmachine-server DOWN, so there is no
# write activity and the dumps are consistent. The script refuses to run if it
# detects a running server (override with FORCE=1 / --force).
#
# HYBRID artifacts — each store gets a binary artifact used for RESTORE, plus
# (for Neo4j) a portable text dump kept for auditing / incremental replication:
#
# Postgres : pg_dump custom-format archive (postgres.dump). Restore = pg_restore.
# Neo4j    : neo4j.dump (offline `neo4j-admin database dump` via a throwaway
#            container — the fast, reliable RESTORE path) AND neo4j.cypher
#            (online `apoc.export.cypher.all`, a replayable cypher-shell text
#            dump kept for portability; NOT used by restore because replaying a
#            large embedding-heavy graph is impractically slow). The cypher
#            export needs the apoc plugin with apoc.export.file.enabled=true
#            (see scripts/dev-db.sh).
# Config   : copies configuration.yml / .env (whichever exist) for reference.
# Cache    : copies llm_cache.db (+ -wal/-shm sidecars) from the repo root if
#            present — the persistent LLM/embedding cache. Restored in place.
#
# Container names and credentials match scripts/dev-db.sh; override via env.
set -euo pipefail

PG_CONTAINER="${PG_CONTAINER:-memmachine-postgres-dev}"
NEO_CONTAINER="${NEO_CONTAINER:-memmachine-neo4j-dev}"
PG_USER="${PG_USER:-memmachine}"
PG_DB="${PG_DB:-memmachine}"
NEO_DB="${NEO_DB:-neo4j}"
NEO_USER="${NEO_USER:-neo4j}"
NEO_PASS="${NEO_PASS:-neo4j_password}"
# uid:gid the Neo4j data files are owned by inside the container (official
# image uses 7474:7474). Used to chown /data back after an offline restore.
NEO_UID="${NEO_UID:-7474}"
NEO_GID="${NEO_GID:-7474}"
# apoc writes exports into the server import dir; overridable if relocated.
NEO_IMPORT_DIR="${NEO_IMPORT_DIR:-/var/lib/neo4j/import}"
# Transient export filename inside the container (copied out to neo4j.cypher).
NEO_CYPHER_FILE="${NEO_CYPHER_FILE:-mm-neo4j.cypher}"
# apoc export batch size = entities per :begin/:commit block in neo4j.cypher.
# Kept small because nodes carry large embedding vectors (the default 20000
# builds transactions too big to replay within a modest container heap).
NEO_BATCH_SIZE="${NEO_BATCH_SIZE:-1000}"
# Space-separated, resolved relative to the repo root.
CONFIG_FILES="${CONFIG_FILES:-configuration.yml .env}"
# SQLite LLM/embedding cache at the repo root; backed up if present.
LLM_CACHE="${LLM_CACHE:-llm_cache.db}"
FORCE="${FORCE:-0}"

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
  cat <<EOF
Usage: $0 backup  <target-dir> [--force]
       $0 restore <source-dir> [--force]

  backup   Dump Postgres + Neo4j (binary + cypher) and copy config into <target-dir>.
  restore  OVERWRITE the live Postgres + Neo4j data from <source-dir> (binary dumps).

Both require the two DB containers running and memmachine-server stopped.
Env overrides: PG_CONTAINER NEO_CONTAINER PG_USER PG_DB NEO_DB NEO_USER NEO_PASS
               NEO_UID NEO_GID NEO_IMPORT_DIR NEO_CYPHER_FILE NEO_BATCH_SIZE
               CONFIG_FILES FORCE
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

neo_cypher()        { docker exec "$NEO_CONTAINER" cypher-shell -u "$NEO_USER" -p "$NEO_PASS" "$@"; }
neo_image()         { docker inspect -f '{{.Config.Image}}' "$NEO_CONTAINER"; }
neo_start_safety()  { docker start "$NEO_CONTAINER" >/dev/null 2>&1 || true; }

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

  # Neo4j text dump (online) — portability/audit artifact; restore does NOT use
  # it. apoc writes into the server import dir; copy it out and remove the copy.
  echo "[*] Neo4j: apoc.export.cypher.all -> $target/neo4j.cypher"
  neo_cypher \
    "CALL apoc.export.cypher.all('$NEO_CYPHER_FILE', {format:'cypher-shell', batchSize:$NEO_BATCH_SIZE})
     YIELD file, nodes, relationships RETURN file, nodes, relationships;" \
    || die "neo4j cypher export failed — is apoc.export.file.enabled=true? Recreate the container (scripts/dev-db.sh down && up)."
  docker cp "$NEO_CONTAINER:$NEO_IMPORT_DIR/$NEO_CYPHER_FILE" "$target/neo4j.cypher"
  docker exec "$NEO_CONTAINER" rm -f "$NEO_IMPORT_DIR/$NEO_CYPHER_FILE"

  # Neo4j binary dump (offline) — the RESTORE path. Stop the container, dump the
  # data volume from a throwaway container running as root (bypass entrypoint so
  # it keeps root and can write the host-owned bind mount), chown the dump back,
  # then restart. rm -f first: --overwrite-destination writes in place but does
  # not truncate, so a smaller dump over a larger old one leaves a corrupt tail.
  echo "[*] Neo4j: binary dump -> $target/neo4j.dump"
  local img abs_target
  img="$(neo_image)"
  abs_target="$(cd "$target" && pwd)"
  docker stop "$NEO_CONTAINER" >/dev/null
  trap neo_start_safety EXIT
  docker run --rm --user root --entrypoint sh \
    --volumes-from "$NEO_CONTAINER" \
    -v "$abs_target:/mm-backup" \
    "$img" \
    -c "rm -f '/mm-backup/$NEO_DB.dump' \
        && neo4j-admin database dump '$NEO_DB' --to-path=/mm-backup --overwrite-destination=true \
        && chown ${HOST_UID}:${HOST_GID} '/mm-backup/$NEO_DB.dump'"
  docker start "$NEO_CONTAINER" >/dev/null
  trap - EXIT
  [ "$NEO_DB" = "neo4j" ] || mv "$target/$NEO_DB.dump" "$target/neo4j.dump"

  for f in $CONFIG_FILES; do
    if [ -f "$REPO_ROOT/$f" ]; then
      cp "$REPO_ROOT/$f" "$target/config/"
      echo "[*] config: $f"
    fi
  done

  local files="postgres.dump neo4j.dump neo4j.cypher config/"
  if [ -f "$REPO_ROOT/$LLM_CACHE" ]; then
    # Server is down, so a cold copy of the db plus its WAL/SHM sidecars is
    # consistent (SQLite recovers the WAL on next open).
    for ext in "" "-wal" "-shm"; do
      [ -f "$REPO_ROOT/$LLM_CACHE$ext" ] && cp "$REPO_ROOT/$LLM_CACHE$ext" "$target/"
    done
    files="$files $LLM_CACHE"
    echo "[*] llm cache: $LLM_CACHE"
  fi

  {
    echo "created: $ts"
    echo "pg_container: $PG_CONTAINER  pg_db: $PG_DB"
    echo "neo_container: $NEO_CONTAINER  neo_db: $NEO_DB"
    echo "files: $files"
  } > "$target/MANIFEST.txt"

  echo "[✓] backup complete -> $target"
  ls -lh "$target"
}

restore() {
  local src="${1:-}"
  [ -n "$src" ] || { usage; exit 1; }
  [ -f "$src/postgres.dump" ] || die "missing $src/postgres.dump"
  [ -f "$src/neo4j.dump" ]    || die "missing $src/neo4j.dump (binary dump is the restore path)"
  require_running "$PG_CONTAINER"
  require_running "$NEO_CONTAINER"
  ensure_server_down

  if [ "$FORCE" != "1" ]; then
    echo "This OVERWRITES the live Postgres ($PG_DB) and Neo4j ($NEO_DB) data from $src."
    read -r -p "Continue? [y/N] " a
    case "$a" in [yY]|[yY][eE][sS]) ;; *) echo "Aborted."; exit 1 ;; esac
  fi

  # Drop BOTH schemas the dump populates (public + metadata). Dropping only
  # public left 'CREATE SCHEMA metadata' in the dump failing with "already
  # exists", which (under set -e) aborted the whole restore. public is recreated
  # here (pg_dump omits it); metadata is recreated by the restore itself.
  echo "[*] Postgres: drop schemas + pg_restore"
  docker cp "$src/postgres.dump" "$PG_CONTAINER:/tmp/postgres.dump"
  docker exec "$PG_CONTAINER" psql -U "$PG_USER" -d "$PG_DB" -v ON_ERROR_STOP=1 \
    -c "DROP SCHEMA IF EXISTS public CASCADE; DROP SCHEMA IF EXISTS metadata CASCADE; CREATE SCHEMA public;"
  docker exec "$PG_CONTAINER" pg_restore --no-owner -U "$PG_USER" -d "$PG_DB" /tmp/postgres.dump
  docker exec "$PG_CONTAINER" rm -f /tmp/postgres.dump

  # Neo4j binary load (offline) — stop, load into the shared /data volume from a
  # throwaway root container, chown /data back to the Neo4j uid so the DBMS can
  # start (the load ran as root), then restart.
  echo "[*] Neo4j: binary load from $src/neo4j.dump"
  local img abs_src
  img="$(neo_image)"
  abs_src="$(cd "$src" && pwd)"
  docker stop "$NEO_CONTAINER" >/dev/null
  trap neo_start_safety EXIT
  docker run --rm --user root --entrypoint sh \
    --volumes-from "$NEO_CONTAINER" \
    -v "$abs_src:/mm-backup:ro" \
    "$img" \
    -c "cp '/mm-backup/neo4j.dump' '/tmp/$NEO_DB.dump' \
        && neo4j-admin database load '$NEO_DB' --from-path=/tmp --overwrite-destination=true \
        && chown -R ${NEO_UID}:${NEO_GID} /data"
  docker start "$NEO_CONTAINER" >/dev/null
  trap - EXIT

  if [ -f "$src/$LLM_CACHE" ]; then
    for ext in "" "-wal" "-shm"; do
      [ -f "$src/$LLM_CACHE$ext" ] && cp "$src/$LLM_CACHE$ext" "$REPO_ROOT/"
    done
    echo "[*] llm cache: restored $LLM_CACHE -> repo root"
  fi

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
