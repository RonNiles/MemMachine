#!/usr/bin/env bash
# Backup / restore the MemMachine databases (Postgres + Neo4j).
#
# Run with the DB containers UP but memmachine-server DOWN, so there is no
# write activity and the dumps are consistent. The script refuses to run if it
# detects a running server (override with FORCE=1 / --force).
#
# Postgres : pg_dump custom-format archive (postgres.dump).
# Neo4j    : stop the container -> offline `neo4j-admin database dump` in a
#            throwaway container sharing the data volume -> start the container.
#            STOP/START DATABASE is Enterprise-only, so Community Edition must
#            take the whole DBMS offline for a consistent dump.
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
# uid:gid the Neo4j data files are owned by inside the container (official
# image uses 7474:7474). Used to chown /data back after an offline restore.
NEO_UID="${NEO_UID:-7474}"
NEO_GID="${NEO_GID:-7474}"
# Space-separated, resolved relative to the repo root.
CONFIG_FILES="${CONFIG_FILES:-configuration.yml .env}"
# SQLite LLM/embedding cache at the repo root; backed up if present.
LLM_CACHE="${LLM_CACHE:-llm_cache.db}"
FORCE="${FORCE:-0}"
# Verify each fresh Neo4j dump by test-loading it in a throwaway container
# (catches a corrupt/truncated archive at backup time). Set VERIFY=0 to skip.
VERIFY="${VERIFY:-1}"

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
  cat <<EOF
Usage: $0 backup  <target-dir> [--force]
       $0 restore <source-dir> [--force]

  backup   Dump Postgres + Neo4j and copy config files into <target-dir>.
  restore  OVERWRITE the live Postgres + Neo4j data from <source-dir>.

Both require the two DB containers running and memmachine-server stopped.
Env overrides: PG_CONTAINER NEO_CONTAINER PG_USER PG_DB
               NEO_DB NEO_UID NEO_GID CONFIG_FILES FORCE VERIFY
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

neo_image()         { docker inspect -f '{{.Config.Image}}' "$NEO_CONTAINER"; }
neo_start_safety()  { docker start "$NEO_CONTAINER" >/dev/null 2>&1 || true; }

# Test-load a just-created dump in a throwaway container with NO access to the
# live /data volume (its own ephemeral /data), proving the archive restores
# end-to-end. A truncated/corrupt dump makes neo4j-admin load exit non-zero.
# $1 = image, $2 = absolute dir holding '<NEO_DB>.dump'.
neo_verify_dump() {
  local img="$1" dir="$2"
  echo "[*] Neo4j: verifying dump (test-load in throwaway container)"
  docker run --rm --user root --entrypoint sh \
    -v "$dir:/mm-backup:ro" \
    "$img" \
    -c "cp '/mm-backup/$NEO_DB.dump' '/tmp/$NEO_DB.dump' \
        && neo4j-admin database load '$NEO_DB' --from-path=/tmp --overwrite-destination=true >/dev/null 2>&1" \
    || die "dump verification failed: '$dir/$NEO_DB.dump' is not a loadable archive (corrupt/truncated)."
}

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

  echo "[*] Neo4j (community): stop container, offline dump, start"
  local img abs_target
  img="$(neo_image)"
  abs_target="$(cd "$target" && pwd)"
  docker stop "$NEO_CONTAINER" >/dev/null
  trap neo_start_safety EXIT
  # Throwaway container shares the stopped container's /data volume; run as
  # root so it can read the 7474-owned store and write the host bind mount,
  # then chown the resulting dump back to the invoking user. --entrypoint sh
  # bypasses the image entrypoint, which would otherwise drop back to the
  # neo4j user and lose write access to the host-owned bind mount.
  #
  # rm -f the destination FIRST: neo4j-admin's --overwrite-destination writes
  # over the existing file in place but does NOT truncate it, so dumping a
  # smaller archive onto a larger pre-existing one (e.g. a prior backup into
  # the same dir) leaves the old tail past the new zstd frame -> a corrupt
  # archive that loads partway then fails with "Unknown frame descriptor".
  docker run --rm --user root --entrypoint sh \
    --volumes-from "$NEO_CONTAINER" \
    -v "$abs_target:/mm-backup" \
    "$img" \
    -c "rm -f '/mm-backup/$NEO_DB.dump' \
        && neo4j-admin database dump '$NEO_DB' --to-path=/mm-backup --overwrite-destination=true \
        && chown ${HOST_UID}:${HOST_GID} '/mm-backup/$NEO_DB.dump'"
  docker start "$NEO_CONTAINER" >/dev/null
  trap - EXIT
  [ "$VERIFY" = "1" ] && neo_verify_dump "$img" "$abs_target"
  [ "$NEO_DB" = "neo4j" ] || mv "$target/$NEO_DB.dump" "$target/neo4j.dump"

  for f in $CONFIG_FILES; do
    if [ -f "$REPO_ROOT/$f" ]; then
      cp "$REPO_ROOT/$f" "$target/config/"
      echo "[*] config: $f"
    fi
  done

  local files="postgres.dump neo4j.dump config/"
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

  echo "[*] Neo4j (community): stop container, offline load, start"
  local img abs_src
  img="$(neo_image)"
  abs_src="$(cd "$src" && pwd)"
  docker stop "$NEO_CONTAINER" >/dev/null
  trap neo_start_safety EXIT
  # Copy the dump to a writable path under the expected '<db>.dump' name, load
  # it into the shared /data volume, then chown /data back to the Neo4j uid so
  # the DBMS can start (the load ran as root).
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
