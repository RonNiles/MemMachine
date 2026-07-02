#!/usr/bin/env bash
# Start / stop the Postgres + Neo4j containers MemMachine needs, without docker-compose.
# Credentials match sample_configs/env.dockercompose so the same configuration.yml works.
set -euo pipefail

PG_NAME="memmachine-postgres-dev"
NEO_NAME="memmachine-neo4j-dev"

PG_USER="memmachine"
PG_PASS="memmachine_password"
PG_DB="memmachine"

# Host port to map Postgres to. Override if 5432 is already taken on your
# machine, e.g. `PG_PORT=5433 ./scripts/dev-db.sh up`. The container always
# listens on 5432 internally; only the host-side mapping changes. Make sure
# your .env sets POSTGRES_PORT to the same value.
PG_PORT="${PG_PORT:-5432}"

NEO_USER="neo4j"
NEO_PASS="neo4j_password"

usage() {
  cat <<EOF
Usage: $0 {up|down|status|logs|psql|cypher|reset}

  up      Start Postgres (pgvector) and Neo4j containers (idempotent).
  down    Stop and remove both containers. Named volumes are preserved.
  status  Show docker ps for the two containers.
  logs    Tail logs for both containers (Ctrl-C to exit).
  psql    Open a psql shell inside the Postgres container.
  cypher  Open a cypher-shell inside the Neo4j container.
  reset   Delete the data volumes (mm-pg-data, mm-neo4j-data) for a clean
          slate. Containers must not be running; prompts for confirmation.
EOF
}

start_one() {
  local name="$1"
  if docker ps --format '{{.Names}}' | grep -qx "$name"; then
    echo "[=] $name already running"
    return
  fi
  if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
    echo "[+] starting existing $name"
    docker start "$name" >/dev/null
    return
  fi
  return 1  # caller should docker run
}

up() {
  if ! start_one "$PG_NAME"; then
    echo "[+] creating $PG_NAME (pgvector/pgvector:pg16)"
    docker run -d \
      --name "$PG_NAME" \
      -p "${PG_PORT}:5432" \
      -e POSTGRES_USER="$PG_USER" \
      -e POSTGRES_PASSWORD="$PG_PASS" \
      -e POSTGRES_DB="$PG_DB" \
      -v mm-pg-data:/var/lib/postgresql/data \
      pgvector/pgvector:pg16 >/dev/null
  fi

  if ! start_one "$NEO_NAME"; then
    echo "[+] creating $NEO_NAME (neo4j:5.23-community)"
    docker run -d \
      --name "$NEO_NAME" \
      -p 7474:7474 -p 7687:7687 \
      -e NEO4J_AUTH="${NEO_USER}/${NEO_PASS}" \
      -e NEO4J_PLUGINS='["apoc","graph-data-science"]' \
      -e NEO4J_apoc_export_file_enabled=true \
      -e NEO4J_server_memory_heap_initial__size=512m \
      -e NEO4J_server_memory_heap_max__size=1G \
      -v mm-neo4j-data:/data \
      neo4j:5.23-community >/dev/null
  fi

  echo "[*] Postgres -> localhost:${PG_PORT}  user=$PG_USER db=$PG_DB"
  echo "[*] Neo4j    -> bolt://localhost:7687  http://localhost:7474  user=$NEO_USER"
  echo
  echo "Run migrations once Postgres is healthy:"
  echo "  uv run alembic upgrade head"
}

down() {
  for n in "$PG_NAME" "$NEO_NAME"; do
    if docker ps -a --format '{{.Names}}' | grep -qx "$n"; then
      echo "[-] removing $n"
      docker rm -f "$n" >/dev/null
    fi
  done
}

reset() {
  for n in "$PG_NAME" "$NEO_NAME"; do
    if docker ps --format '{{.Names}}' | grep -qx "$n"; then
      echo "[!] $n is still running — run '$0 down' first" >&2
      exit 1
    fi
  done

  echo "This will permanently delete the data volumes mm-pg-data and mm-neo4j-data."
  read -r -p "Continue? [y/N] " answer
  case "$answer" in
    [yY]|[yY][eE][sS]) ;;
    *) echo "Aborted."; exit 1 ;;
  esac

  for v in mm-pg-data mm-neo4j-data; do
    if docker volume inspect "$v" >/dev/null 2>&1; then
      echo "[-] removing volume $v"
      docker volume rm "$v" >/dev/null
    else
      echo "[=] volume $v does not exist"
    fi
  done
}

status() { docker ps -a --filter "name=$PG_NAME" --filter "name=$NEO_NAME"; }
logs()   { docker logs -f "$PG_NAME" & docker logs -f "$NEO_NAME"; wait; }
psql()   { docker exec -it "$PG_NAME" psql -U "$PG_USER" -d "$PG_DB"; }
cypher() { docker exec -it "$NEO_NAME" cypher-shell -u "$NEO_USER" -p "$NEO_PASS"; }

case "${1:-}" in
  up)     up ;;
  down)   down ;;
  status) status ;;
  logs)   logs ;;
  psql)   psql ;;
  cypher) cypher ;;
  reset)  reset ;;
  *)      usage; exit 1 ;;
esac
