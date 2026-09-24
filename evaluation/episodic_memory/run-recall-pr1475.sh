#!/usr/bin/env bash
# Reproduce PR #1475's LongMemEval_M retrieval-recall table (896 gold turns):
#   Vector top-k 50/60/70/80 vs Hybrid (Vector + FTS append-10) top-k 50.
# Their setup, as far as the PR states or implies: Qwen3-Embedding-4B, server
# defaults (no sentence chunking, rrf-hybrid identity+BM25 reranker, index
# threshold 1000 -> per-question HNSW vector index), append-10 fusion.
# No answer/judge LLM is needed (recall only). Needs llm_cache_qwen.db here
# (else every derivative is re-embedded via DeepInfra) and EMBEDDING_API_KEY
# in .env for any cache misses.
#   tmux new -s recall './run-recall-pr1475.sh'
# Env: NO_VECTOR_INDEX=1 -> exact brute-force search instead of HNSW (bounded
#      Neo4j heap, and what our earlier no-index runs used).
#      SKIP_INGEST=1     -> reuse an already-ingested graph.
set -euo pipefail
cd "$(dirname "$0")"
set -a; source .env; set +a
UV=~/.local/bin/uv
DATA=~/data/longmemeval_m_cleaned.json
Q=(--embedding-model Qwen/Qwen3-Embedding-4B --embedding-dimensions 2560
   --embedding-base-url https://api.deepinfra.com/v1/openai
   --embedding-cache ./llm_cache_qwen.db)
[ -f llm_cache_qwen.db ] || echo "WARN: llm_cache_qwen.db missing — full re-embed via DeepInfra"

echo "[1/4] waiting for Neo4j bolt..."
until (echo > /dev/tcp/localhost/7687) 2>/dev/null; do sleep 3; done

if [ "${SKIP_INGEST:-0}" != 1 ]; then
  INDEX_ARGS=(--index-threshold 1000)
  [ "${NO_VECTOR_INDEX:-0}" = 1 ] && INDEX_ARGS+=(--no-vector-index)
  echo "[2/4] ingest (no sentence chunking, ${INDEX_ARGS[*]})"
  $UV run python longmemeval_ingest.py --data-path "$DATA" --no-sentence-chunking \
    "${INDEX_ARGS[@]}" "${Q[@]}" 2>&1 | tee ~/ingest.log
fi

echo "[3/4] backfill FTS indexes (ingest-time creation is fire-and-forget)"
$UV run python backfill_fts_indexes.py 2>&1 | tee ~/backfill.log

echo "[4/4] retrieval-only searches"
S=(--data-path "$DATA" --retrieval-only "${Q[@]}")
OUT=()
for k in 50 60 70 80; do
  $UV run python longmemeval_search.py "${S[@]}" --reranker rrf-bm25 --top-k $k \
    --target-path recall_vector_k$k.json > ~/recall_vector_k$k.log 2>&1
  OUT+=("recall_vector_k$k.json")
done
$UV run python longmemeval_search.py "${S[@]}" --reranker rrf-bm25 --top-k 50 \
  --use-fts --fusion append --append-n 10 \
  --target-path recall_append_k50.json > ~/recall_append_k50.log 2>&1
OUT+=(recall_append_k50.json)
# Same pair without the BM25 reranker, in case the author's config differed.
for arm in vector append; do
  extra=(); [ $arm = append ] && extra=(--use-fts --fusion append --append-n 10)
  $UV run python longmemeval_search.py "${S[@]}" --reranker identity --top-k 50 \
    "${extra[@]}" --target-path recall_${arm}_k50_identity.json \
    > ~/recall_${arm}_k50_identity.log 2>&1
  OUT+=("recall_${arm}_k50_identity.json")
done

$UV run python longmemeval_recall.py --data-path "$DATA" "${OUT[@]}" | tee ~/RESULTS_RECALL.txt
echo "PR #1475 reported: vector k50 0.8811 | k60 0.9006 | k70 0.8945 | k80 0.9010 | hybrid k50 0.9108" \
  | tee -a ~/RESULTS_RECALL.txt
