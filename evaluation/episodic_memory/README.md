# LoCoMo

## Tool-Specific Prerequisites

- Please ensure your `cfg.yml` file has been copied into your `episodic_memory` directory (`/memmachine/evaluation/episodic_memory/`) and renamed to `locomo_config.yaml`.


## Running the Benchmark

Ready to go? Follow these simple steps:

**A.** All commands should be run from their respective tool directory (default `evaluation/episodic_memory/`).

**B.** The path to your data file, `locomo10.json`, should be updated to match its location. By default, you can find it in `/memmachine/evaluation/data/`.

**C.** Once you have performed step 1 below, you can repeat the benchmark run by performing steps 2-4.  Once are you finished performing the benchmark, run step 5.

**Note:** For the recommended retrieval-agent benchmark workflow and
cross-benchmark command references, see `evaluation/README.md`.

### Step 1: Ingest a Conversation

First, let's add conversation data to MemMachine. This only needs to be done once per test run.
```sh
python locomo_ingest.py --data-path path/to/locomo10.json
```

### Step 2: Search the Conversation

Let's search through the data you just added.
```sh
python locomo_search.py --data-path path/to/locomo10.json --target-path results.json
```

### Step 3: Evaluate the Responses

Next, run a LoCoMo evaluation against the search results.
```sh
python locomo_evaluate.py --data-path results.json --target-path evaluation_metrics.json
```

### Step 4: Generate Your Final Score

Once the evaluation is complete, you can generate the final scores.
```sh
python generate_scores.py
```

The output will be a table in your shell showing the mean scores for each category and an overall score, like the example below:
```sh
Mean Scores Per Category:
          llm_score  count         type
category
1            0.8050    282    multi_hop
2            0.7259    321     temporal
3            0.6458     96  open_domain
4            0.9334    841   single_hop

Overall Mean Scores:
llm_score    0.8487
dtype: float64
```

### Step 5: Clean Up Your Data

When you're finished, you may want to delete the test data.
```sh
python locomo_delete.py --data-path path/to/locomo10.json
```

# LongMemEval

## Compatibility Note

The scripts use lower-level MemMachine server episodic memory directly without the same metadata and data structuring as the full MemMachine server.
As such, the data ingested will not be compatible with the full MemMachine server, and no config file is needed.

## Running the Benchmark

These steps work for any split; use `longmemeval_s_cleaned.json` or, for the
larger haystacks, `longmemeval_m_cleaned.json` (500 questions). **Note:** the M
split has much larger per-question haystacks than S, so expect substantially
more disk and ingestion time than the figures below (which are for S).

Get the LongMemEval dataset:
https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/tree/main

No config file is needed.

Set up Neo4j by any method.

> [! WARNING]
> We don't provide a way to clean up just the data ingested for LongMemEval, so we highly recommend against ingesting the data into a database with existing data.


> [! WARNING]
> As configured in the scripts, ~50GB of free space on disk is required. Ingestion will take ~1.5 hours. Required disk and ingestion time can be reduced by a factor of ~5x by setting `message_sentence_chunking=False` in the ingestion script, potentially with slightly lower scores (~1-2% lower).

The scripts run with **no reranker** (`IdentityReranker`, which preserves
retrieval order without reordering). Declarative memory still requires a
reranker object, so to use a real one, import and construct a different
`Reranker` in `longmemeval_ingest.py` / `longmemeval_search.py`.

Set the following environment variables:

- `NEO4J_URI`: URI for the Neo4j database

- `NEO4J_USERNAME`: as configured for the Neo4j database

- `NEO4J_PASSWORD`: as configured for the Neo4j database

- `OPENAI_API_KEY`: for embeddings, answer generation, and LLM-as-a-judge scoring

### Step 1: Ingest LongMemEval

```sh
python longmemeval_ingest.py --data-path path/to/longmemeval_s_cleaned.json
```

### Step 2: Query the Memory and Generate Responses to Questions

Search is vector-only by default. Add `--use-fts` to run **hybrid Vector + FTS**
retrieval (fused via Reciprocal Rank Fusion). To A/B the two, run it twice with
different `--target-path` files:

```sh
# Vector-only baseline
python longmemeval_search.py --data-path path/to/longmemeval_m_cleaned.json --target-path search_vector.json

# Hybrid Vector + FTS (RRF)
python longmemeval_search.py --data-path path/to/longmemeval_m_cleaned.json --target-path search_fts.json --use-fts
```

> **Before the `--use-fts` run**, create the full-text indexes on the ingested
> Derivative collections (ingestion does not create them reliably):
> ```sh
> python backfill_fts_indexes.py
> ```
> It is idempotent and safe to run once after ingestion.

It may be useful to direct stdout to a file, e.g. `... --target-path search_fts.json --use-fts > search_fts.out`.

### Step 3: Evaluate the Responses

```sh
python longmemeval_evaluate.py --data-path search.json --target-path eval.json
```

### Step 4: Print Scores

```sh
python lme_generate.py --data-path eval.json
```

```sh
overall: 0.9580
multi-session: 0.9323
temporal-reasoning: 0.9624
knowledge-update: 0.9359
single-session-user: 0.9857
single-session-assistant: 1.0000
single-session-preference: 0.9667
```
