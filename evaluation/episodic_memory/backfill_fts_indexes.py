#!/usr/bin/env python3
"""Create Neo4j FTS indexes for every Derivative collection after ingestion.

The hybrid-search (`--use-fts`) path queries a full-text index named
`fts_<sanitized_collection>_content` on each Derivative collection. During
ingestion those indexes are only created lazily (fire-and-forget) once a
collection crosses the range-index threshold, which is unreliable for a
benchmark run — especially LongMemEval, where each question is its own
collection. This script creates the index for every Derivative collection
deterministically and waits for them to come online.

Idempotent (`CREATE ... IF NOT EXISTS`). Run once after `longmemeval_ingest.py`
and before `longmemeval_search.py --use-fts`.

Reads connection from the same env vars the LongMemEval scripts use:
NEO4J_URI, NEO4J_USERNAME (or NEO4J_USER), NEO4J_PASSWORD.
"""

import os
import re
import sys

from neo4j import GraphDatabase

DERIVATIVE_LABEL_PREFIX = "SANITIZED_Derivative_u5f_"
CONTENT_PROPERTY = "SANITIZED_property_u5f_content"
SAFE_IDENT = re.compile(r"^[A-Za-z0-9_]+$")  # labels from db.labels() are sanitized

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or "neo4j"
PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j_password")


def main() -> int:
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    try:
        with driver.session() as session:
            labels = [
                rec["label"]
                for rec in session.run("CALL db.labels() YIELD label RETURN label")
                if rec["label"].startswith(DERIVATIVE_LABEL_PREFIX)
            ]
            if not labels:
                print(f"No labels starting with {DERIVATIVE_LABEL_PREFIX!r} found. "
                      "Did ingestion run against this Neo4j instance?")
                return 1

            print(f"Creating FTS indexes for {len(labels)} Derivative collection(s)...")
            created = 0
            for label in labels:
                if not SAFE_IDENT.match(label):
                    print(f"  !! skipping unsafe label {label!r}")
                    continue
                idx = f"fts_{label}_content"
                session.run(
                    f"CREATE FULLTEXT INDEX {idx} IF NOT EXISTS "
                    f"FOR (n:`{label}`) ON EACH [n.`{CONTENT_PROPERTY}`] "
                    "OPTIONS { indexConfig: { `fulltext.analyzer`: 'standard' } }"
                )
                created += 1

            print(f"Issued {created} index statement(s); waiting for them to come online...")
            session.run("CALL db.awaitIndexes(600)")

            online = next(iter(session.run(
                "SHOW FULLTEXT INDEXES YIELD name, state, populationPercent "
                "WHERE name STARTS WITH 'fts_' "
                "RETURN count(*) AS total, "
                "sum(CASE WHEN state='ONLINE' THEN 1 ELSE 0 END) AS online"
            )))
            print(f"FTS indexes: {online['online']}/{online['total']} ONLINE")
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
