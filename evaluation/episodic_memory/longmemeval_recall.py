"""Score retrieval recall of LongMemEval gold turns from search output.

Recall = gold (``has_answer``) turns retrieved / total gold turns, pooled over
all questions (LongMemEval_M has 896 gold turns across 500 questions), plus a
per-question-type breakdown. This is the "Overall Recall" metric reported in
PR #1475. Needs search output from ``longmemeval_search.py`` (it records
``retrieved_uids``) over a graph ingested with deterministic
``<question>:<session>:<turn>`` episode uids.

Usage:
    python longmemeval_recall.py --data-path longmemeval_m_cleaned.json \\
        search_vector_k50.json search_append_k50.json ...
"""

import argparse
import json
from collections import defaultdict

from longmemeval_models import iter_longmemeval_dataset


def load_gold(data_path: str) -> dict[str, tuple[str, set[str]]]:
    """Map question_id -> (question_type, {"<session>:<turn>", ...})."""
    return {
        item.question_id: (item.question_type.value, set(item.answer_turn_indices))
        for item in iter_longmemeval_dataset(data_path)
    }


def score(results: list[dict], gold: dict[str, tuple[str, set[str]]]) -> dict:
    hits: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    retrieved_counts = []
    for result in results:
        qid = result["question_id"]
        qtype, gold_turns = gold[qid]
        prefix = f"{qid}:"
        uids = result["retrieved_uids"]
        if uids and not all(uid.startswith(prefix) for uid in uids):
            raise ValueError(
                f"{qid}: retrieved uids are not '<question>:<session>:<turn>'; "
                "re-ingest with the current longmemeval_ingest.py"
            )
        retrieved = {uid.removeprefix(prefix) for uid in uids}
        found = len(gold_turns & retrieved)
        for key in (qtype, "overall"):
            hits[key] += found
            totals[key] += len(gold_turns)
        retrieved_counts.append(len(uids))
    return {
        "questions": len(results),
        "mean_retrieved": sum(retrieved_counts) / max(len(retrieved_counts), 1),
        "recall": {
            key: (
                hits[key] / totals[key] if totals[key] else None,
                hits[key],
                totals[key],
            )
            for key in sorted(totals, key=lambda k: (k != "overall", k))
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="LongMemEval dataset")
    parser.add_argument("search_paths", nargs="+", help="longmemeval_search.py outputs")
    args = parser.parse_args()

    gold = load_gold(args.data_path)
    for path in args.search_paths:
        with open(path) as f:
            report = score(json.load(f), gold)
        print(
            f"== {path}: {report['questions']} questions, "
            f"mean {report['mean_retrieved']:.1f} episodes retrieved"
        )
        for key, (recall, found, total) in report["recall"].items():
            shown = f"{recall:.4f}" if recall is not None else "  n/a "
            print(f"  {key:<28} {shown}  ({found}/{total})")


if __name__ == "__main__":
    main()
