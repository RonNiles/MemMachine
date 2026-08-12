import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", type=str, default="evaluation.json")
args = parser.parse_args()

# Canonical display order; only categories actually present are printed, so a
# subset that doesn't cover every question type won't crash.
CANONICAL = [
    "overall",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
]

# Load the evaluation metrics data
with open(args.data_path, "r") as f:
    data = json.load(f)

ordered = [c for c in CANONICAL if c in data] + [
    c for c in data if c not in CANONICAL
]
for category in ordered:
    entry = data[category]
    if not isinstance(entry, dict) or "llm_score" not in entry:
        continue
    count = entry.get("count")
    suffix = f"  (n={count})" if count is not None else ""
    print(f"{category}: {entry['llm_score']:.4f}{suffix}")
