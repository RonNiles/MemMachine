#!/usr/bin/env python3
"""Analyze a MemMachine stage-timing CSV to find what slows ingestion over time.

The server writes the CSV when ``MEMMACHINE_STAGE_TIMING_CSV`` is set (see
``memmachine_server.common.stage_timing``). Each snapshot records, per stage,
the cumulative and per-window (delta) seconds and call counts, plus progress
counters. This tool converts that into **per-episode cost by stage over time**
and ranks stages by how much their per-episode cost grows as the databases
fill — i.e. which stage is responsible for the superlinear slowdown.

Usage::

    python tools/stage_timing_report.py stage_timing.csv
    python tools/stage_timing_report.py stage_timing.csv --timeline --top 12

Reads only the CSV; needs no database access and no third-party packages.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, field

# Stages that aggregate other stages; shown but flagged so nobody sums across
# levels (e.g. semantic.process_set already contains the semantic.* leaves,
# and semantic.consolidate_category contains semantic.consolidate_fetch).
AGGREGATE_STAGES = {
    "semantic.process_set",
    "semantic.consolidate_category",
}

PRIMARY_COUNTER = "messages_ingested"


@dataclass
class Window:
    """One inter-snapshot window."""

    snapshot: int
    elapsed_sec: float
    primary_cum: int
    primary_delta: int
    # stage -> (delta_seconds, delta_calls)
    stages: dict[str, tuple[float, int]] = field(default_factory=dict)


def load_windows(path: str) -> list[Window]:
    by_snap: dict[int, Window] = {}
    counters_delta: dict[int, dict[str, int]] = defaultdict(dict)
    counters_cum: dict[int, dict[str, int]] = defaultdict(dict)
    meta: dict[int, tuple[float]] = {}

    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            snap = int(row["snapshot"])
            meta[snap] = (float(row["elapsed_sec"]),)
            if row["kind"] == "stage":
                by_snap.setdefault(
                    snap,
                    Window(snapshot=snap, elapsed_sec=float(row["elapsed_sec"]),
                           primary_cum=0, primary_delta=0),
                )
                by_snap[snap].stages[row["name"]] = (
                    float(row["delta_seconds"]),
                    int(row["delta_count"]),
                )
            elif row["kind"] == "counter":
                counters_delta[snap][row["name"]] = int(row["delta_count"])
                counters_cum[snap][row["name"]] = int(row["cum_count"])

    windows: list[Window] = []
    for snap in sorted(set(by_snap) | set(counters_cum)):
        w = by_snap.get(snap) or Window(
            snapshot=snap, elapsed_sec=meta.get(snap, (0.0,))[0],
            primary_cum=0, primary_delta=0,
        )
        w.primary_cum = counters_cum.get(snap, {}).get(PRIMARY_COUNTER, 0)
        w.primary_delta = counters_delta.get(snap, {}).get(PRIMARY_COUNTER, 0)
        windows.append(w)
    return windows


def per_msg_ms(delta_seconds: float, primary_delta: int) -> float | None:
    if primary_delta <= 0:
        return None
    return 1000.0 * delta_seconds / primary_delta


def print_growth_summary(windows: list[Window], top: int) -> None:
    scored = [w for w in windows if w.primary_delta > 0]
    if len(scored) < 2:
        print("Not enough windows with progress to compute growth. "
              "Lower MEMMACHINE_STAGE_TIMING_EVERY or run longer.")
        return

    # Average the first and last ~20% of progress-bearing windows for stability.
    span = max(1, len(scored) // 5)
    early, late = scored[:span], scored[-span:]

    def avg_ms(win_list: list[Window], stage: str) -> float | None:
        vals = [
            v for w in win_list
            if (v := per_msg_ms(w.stages.get(stage, (0.0, 0))[0], w.primary_delta))
            is not None
        ]
        return sum(vals) / len(vals) if vals else None

    all_stages = sorted({s for w in scored for s in w.stages})
    rows = []
    for stage in all_stages:
        early_ms, late_ms = avg_ms(early, stage), avg_ms(late, stage)
        if early_ms is None or late_ms is None:
            continue
        growth = (late_ms / early_ms) if early_ms > 0 else float("inf")
        rows.append((late_ms - early_ms, growth, early_ms, late_ms, stage))

    rows.sort(reverse=True)  # by absolute per-episode increase (ms)

    lo_n = f"~{scored[0].primary_cum}"
    hi_n = f"~{scored[-1].primary_cum}"
    print(f"\n=== Per-episode cost growth by stage "
          f"(early n={lo_n} vs late n={hi_n}) ===")
    print(f"{'stage':38} {'early ms':>9} {'late ms':>9} {'growth':>7} "
          f"{'Δ ms/ep':>9}")
    print("-" * 76)
    for delta, growth, early_ms, late_ms, stage in rows[:top]:
        flag = " (agg)" if stage in AGGREGATE_STAGES else ""
        g = "inf" if growth == float("inf") else f"{growth:5.1f}x"
        print(f"{stage + flag:38} {early_ms:9.2f} {late_ms:9.2f} {g:>7} "
              f"{delta:9.2f}")
    print("\nRanked by Δ ms/episode (late-early). Stages growing with N drive "
          "the O(N^2) curve.\n'(agg)' aggregates its children — don't sum it "
          "with them.")


def print_timeline(windows: list[Window], top: int) -> None:
    scored = [w for w in windows if w.primary_delta > 0]
    if not scored:
        print("No windows with progress recorded.")
        return
    # Choose the stages with the largest late-window per-episode cost to show.
    last = scored[-1]
    ranked = sorted(
        last.stages,
        key=lambda s: per_msg_ms(last.stages[s][0], last.primary_delta) or 0.0,
        reverse=True,
    )[:top]

    print("\n=== Timeline: per-episode ms by stage (columns = stages) ===")
    header = f"{'n_ingested':>11} {'elapsed_s':>9}  " + "  ".join(
        f"{s.split('.')[-1][:14]:>14}" for s in ranked
    )
    print(header)
    print("-" * len(header))
    for w in scored:
        cells = []
        for s in ranked:
            v = per_msg_ms(w.stages.get(s, (0.0, 0))[0], w.primary_delta)
            cells.append(f"{v:14.2f}" if v is not None else f"{'-':>14}")
        print(f"{w.primary_cum:11d} {w.elapsed_sec:9.1f}  " + "  ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_path", help="Path to the stage-timing CSV")
    ap.add_argument("--top", type=int, default=12, help="Stages to show")
    ap.add_argument("--timeline", action="store_true",
                    help="Also print the full per-window timeline")
    args = ap.parse_args()

    windows = load_windows(args.csv_path)
    if not windows:
        print("No snapshots found in CSV.")
        return

    total_ingested = max((w.primary_cum for w in windows), default=0)
    print(f"Loaded {len(windows)} snapshots; {total_ingested} episodes ingested; "
          f"{windows[-1].elapsed_sec:.0f}s elapsed.")

    print_growth_summary(windows, args.top)
    if args.timeline:
        print_timeline(windows, args.top)


if __name__ == "__main__":
    main()
