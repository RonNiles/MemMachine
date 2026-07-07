"""Opt-in, low-overhead per-stage timing for ingestion performance analysis.

This is a diagnostic aid for understanding why ingestion throughput degrades as
the databases grow. It accumulates wall-clock time and call counts per named
*stage* (e.g. ``semantic.consolidate_category``, ``vector_graph_store_neo4j.
add_edges``) and periodically writes a snapshot row per stage to a CSV. By
comparing the per-window (delta) seconds of each stage against the number of
episodes ingested in that window, an analyzer can see which stage's per-episode
cost grows with N — i.e. which stage is responsible for superlinear slowdown.

**Disabled by default.** Nothing is recorded and every entry point is a cheap
no-op unless the ``MEMMACHINE_STAGE_TIMING_CSV`` environment variable names an
output file. This makes it safe to leave the instrumentation permanently in hot
paths.

Environment variables:

- ``MEMMACHINE_STAGE_TIMING_CSV`` — output CSV path. Presence enables timing.
- ``MEMMACHINE_STAGE_TIMING_EVERY`` — write a snapshot every N units of the
  primary progress counter (default 100). Set to 0 to disable count-triggered
  snapshots.
- ``MEMMACHINE_STAGE_TIMING_INTERVAL_SEC`` — also write a snapshot at least this
  often in wall-clock seconds (default 30). Set to 0 to disable time-triggered
  snapshots.

The CSV is long-format — one row per (snapshot, metric):

    snapshot,wall_time,elapsed_sec,kind,name,status,cum_count,cum_seconds,delta_count,delta_seconds

``kind`` is ``stage`` (timed operation) or ``counter`` (progress counter such as
``episodes_ingested``); for counters the ``*_seconds`` columns are 0.
"""

from __future__ import annotations

import atexit
import csv
import os
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

# Primary progress counter that count-triggered snapshots key on. The semantic
# ingestion loop bumps this once per message it marks ingested, so it maps 1:1
# to the "episodes" axis the operator observes.
PRIMARY_COUNTER = "messages_ingested"

_CSV_PATH = os.environ.get("MEMMACHINE_STAGE_TIMING_CSV") or None
_ENABLED = _CSV_PATH is not None
_EVERY = int(os.environ.get("MEMMACHINE_STAGE_TIMING_EVERY", "100"))
_INTERVAL = float(os.environ.get("MEMMACHINE_STAGE_TIMING_INTERVAL_SEC", "30"))

_lock = threading.Lock()
# stage name -> [calls, seconds, errors]
_stages: dict[str, list[float]] = {}
# counter name -> cumulative value
_counters: dict[str, float] = {}
# cumulative values captured at the previous snapshot, for delta computation
_stages_at_last: dict[str, list[float]] = {}
_counters_at_last: dict[str, float] = {}

_snapshot_seq = 0
_start_mono: float | None = None
_last_snapshot_mono: float | None = None
_last_primary_bucket = 0

_fh: Any = None
_writer: Any = None

_CSV_HEADER = [
    "snapshot",
    "wall_time",
    "elapsed_sec",
    "kind",
    "name",
    "status",
    "cum_count",
    "cum_seconds",
    "delta_count",
    "delta_seconds",
]


def enabled() -> bool:
    """Return True when stage timing is active (output CSV configured)."""
    return _ENABLED


def _ensure_started_locked() -> None:
    global _fh, _writer, _start_mono, _last_snapshot_mono
    if _writer is not None or _CSV_PATH is None:
        return
    _start_mono = time.monotonic()
    _last_snapshot_mono = _start_mono
    # newline="" per csv module guidance; line-buffered so partial runs are
    # still readable if the process is killed mid-benchmark.
    _fh = open(_CSV_PATH, "w", newline="", buffering=1)  # noqa: SIM115, PTH123
    _writer = csv.writer(_fh)
    _writer.writerow(_CSV_HEADER)
    atexit.register(_final_snapshot)


def record(stage: str, elapsed: float, status: str = "ok") -> None:
    """Record one timed observation for ``stage`` (seconds). No-op if disabled."""
    if not _ENABLED:
        return
    with _lock:
        _ensure_started_locked()
        slot = _stages.get(stage)
        if slot is None:
            slot = [0.0, 0.0, 0.0]
            _stages[stage] = slot
        slot[0] += 1.0
        slot[1] += elapsed
        if status != "ok":
            slot[2] += 1.0
        _maybe_snapshot_locked()


def bump(counter: str = PRIMARY_COUNTER, n: float = 1.0) -> None:
    """Advance a progress counter (e.g. episodes ingested). No-op if disabled."""
    if not _ENABLED:
        return
    with _lock:
        _ensure_started_locked()
        _counters[counter] = _counters.get(counter, 0.0) + n
        _maybe_snapshot_locked()


@contextmanager
def timed(stage: str) -> Iterator[None]:
    """Sync context manager timing a block as ``stage``."""
    if not _ENABLED:
        yield
        return
    start = time.monotonic()
    status = "ok"
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        record(stage, time.monotonic() - start, status)


@asynccontextmanager
async def atimed(stage: str) -> AsyncIterator[None]:
    """Async context manager timing an awaited block as ``stage``."""
    if not _ENABLED:
        yield
        return
    start = time.monotonic()
    status = "ok"
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        record(stage, time.monotonic() - start, status)


def _maybe_snapshot_locked() -> None:
    global _last_primary_bucket
    now = time.monotonic()
    due = False
    if _INTERVAL > 0 and _last_snapshot_mono is not None:
        due = due or (now - _last_snapshot_mono) >= _INTERVAL
    if _EVERY > 0:
        bucket = int(_counters.get(PRIMARY_COUNTER, 0.0)) // _EVERY
        if bucket > _last_primary_bucket:
            _last_primary_bucket = bucket
            due = True
    if due:
        _write_snapshot_locked()


def _write_snapshot_locked() -> None:
    global _snapshot_seq, _last_snapshot_mono
    if _writer is None or _start_mono is None:
        return
    _snapshot_seq += 1
    wall = time.time()
    elapsed = time.monotonic() - _start_mono
    for name, (calls, seconds, _errors) in sorted(_stages.items()):
        prev = _stages_at_last.get(name, [0.0, 0.0, 0.0])
        _writer.writerow(
            [
                _snapshot_seq,
                f"{wall:.3f}",
                f"{elapsed:.3f}",
                "stage",
                name,
                "ok",
                int(calls),
                f"{seconds:.6f}",
                int(calls - prev[0]),
                f"{seconds - prev[1]:.6f}",
            ]
        )
        _stages_at_last[name] = [calls, seconds, _errors]
    for name, value in sorted(_counters.items()):
        prev_v = _counters_at_last.get(name, 0.0)
        _writer.writerow(
            [
                _snapshot_seq,
                f"{wall:.3f}",
                f"{elapsed:.3f}",
                "counter",
                name,
                "ok",
                int(value),
                "0",
                int(value - prev_v),
                "0",
            ]
        )
        _counters_at_last[name] = value
    _last_snapshot_mono = time.monotonic()


def snapshot() -> None:
    """Force-write a snapshot now (e.g. at end of a benchmark run)."""
    if not _ENABLED:
        return
    with _lock:
        _ensure_started_locked()
        _write_snapshot_locked()


def _final_snapshot() -> None:
    try:
        with _lock:
            _write_snapshot_locked()
            if _fh is not None:
                _fh.flush()
    except Exception:  # atexit best-effort; never fail on shutdown
        pass
