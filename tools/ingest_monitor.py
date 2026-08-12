#!/usr/bin/env python3
"""Visualize ingestion rate and API latency from the live llm_cache.db.

Primary source is llm_cache.db (read-only): every cold LLM/embedding call
writes a `created_at` timestamp and the real `latency_ms` (which includes the
provider's internal retry/backoff — so rate-limiting shows up as rising
latency). Postgres `set_ingested_history` is used only for the authoritative
remaining count that drives the ETA.

Usage:
  python tools/ingest_monitor.py                 # one-shot report
  python tools/ingest_monitor.py --watch 15      # refresh every 15s
  python tools/ingest_monitor.py --window 30     # only last 30 min of buckets
  python tools/ingest_monitor.py --full-corpus 100000   # extrapolate to full corpus
  python tools/ingest_monitor.py --no-pg --pending 3593 # skip Postgres, supply remaining

Read-only throughout: the cache is opened with mode=ro and Postgres is a
SELECT via `docker exec`. Safe to run against an active ingestion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime

HISTORY_RE = re.compile(r"<HISTORY>\s*(.*?)\s*</HISTORY>", re.DOTALL)


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _user_text(request_json: str) -> str:
    try:
        d = json.loads(request_json)
    except (ValueError, TypeError):
        return ""
    u = d.get("user") or d.get("user_prompt") or ""
    return json.dumps(u) if isinstance(u, list) else str(u)


def read_cache(db_path: str):
    """Return (llm_rows, emb_rows). Each row: dict(ts, lat, kind[, msg])."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        llm = []
        for r in con.execute(
            "SELECT created_at, latency_ms, request_json, "
            "LENGTH(request_json) + LENGTH(COALESCE(response_json, '')) AS chars "
            "FROM llm_cache"
        ):
            user = _user_text(r["request_json"])
            is_extraction = "<OLD_PROFILE>" in user
            msg = None
            if is_extraction:
                m = HISTORY_RE.search(user)
                if m:
                    msg = hashlib.sha1(m.group(1).encode()).hexdigest()
            llm.append(
                {
                    "ts": _parse_ts(r["created_at"]),
                    "lat": r["latency_ms"],
                    "kind": "extraction" if is_extraction else "consolidation",
                    "msg": msg,
                    "chars": r["chars"] or 0,
                }
            )
        emb = [
            {"ts": _parse_ts(r["created_at"]), "lat": r["latency_ms"], "kind": "embed"}
            for r in con.execute("SELECT created_at, latency_ms FROM embedding_cache")
        ]
    finally:
        con.close()
    return llm, emb


def pg_remaining(container: str):
    """Return (done, pending, total, features) from Postgres, or None on failure.

    Counts are per distinct *history* (message), not per (set_id, history_id)
    row. The same message is shared across all of a user's sets, and the ETA
    rate is measured in distinct messages/min (the extraction prompt's
    <HISTORY> hash is set-independent), so counting raw rows would inflate the
    remaining work by the sets-per-history factor and overestimate the ETA. A
    history is "done" only once all its set rows are ingested.
    """
    sql = (
        "WITH h AS (SELECT history_id, bool_and(ingested) AS done "
        "FROM set_ingested_history GROUP BY history_id) "
        "SELECT (SELECT count(*) FILTER (WHERE done) FROM h), "
        "(SELECT count(*) FILTER (WHERE NOT done) FROM h), "
        "(SELECT count(*) FROM h), "
        "(SELECT count(*) FROM feature);"
    )
    try:
        proc = subprocess.run(
            ["docker", "exec", container, "psql", "-U", "memmachine",
             "-d", "memmachine", "-t", "-A", "-F", "|", "-c", sql],
            capture_output=True, text=True, timeout=15, check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    try:
        done, pending, total, feats = (int(x) for x in proc.stdout.strip().split("|"))
    except ValueError:
        return None
    return done, pending, total, feats


def _pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p))]


def _fmt_eta(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds:
        return "n/a"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def _pctstr(x, limit):
    return f"{100 * x / limit:.0f}%" if limit else "n/a"


def limits_panel(now, llm, args):
    """RPM / TPM / RPD panel for synchronous gpt-4o-mini calls.

    RPM/TPM use the last 15 min; calls-per-message uses the last 30 min for a
    steadier ratio; RPD uses the true rolling-24h request count (each cache row
    is one real API call). The RPD time-to-wall is a naive projection that
    assumes no requests age out of the 24h window, so it's the *soonest* the
    wall could arrive.
    """
    def win(mins):
        c = now.timestamp() - mins * 60
        rows = [r for r in llm if r["ts"].timestamp() >= c]
        msgs = {r["msg"] for r in rows if r["msg"]}
        toks = sum(r["chars"] for r in rows) / args.chars_per_token
        return rows, msgs, toks

    rows15, _, toks15 = win(15)
    rpm, tpm = len(rows15) / 15, toks15 / 15
    rows30, msgs30, _ = win(30)
    cpm = len(rows30) / len(msgs30) if msgs30 else 0.0

    used = sum(1 for r in llm if r["ts"].timestamp() >= now.timestamp() - 86400)
    remaining = max(0, args.rpd - used)
    sustainable = args.rpd / 1440.0  # req/min that exactly fills RPD over 24h

    lines = ["", "rate limits (gpt-4o-mini, synchronous — Batch TPD does not apply):"]
    lines.append(f"  RPM: {rpm:8.1f} / {args.rpm_limit:,}   "
                 f"({_pctstr(rpm, args.rpm_limit)})")
    lines.append(f"  TPM: {tpm / 1e3:7.1f}K / {args.tpm_limit // 1000:,}K "
                 f"({_pctstr(tpm, args.tpm_limit)})  ~{args.chars_per_token:.0f} chars/token est")
    lines.append(f"  RPD (rolling 24h): {used:,} / {args.rpd:,} used "
                 f"({_pctstr(used, args.rpd)})  |  {remaining:,} requests left")
    if cpm > 0:
        lines.append(f"       calls/msg {cpm:.2f}  ->  capacity ~{int(args.rpd / cpm):,} "
                     f"msgs/day  |  ~{int(remaining / cpm):,} msgs left in today's budget")
    if rpm > sustainable:
        lines.append(f"       {rpm:.1f} req/min is ABOVE the {sustainable:.1f}/min sustainable "
                     f"rate -> RPD wall in ~{remaining / (rpm * 60):.1f} h (soonest; no age-out)")
    elif rpm > 0:
        lines.append(f"       {rpm:.1f} req/min is under the {sustainable:.1f}/min sustainable "
                     f"rate -> RPD not hit at this pace")
    return lines


def render(args) -> str:
    now = datetime.now(UTC)
    llm, emb = read_cache(args.db)
    if not llm and not emb:
        return "llm_cache.db has no rows yet."

    cutoff = now.timestamp() - args.window * 60
    bucket = args.bucket
    out = []

    last_ts = max((r["ts"] for r in llm + emb), default=None)
    stale = (now - last_ts).total_seconds() if last_ts else 0
    out.append(f"== ingestion monitor @ {now:%Y-%m-%d %H:%M:%S}Z  "
               f"(last call {int(stale)}s ago)  window={args.window}m ==")

    # ---- per-bucket aggregation (within window) ----
    rows = defaultdict(lambda: {"ext": 0, "con": 0, "emb": 0,
                                "chat_lat": [], "emb_lat": [], "msgs": set()})
    for r in llm:
        if r["ts"].timestamp() < cutoff:
            continue
        b = int(r["ts"].timestamp() // bucket) * bucket
        rows[b]["ext" if r["kind"] == "extraction" else "con"] += 1
        rows[b]["chat_lat"].append(r["lat"])
        if r["msg"]:
            rows[b]["msgs"].add(r["msg"])
    for r in emb:
        if r["ts"].timestamp() < cutoff:
            continue
        b = int(r["ts"].timestamp() // bucket) * bucket
        rows[b]["emb"] += 1
        rows[b]["emb_lat"].append(r["lat"])

    if rows:
        maxcalls = max((d["ext"] + d["con"] for d in rows.values()), default=1) or 1
        out.append("")
        out.append(f"{'time':>8} {'msg':>4} {'extr':>4} {'cons':>4} {'emb':>4} "
                   f"{'chat p50/p90/max ms':>21}  rate(calls/" + f"{bucket}s)")
        for b in sorted(rows):
            d = rows[b]
            calls = d["ext"] + d["con"]
            p50, p90, mx = (_pct(d["chat_lat"], q) for q in (0.5, 0.9, 1.0))
            bar = "█" * int(20 * calls / maxcalls)
            flag = " ⚠RL" if mx >= args.slow_ms else ""
            t = datetime.fromtimestamp(b, UTC).strftime("%H:%M")
            out.append(f"{t:>8} {len(d['msgs']):>4} {d['ext']:>4} {d['con']:>4} "
                       f"{d['emb']:>4} {p50:>6.0f}/{p90:>6.0f}/{mx:>6.0f}  {bar}{flag}")

    # ---- throughput over recent sub-windows (messages/min via distinct HISTORY) ----
    out.append("")
    out.append("throughput (distinct messages first-seen):")
    for mins in (5, 15, 30):
        c = now.timestamp() - mins * 60
        msgs = {r["msg"] for r in llm if r["msg"] and r["ts"].timestamp() >= c}
        rate = len(msgs) / mins
        out.append(f"  last {mins:>2}m: {len(msgs):>4} msgs  = {rate:6.1f} msg/min  "
                   f"({rate * 60:6.0f}/hr)")

    # recent rate for ETA: use the longest *fully populated* interval up to 15m,
    # measured back from now. Stale rows from a prior run can sit far in the
    # past; keying the window off the earliest row would divide a recent burst
    # by a mostly-idle 15m and badly inflate the ETA. Instead, walk back from
    # now and stop at the first idle gap longer than --eta-gap minutes, so only
    # contiguous recent activity sets the window.
    first_seen = {}
    for r in llm:
        if r["msg"]:
            ts = r["ts"].timestamp()
            first_seen[r["msg"]] = min(ts, first_seen.get(r["msg"], ts))

    def rate_per_min(mins):
        c = now.timestamp() - mins * 60
        return sum(1 for ts in first_seen.values() if ts >= c) / mins

    gap = args.eta_gap * 60
    prev = now.timestamp()
    window_start = prev
    for ts in sorted(first_seen.values(), reverse=True):
        if prev - ts > gap:
            break
        window_start, prev = ts, ts
    eta_window = min((now.timestamp() - window_start) / 60, 15)
    # within 15m: use the contiguous populated span; else fall back wider
    eta_rate = (rate_per_min(eta_window) if eta_window > 0 else 0) \
        or rate_per_min(30) or rate_per_min(60)

    # ---- latency / rate-limit summary (last window) ----
    recent_chat = [r["lat"] for r in llm if r["ts"].timestamp() >= cutoff]
    recent_emb = [r["lat"] for r in emb if r["ts"].timestamp() >= cutoff]
    if recent_chat:
        slow = sum(1 for x in recent_chat if x >= args.slow_ms)
        out.append("")
        out.append(f"latency last {args.window}m  chat: p50={_pct(recent_chat, .5):.0f} "
                   f"p90={_pct(recent_chat, .9):.0f} p99={_pct(recent_chat, .99):.0f} "
                   f"max={max(recent_chat):.0f} ms | "
                   f"{slow}/{len(recent_chat)} calls >= {args.slow_ms}ms "
                   f"({100 * slow / len(recent_chat):.0f}% — rate-limit/backoff)")
    if recent_emb:
        out.append(f"             embed: p50={_pct(recent_emb, .5):.0f} "
                   f"p90={_pct(recent_emb, .9):.0f} max={max(recent_emb):.0f} ms")

    # ---- rate-limit panel (RPM / TPM / RPD) ----
    out += limits_panel(now, llm, args)

    # ---- ETA ----
    out.append("")
    pending = args.pending
    pg = None if args.no_pg else pg_remaining(args.pg_container)
    if pg:
        done, pg_pending, total, feats = pg
        out.append(f"progress (postgres): {done}/{total} msgs ingested, "
                   f"{pg_pending} pending | {feats} profile features")
        if pending is None:
            pending = pg_pending
    if pending is None:
        out.append("ETA: supply --pending N or enable Postgres for remaining count.")
    elif eta_rate <= 0:
        out.append(f"ETA: {pending} remaining but no recent message activity to rate.")
    else:
        eta = pending / eta_rate * 60
        out.append(f"ETA (this queue): {pending} msgs / {eta_rate:.1f} msg/min "
                   f"= {_fmt_eta(eta)}  (done ~{now.replace(microsecond=0)} + {_fmt_eta(eta)})")
        if args.full_corpus:
            full = args.full_corpus / eta_rate * 60
            out.append(f"full corpus ({args.full_corpus} msgs) at this rate: "
                       f"{_fmt_eta(full)} wall-clock")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="llm_cache.db")
    ap.add_argument("--window", type=int, default=60, help="minutes of buckets to show")
    ap.add_argument("--bucket", type=int, default=60, help="bucket size in seconds")
    ap.add_argument("--slow-ms", type=float, default=10000,
                    help="latency >= this is flagged as rate-limit/backoff")
    ap.add_argument("--pending", type=int, default=None,
                    help="remaining messages (overrides Postgres)")
    ap.add_argument("--full-corpus", type=int, default=None,
                    help="extrapolate wall-clock for a full corpus of this many msgs")
    ap.add_argument("--eta-gap", type=float, default=3.0,
                    help="idle gap (min) that ends the contiguous ETA rate window")
    ap.add_argument("--no-pg", action="store_true", help="do not query Postgres")
    ap.add_argument("--pg-container", default="memmachine-postgres-dev")
    ap.add_argument("--watch", type=float, default=0, help="refresh every N seconds")
    ap.add_argument("--rpd", type=int, default=10000, help="requests-per-day limit")
    ap.add_argument("--rpm-limit", type=int, default=500, help="requests-per-minute limit")
    ap.add_argument("--tpm-limit", type=int, default=200000, help="tokens-per-minute limit")
    ap.add_argument("--chars-per-token", type=float, default=4.0,
                    help="chars/token estimate for TPM (usage not stored by the API path)")
    args = ap.parse_args()

    if args.watch:
        try:
            while True:
                sys.stdout.write("\033[2J\033[H")  # clear screen
                print(render(args), flush=True)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            pass
    else:
        print(render(args))


if __name__ == "__main__":
    main()
