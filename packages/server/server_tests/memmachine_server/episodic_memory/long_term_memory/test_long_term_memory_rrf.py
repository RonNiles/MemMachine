"""Unit tests for RRF fusion in hybrid search (LongTermMemory._rrf_fuse).

Pure-logic tests — no Neo4j/Docker required, so this module is intentionally
NOT marked `integration`.
"""

from datetime import UTC, datetime

from memmachine_server.common.episode_store import Episode
from memmachine_server.episodic_memory.long_term_memory.long_term_memory import (
    LongTermMemory,
)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _ep(uid: str, content: str = "") -> Episode:
    return Episode(
        uid=uid,
        content=content or uid,
        session_key="s",
        created_at=_NOW,
        producer_id="p",
        producer_role="user",
    )


def _rrf(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def test_rrf_fuse_boosts_items_in_both_legs():
    # B appears in both legs; everything else in one leg only.
    vector = [(0.9, _ep("A")), (0.8, _ep("B")), (0.7, _ep("C"))]
    fts = [(12.0, _ep("D")), (11.0, _ep("B"))]

    fused = LongTermMemory._rrf_fuse([vector, fts])
    order = [ep.uid for _, ep in fused]

    # B is ranked #2 in both lists -> highest fused score; sole-list items follow.
    assert order[0] == "B"
    assert set(order) == {"A", "B", "C", "D"}
    scores = {ep.uid: score for score, ep in fused}
    assert scores["B"] == _rrf(2) + _rrf(2)
    assert scores["A"] == _rrf(1)
    assert scores["D"] == _rrf(1)
    assert scores["C"] == _rrf(3)
    # C (vector rank 3) is the lowest.
    assert order[-1] == "C"


def test_rrf_fuse_ranks_by_fused_score_descending():
    fused = LongTermMemory._rrf_fuse(
        [[(0.0, _ep("X")), (0.0, _ep("Y")), (0.0, _ep("Z"))]]
    )
    assert [ep.uid for _, ep in fused] == ["X", "Y", "Z"]
    assert [round(s, 6) for s, _ in fused] == [
        round(_rrf(1), 6),
        round(_rrf(2), 6),
        round(_rrf(3), 6),
    ]


def test_rrf_fuse_dedupes_keeping_first_list_instance():
    # Same uid in both legs with different content; vector passed first must win.
    vector = [(0.5, _ep("dup", content="vector-episode"))]
    fts = [(9.0, _ep("dup", content="fts-episode"))]

    fused = LongTermMemory._rrf_fuse([vector, fts])
    assert len(fused) == 1
    score, episode = fused[0]
    assert episode.content == "vector-episode"
    # Present in both legs at rank 1.
    assert score == _rrf(1) + _rrf(1)


def test_rrf_fuse_respects_k_parameter():
    fused = LongTermMemory._rrf_fuse([[(0.0, _ep("A"))]], k=10)
    assert fused[0][0] == _rrf(1, k=10)


def test_rrf_fuse_empty_input():
    assert LongTermMemory._rrf_fuse([]) == []
    assert LongTermMemory._rrf_fuse([[], []]) == []
