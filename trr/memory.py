"""Phase 2 — Memory: a time-decaying store of past impact edges.

Each impact edge is remembered with the step (day index) at which it was seen.
Its relevance decays exponentially with age:

    R = exp(-(current_step - entry_step) * lambda)

so recent impacts dominate while old ones fade. On retrieval, entries below
`min_relevance` are dropped. A bounded history window keeps the store from
growing without limit. Fully deterministic.
"""
from __future__ import annotations

import math
from collections import deque

from trr.schema import ImpactEdge


class DecayMemory:
    """A bounded, exponentially-decaying memory of impact edges."""

    def __init__(self, max_history: int = 2000) -> None:
        # Each entry is (edge, step); newest appended at the right.
        self._entries: deque[tuple[ImpactEdge, int]] = deque(maxlen=max_history)
        self.max_history = max_history

    def __len__(self) -> int:
        return len(self._entries)

    def update(self, edges: list[ImpactEdge], step: int) -> None:
        """Record `edges` as observed at time `step`."""
        for edge in edges:
            self._entries.append((edge, step))

    def retrieve(
        self,
        current_step: int,
        lam: float,
        min_relevance: float = 0.05,
    ) -> list[tuple[ImpactEdge, float]]:
        """Return remembered edges with their decayed relevance.

        Relevance R = exp(-(current_step - entry_step) * lam). Entries whose age
        falls below `min_relevance`, or that lie in the future, are dropped.
        Results are sorted by descending relevance for stable, deterministic use.
        """
        out: list[tuple[ImpactEdge, float]] = []
        for edge, step in self._entries:
            age = current_step - step
            if age < 0:
                continue
            relevance = math.exp(-age * lam)
            if relevance >= min_relevance:
                out.append((edge, relevance))
        out.sort(key=lambda pair: pair[1], reverse=True)
        return out


def _selftest() -> None:
    """Decay sanity check: older edges must score strictly lower."""
    from datetime import datetime

    mem = DecayMemory()
    old = ImpactEdge("NEWS:old", "BTC", -1, 0.8, datetime(2026, 1, 1), "NEWS:old")
    new = ImpactEdge("NEWS:new", "BTC", -1, 0.8, datetime(2026, 1, 5), "NEWS:new")
    mem.update([old], step=0)
    mem.update([new], step=4)

    scored = dict(
        (e.source_news_id, r) for e, r in mem.retrieve(current_step=4, lam=0.3)
    )
    assert scored["NEWS:new"] > scored["NEWS:old"], scored
    # An edge old enough falls below the threshold and is pruned.
    far = mem.retrieve(current_step=40, lam=0.3, min_relevance=0.05)
    assert all(e.source_news_id != "NEWS:old" for e, _ in far), far
    print("[memory] decay self-test passed:", scored)


if __name__ == "__main__":
    _selftest()
