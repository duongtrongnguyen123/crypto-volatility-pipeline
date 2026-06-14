"""TRR pipeline — the four phases over a temporal news stream.

For each day, in chronological order:
    1. Brainstorm   — build the day's impact graph from its news.
    2. Memory       — update the decaying memory with the new edges.
    3. Retrieve     — pull decayed edges from memory and union with today's
                      (this is the "temporal" carry-over across days).
    4. Attention    — PageRank-prune the combined edges to the top_k sub-graph.
    5. Reasoning    — predict the crash probability for the day.

Memory persists across days, so accumulated negative impacts keep elevating the
crash probability even after the originating news ages — the paper's temporal
relational reasoning, adapted to crypto.
"""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from trr.attention import pagerank_prune
from trr.brainstorm import build_impact_graph
from trr.llm import MockLLM, ReasoningLLM
from trr.memory import DecayMemory
from trr.reason import memory_context, reason_crash
from trr.schema import PORTFOLIO, Prediction


class TRRPipeline:
    """Temporal Relational Reasoning pipeline for portfolio crash detection."""

    def __init__(
        self,
        llm: ReasoningLLM = None,
        lam: float = 0.3,
        top_k: int = 30,
        label_threshold: float = 0.5,
        portfolio: list[str] = PORTFOLIO,
        mem_min_relevance: float = 0.5,
    ) -> None:
        self.llm = llm if llm is not None else MockLLM()
        self.lam = lam
        self.top_k = top_k
        self.label_threshold = label_threshold
        self.portfolio = portfolio
        # Memory edges are carried into reasoning only while still salient; with
        # the exponential decay this lets a quiet day shed stale negatives so the
        # crash signal genuinely fades over time.
        self.mem_min_relevance = mem_min_relevance
        self.memory = DecayMemory()

    def _step(self, step: int, day_news: list, day: date) -> Prediction:
        """Run the four phases for a single day."""
        ts = datetime(day.year, day.month, day.day)

        # 1. Brainstorm today's news into an impact graph.
        graph = build_impact_graph(day_news, self.llm, self.portfolio)
        today_edges = list(graph.edges)

        # 2. Update memory with the new edges.
        self.memory.update(today_edges, step)

        # 3. Retrieve decayed temporal context and union with today's edges.
        #    Only edges still above the salience cutoff are carried into
        #    reasoning, so old impacts fade as their relevance decays.
        decayed = self.memory.retrieve(step, self.lam)
        salient = [e for e, r in decayed if r >= self.mem_min_relevance]
        combined = today_edges + [e for e in salient if e not in today_edges]

        # 4. Attention: prune to the most portfolio-relevant sub-graph.
        pruned = pagerank_prune(
            combined, self.portfolio, top_k=self.top_k
        )

        # 5. Reason over the pruned sub-graph (with a memory summary as context).
        context = memory_context(decayed)
        prob, rationale = reason_crash(pruned, self.llm, context=context)

        return Prediction(
            timestamp=ts,
            crash_prob=prob,
            label=int(prob >= self.label_threshold),
            rationale=rationale,
            n_news=len(day_news),
            n_edges=len(pruned),
        )

    def run(
        self,
        news_by_day: dict[date, list],
        dates: list[date] = None,
    ) -> pd.DataFrame:
        """Run the pipeline over a temporal news stream.

        `dates` controls the evaluated days and their order; if omitted, the
        sorted keys of `news_by_day` are used. Days with no news still produce a
        prediction from the decayed memory alone (0 if memory is empty).
        """
        if dates is None:
            dates = sorted(news_by_day.keys())

        rows: list[Prediction] = []
        for step, day in enumerate(dates):
            day_news = news_by_day.get(day, [])
            rows.append(self._step(step, day_news, day))

        df = pd.DataFrame(
            {
                "crash_prob": [p.crash_prob for p in rows],
                "label": [p.label for p in rows],
                "n_news": [p.n_news for p in rows],
                "n_edges": [p.n_edges for p in rows],
                "rationale": [p.rationale for p in rows],
            },
            index=pd.Index(list(dates), name="day"),
        )
        return df


def _demo_news() -> dict:
    """A small synthetic, deterministic news stream across several days.

    Two clearly negative days (a hack and a regulatory lawsuit) are flanked by
    neutral/positive days, with a quiet tail so the temporal decay of the crash
    signal is visible.
    """
    from trr.schema import NewsItem

    def d(day):
        return date(2026, 5, day)

    def item(i, day, title, assets=()):
        return NewsItem(
            id=f"n{i}",
            timestamp=datetime(day.year, day.month, day.day),
            title=title,
            assets=list(assets),
        )

    day1, day2, day3, day4 = d(1), d(2), d(3), d(4)
    day5, day6, day7, day8 = d(5), d(6), d(7), d(8)

    return {
        day1: [
            item(1, day1, "BTC ETF inflows hit a record as adoption surges", ["BTC"]),
            item(2, day1, "ETH network upgrade rallies developer interest", ["ETH"]),
        ],
        day2: [
            item(3, day2, "Routine market update: BTC trades sideways", ["BTC"]),
        ],
        day3: [
            item(4, day3, "Major exchange hacked, BTC plunges amid panic selloff", ["BTC"]),
            item(5, day3, "ETH dumps as contagion fear spreads, liquidations mount", ["ETH"]),
            item(6, day3, "SOL collapse: validators halt after exploit", ["SOL"]),
        ],
        day4: [
            item(7, day4, "Regulators open lawsuit; SEC ban fears trigger selloff", ["BTC", "ETH"]),
        ],
        day5: [
            item(8, day5, "Analysts note calmer conditions across majors", ["BTC"]),
        ],
        day6: [
            item(9, day6, "Partnership and upgrade news lifts SOL and BNB", ["SOL", "BNB"]),
            item(10, day6, "DOGE gains on renewed retail interest", ["DOGE"]),
        ],
        # Quiet tail: no news — the carried crash signal should decay away.
        day7: [],
        day8: [],
    }


def _main() -> None:
    news_by_day = _demo_news()
    days = sorted(news_by_day.keys())
    pipe = TRRPipeline(llm=MockLLM())
    df = pipe.run(news_by_day, days)

    pd.set_option("display.width", 120)
    pd.set_option("display.max_colwidth", 50)
    print("[trr] prediction stream:")
    print(df)

    # --- Validation: negative-news days outrank baseline-neutral days. -----
    # Baseline-neutral = days NOT carrying a fresh crash signal: the pre-crash
    # routine day and the fully-decayed quiet tail.
    neg_days = [date(2026, 5, 3), date(2026, 5, 4)]            # hack / lawsuit
    neutral_days = [date(2026, 5, 2), date(2026, 5, 7), date(2026, 5, 8)]
    min_neg = df.loc[neg_days, "crash_prob"].min()
    max_neutral = df.loc[neutral_days, "crash_prob"].max()
    assert min_neg > max_neutral, (
        f"negative days ({min_neg:.3f}) must exceed neutral days "
        f"({max_neutral:.3f})"
    )
    print(f"\n[trr] OK: min negative-day crash_prob {min_neg:.3f} > "
          f"max neutral-day crash_prob {max_neutral:.3f}")

    # --- Validation: temporal decay of the crash signal after the event. ----
    # The crash peaks on the lawsuit day, then fades across the quiet tail.
    decay_trail = df.loc[
        [date(2026, 5, 4), date(2026, 5, 6), date(2026, 5, 7), date(2026, 5, 8)],
        "crash_prob",
    ].tolist()
    assert decay_trail == sorted(decay_trail, reverse=True), decay_trail
    assert decay_trail[0] > decay_trail[-1], decay_trail
    print(f"[trr] OK: crash signal decays over the quiet tail: "
          f"{[round(x, 3) for x in decay_trail]}")

    # --- Validation: pruning bound + portfolio-adjacency ranking. ----------
    from trr.attention import pagerank_prune
    from trr.schema import ImpactEdge

    edges = [
        ImpactEdge("NEWS:a", "BTC", -1, 0.9, datetime(2026, 5, 3), "a"),
        ImpactEdge("NEWS:b", "ETH", -1, 0.8, datetime(2026, 5, 3), "b"),
        ImpactEdge("NEWS:c", "RANDOM", -1, 0.1, datetime(2026, 5, 3), "c"),
    ]
    pruned = pagerank_prune(edges, top_k=2)
    assert len(pruned) <= 2, len(pruned)
    pruned_objs = {e.object for e in pruned}
    assert "BTC" in pruned_objs and "ETH" in pruned_objs, pruned_objs
    print(f"[trr] OK: pagerank_prune returned {len(pruned)} <= top_k=2, "
          f"portfolio-adjacent edges kept: {sorted(pruned_objs)}")

    # --- Validation: memory decay. -----------------------------------------
    from trr.memory import _selftest as _mem_selftest
    _mem_selftest()


if __name__ == "__main__":
    _main()
