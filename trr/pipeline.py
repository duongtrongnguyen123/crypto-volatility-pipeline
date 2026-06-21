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

import numpy as np
import pandas as pd

from trr.attention import pagerank_prune
from trr.brainstorm import build_impact_graph
from trr.llm import MockLLM, ReasoningLLM
from trr.memory import DecayMemory
from trr.rag import day_text
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
        batch: bool = False,
        max_items_per_day: int = 40,
        cross_batch: bool = False,
        per_asset: bool = False,
        target_mode: str = "crash",
        reason_samples: int = 1,
        reason_temp: float = 0.0,
        reason_max_new_tokens: int = 256,
        brainstorm_max_new_tokens: int = 768,
        rag=None,
        rag_labels=None,
        multi_hop=False,
    ) -> None:
        self.llm = llm if llm is not None else MockLLM()
        # multi_hop: enrich the reasoning context with multi-hop causal chains
        # (Graph-RAG) walked from the accumulated impact graph.
        self.multi_hop = multi_hop
        # rag: optional CausalRAG retriever that injects case-based few-shot
        # (similar PAST days + their realized outcomes) into the reasoning
        # context. rag_labels maps a day -> realized crash label (0/1); only
        # days older than the retriever's embargo are ever used, so it is causal.
        self.rag = rag
        self.rag_labels = rag_labels
        self.lam = lam
        self.top_k = top_k
        self.label_threshold = label_threshold
        self.portfolio = portfolio
        # Batched brainstorming: one LLM call per day instead of per article. On
        # the real ~31k-article corpus this is the difference between ~1400 and
        # ~31k generations — the only feasible path on the GPU quota.
        self.batch = batch
        self.max_items_per_day = max_items_per_day
        # cross_batch: batch the LLM calls ACROSS days (all brainstorm prompts in
        # one batched pass, then all reason prompts) instead of day-by-day. Same
        # results — the per-day memory/attention is still sequential — but the two
        # LLM phases run as batched forwards, the big speedup for the full window.
        self.cross_batch = cross_batch
        # per_asset: emit a crash probability per portfolio asset (cross_batch).
        self.per_asset = per_asset
        # target_mode: "crash" (default, all existing behaviour) asks the LLM for
        # a crash probability; "direction" asks for the next-day price-up
        # probability and surfaces it as an `up_prob` column. Other phases are
        # unchanged — only the Reasoning phase and the output column differ.
        if target_mode not in ("crash", "direction"):
            raise ValueError(f"target_mode must be 'crash' or 'direction', got {target_mode!r}")
        self.target_mode = target_mode
        # Self-consistency: sample reason_samples reasoning traces at reason_temp
        # and average (test-time compute scaling). reason_max_new_tokens is large
        # for reasoning models that emit a <think> trace.
        self.reason_samples = reason_samples
        self.reason_temp = reason_temp
        self.reason_max_new_tokens = reason_max_new_tokens
        self.brainstorm_max_new_tokens = brainstorm_max_new_tokens
        # Memory edges are carried into reasoning only while still salient; with
        # the exponential decay this lets a quiet day shed stale negatives so the
        # crash signal genuinely fades over time.
        self.mem_min_relevance = mem_min_relevance
        self.memory = DecayMemory()

    def _step(self, step: int, day_news: list, day: date) -> Prediction:
        """Run the four phases for a single day."""
        ts = datetime(day.year, day.month, day.day)

        # 1. Brainstorm today's news into an impact graph. In batched mode this
        #    is ONE LLM call covering the whole day's (capped) headlines.
        graph = build_impact_graph(
            day_news, self.llm, self.portfolio,
            batch=self.batch, max_items=self.max_items_per_day,
        )
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
        if self.target_mode == "direction":
            prob, rationale = self.llm.reason_multi_direction(
                [[e.as_tuple() for e in pruned]], [context],
                max_new_tokens=self.reason_max_new_tokens,
                universe=self.portfolio,
            )[0]
        else:
            prob, rationale = reason_crash(pruned, self.llm, context=context,
                                           universe=self.portfolio)

        return Prediction(
            timestamp=ts,
            crash_prob=prob,
            label=int(prob >= self.label_threshold),
            rationale=rationale,
            n_news=len(day_news),
            n_edges=len(pruned),
        )

    def _run_cross_batch(self, news_by_day: dict, dates: list) -> list:
        """Phase-separated batched execution (same semantics as the per-day loop).

        A: brainstorm ALL days in one batched LLM pass.
        B: sequential memory update + decay + attention prune (no LLM) per day.
        C: reason ALL days in one batched LLM pass.
        """
        day_news = [news_by_day.get(d, []) for d in dates]

        # Phase A — batched brainstorming.
        edges_per_day = self.llm.brainstorm_multi(
            day_news, self.portfolio, max_items=self.max_items_per_day,
            max_new_tokens=self.brainstorm_max_new_tokens,
        )

        # Phase B — sequential memory/attention to build each day's reason input.
        tuples_list, contexts, n_edges = [], [], []
        for step, today_edges in enumerate(edges_per_day):
            self.memory.update(today_edges, step)
            decayed = self.memory.retrieve(step, self.lam)
            salient = [e for e, r in decayed if r >= self.mem_min_relevance]
            combined = today_edges + [e for e in salient if e not in today_edges]
            pruned = pagerank_prune(combined, self.portfolio, top_k=self.top_k)
            tuples_list.append([e.as_tuple() for e in pruned])
            ctx = memory_context(decayed)
            if self.multi_hop:
                from trr.graphrag import chains_context
                block = chains_context(combined, self.portfolio)
                if block:
                    ctx = block + ctx
            contexts.append(ctx)
            n_edges.append(len(pruned))

        # Phase B2 (optional) — RAG: prepend case-based few-shot (similar PAST
        # days + realized outcomes) to each day's context. Causal: the retriever
        # only ever looks back beyond its embargo.
        if self.rag is not None:
            self.rag.fit([day_text(dn) for dn in day_news], dates)
            labels = [int(self.rag_labels.get(d, 0)) if self.rag_labels else 0
                      for d in dates]
            for i in range(len(contexts)):
                block = self.rag.fewshot(i, labels)
                if block:
                    contexts[i] = block + "\n" + contexts[i]

        # Phase C — batched reasoning.
        if self.target_mode == "direction":
            results = self.llm.reason_multi_direction(
                tuples_list, contexts, max_new_tokens=self.reason_max_new_tokens,
                n_samples=self.reason_samples, temperature=self.reason_temp,
                universe=self.portfolio,
            )
            rows = []
            for d, (up_prob, rationale), ne, dn in zip(dates, results, n_edges, day_news):
                ts = datetime(d.year, d.month, d.day)
                # Stored in the crash_prob field; surfaced as `up_prob` in run().
                rows.append(Prediction(
                    timestamp=ts, crash_prob=up_prob,
                    label=int(up_prob >= self.label_threshold), rationale=rationale,
                    n_news=len(dn), n_edges=ne,
                ))
            return rows

        if self.per_asset:
            per = self.llm.reason_multi_per_asset(tuples_list, contexts, self.portfolio)
            rows = []
            for d, probs, ne, dn in zip(dates, per, n_edges, day_news):
                ts = datetime(d.year, d.month, d.day)
                # Portfolio-level proxy = mean of the per-asset probabilities.
                pf = float(np.mean(list(probs.values()))) if probs else 0.0
                rows.append(Prediction(
                    timestamp=ts, crash_prob=pf,
                    label=int(pf >= self.label_threshold), rationale="",
                    n_news=len(dn), n_edges=ne, per_asset_direction=dict(probs),
                ))
            return rows

        results = self.llm.reason_multi(
            tuples_list, contexts, max_new_tokens=self.reason_max_new_tokens,
            n_samples=self.reason_samples, temperature=self.reason_temp,
            universe=self.portfolio,
        )
        rows = []
        for d, (prob, rationale), ne, dn in zip(dates, results, n_edges, day_news):
            ts = datetime(d.year, d.month, d.day)
            rows.append(Prediction(
                timestamp=ts, crash_prob=prob,
                label=int(prob >= self.label_threshold), rationale=rationale,
                n_news=len(dn), n_edges=ne,
            ))
        return rows

    @staticmethod
    def _as_date(value) -> date:
        """Coerce a 'YYYY-MM-DD' string / datetime / date into a date."""
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value))

    def run(
        self,
        news_by_day: dict[date, list],
        dates: list[date] = None,
        start=None,
        end=None,
    ) -> pd.DataFrame:
        """Run the pipeline over a temporal news stream.

        `dates` controls the evaluated days and their order; if omitted, the
        sorted keys of `news_by_day` are used. Days with no news still produce a
        prediction from the decayed memory alone (0 if memory is empty).

        `start`/`end` (inclusive) optionally restrict the processed window to a
        date range — accepts a `date`, a `datetime`, or a 'YYYY-MM-DD' string.
        This bounds the LLM call count for a quota-cheap real run (e.g. the FTX
        validation window) without the caller having to pre-filter `news_by_day`.
        """
        if dates is None:
            dates = sorted(news_by_day.keys())

        if start is not None:
            start_d = self._as_date(start)
            dates = [d for d in dates if d >= start_d]
        if end is not None:
            end_d = self._as_date(end)
            dates = [d for d in dates if d <= end_d]

        if self.cross_batch:
            rows = self._run_cross_batch(news_by_day, dates)
        else:
            rows = []
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
        # Per-asset mode: add one crash-probability column per portfolio asset.
        if self.per_asset:
            for ticker in self.portfolio:
                df[f"crash_prob_{ticker}"] = [
                    p.per_asset_direction.get(ticker, 0.0) for p in rows
                ]
        # Direction mode: the reasoning output is a next-day price-up probability,
        # so surface it under its own column name (the value is carried in the
        # Prediction.crash_prob field internally).
        if self.target_mode == "direction":
            df = df.rename(columns={"crash_prob": "up_prob"})
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
