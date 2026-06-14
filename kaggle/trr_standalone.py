"""TRR crypto crash detection — SELF-CONTAINED Kaggle kernel (Qwen / Nemotron).

All TRR code is inlined below (no code dataset). Attaches only:
  - a price-data dataset (the *_5min_long.csv files)  -> crash labels
  - oliviervha/crypto-news (cryptonews.csv)           -> news
  - a HuggingFace model (Qwen2.5-14B-Instruct, etc.)  -> the reasoner
Runs zero-shot on the RTX 6000 Pro (sm_120), no internet.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

BUILD_TAG = "standalone-v2-calib"


class config:  # shim: only HISTORICAL_DIR is referenced by the inlined code
    HISTORICAL_DIR = os.environ.get("HISTORICAL_DIR", "/kaggle/input")


# ===================== trr/schema.py =====================

"""Core data structures for the TRR (Temporal Relational Reasoning) pipeline.

Implements the crypto adaptation of "Temporal Relational Reasoning of Large
Language Models for Detecting Stock Portfolio Crashes" (arXiv:2410.17266).

The pipeline reasons over financial NEWS to detect upcoming crashes in a crypto
portfolio. These types are the contract shared by every phase:

    NewsItem      -> raw input (one article / headline)
    ImpactEdge    -> a directed "X impacts Y" relation the LLM extracts
    ImpactGraph   -> the directed impact graph G=(Z, A) built in Brainstorming
    Prediction    -> the final crash judgement produced by Reasoning
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# The crypto "portfolio" — the relational universe the LLM reasons over.
PORTFOLIO = ["BTC", "ETH", "SOL", "BNB", "AVAX", "DOGE"]
# Map portfolio tickers to the price-data symbols.
SYMBOLS = {t: f"{t}USDT" for t in PORTFOLIO}


@dataclass
class NewsItem:
    """One financial news article/headline."""
    id: str
    timestamp: datetime
    title: str
    body: str = ""
    source: str = ""
    # Tickers explicitly tagged in the source (may be empty — the LLM infers
    # the rest during Brainstorming).
    assets: list[str] = field(default_factory=list)

    def text(self) -> str:
        return f"{self.title}. {self.body}".strip()


@dataclass
class ImpactEdge:
    """A directed impact relation: subject --(polarity)--> object.

    polarity: +1 bullish/positive impact, -1 bearish/negative impact.
    weight:   LLM-estimated strength of the impact in [0, 1].
    """
    subject: str
    object: str
    polarity: int
    weight: float
    timestamp: datetime
    source_news_id: str
    rationale: str = ""

    def as_tuple(self) -> tuple:
        # (t, z_s, a, z_o) — the reasoning-phase input form from the paper.
        return (self.timestamp, self.subject, self.polarity, self.object)


@dataclass
class ImpactGraph:
    """Directed impact graph G=(Z, A) for one time step.

    nodes Z: articles, intermediary entities, and portfolio assets.
    edges A: ImpactEdge relations.
    """
    nodes: set[str] = field(default_factory=set)
    edges: list[ImpactEdge] = field(default_factory=list)

    def add_edge(self, edge: ImpactEdge) -> None:
        self.nodes.add(edge.subject)
        self.nodes.add(edge.object)
        self.edges.append(edge)

    def out_edges(self, node: str) -> list[ImpactEdge]:
        return [e for e in self.edges if e.subject == node]

    def reaches_portfolio(self) -> set[str]:
        """Portfolio assets that appear as an impact object."""
        objs = {e.object for e in self.edges}
        return objs & set(PORTFOLIO)


@dataclass
class Prediction:
    """Final crash judgement for a single time step (e.g. a day)."""
    timestamp: datetime
    crash_prob: float                  # P(portfolio crash next horizon), [0, 1]
    label: int                         # thresholded 0/1
    rationale: str = ""
    # Optional per-asset direction calls: ticker -> -1/0/+1.
    per_asset_direction: dict[str, int] = field(default_factory=dict)
    n_news: int = 0                    # news items considered this step
    n_edges: int = 0                   # impact edges in the pruned subgraph

# ===================== trr/llm.py =====================

"""LLM backend abstraction for the TRR pipeline.

The four TRR phases never call a model directly — they call the two semantic
methods on a `ReasoningLLM`:

    extract_impacts(news, candidate_assets) -> list[ImpactEdge]   # Brainstorming
    predict_crash(tuples, context)          -> (crash_prob, rationale)  # Reasoning

This keeps the phases identical whether the backend is:
    - MockLLM       : deterministic heuristic, for fast offline pipeline tests
    - HFReasoningLLM: a local HuggingFace causal LM (e.g. NVIDIA Nemotron) run
                      zero-shot on the Kaggle RTX 6000 Pro (no internet).

Both implement the same interface, so swapping backends changes nothing else.
"""

import json
import re
from abc import ABC, abstractmethod
from datetime import datetime


# Lightweight lexicons for the deterministic mock backend.
_NEG = {
    "crash", "plunge", "collapse", "hack", "exploit", "bankruptcy", "default",
    "lawsuit", "ban", "selloff", "liquidation", "fraud", "fear", "dump",
    "halt", "insolvent", "contagion", "delist", "sec", "fud", "rug", "depeg",
}
_POS = {
    "surge", "rally", "approval", "etf", "adoption", "partnership", "upgrade",
    "bullish", "record", "inflow", "halving", "breakout", "gain", "soar",
}


def extract_json(text: str):
    """Best-effort extraction of the first JSON object/array in `text`."""
    # Try fenced block first.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text
    # Parse the balanced span of whichever bracket appears FIRST — so a JSON
    # array "[{...},{...}]" is read as the array, not just its first object.
    pos_obj = candidate.find("{")
    pos_arr = candidate.find("[")
    spans = []
    if pos_arr != -1:
        spans.append((pos_arr, "[", "]"))
    if pos_obj != -1:
        spans.append((pos_obj, "{", "}"))
    spans.sort()
    for start, opener, closer in spans:
        depth = 0
        for i in range(start, len(candidate)):
            if candidate[i] == opener:
                depth += 1
            elif candidate[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(candidate[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


class ReasoningLLM(ABC):
    """Backend interface. Subclasses implement `generate`; the two semantic
    methods have prompt-based default implementations that call it.
    """

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.0) -> str:
        ...

    # --- Phase 1: Brainstorming -------------------------------------------
    def extract_impacts(self, news: NewsItem,
                        candidate_assets: list[str]) -> list[ImpactEdge]:
        prompt = self._impact_prompt(news, candidate_assets)
        raw = self.generate(prompt, max_new_tokens=512)
        data = extract_json(raw) or []
        edges: list[ImpactEdge] = []
        for d in data if isinstance(data, list) else []:
            try:
                edges.append(ImpactEdge(
                    subject=str(d["subject"]).upper(),
                    object=str(d["object"]).upper(),
                    polarity=1 if int(d.get("polarity", -1)) >= 0 else -1,
                    weight=max(0.0, min(1.0, float(d.get("weight", 0.5)))),
                    timestamp=news.timestamp,
                    source_news_id=news.id,
                    rationale=str(d.get("rationale", "")),
                ))
            except (KeyError, ValueError, TypeError):
                continue
        return edges

    # --- Phase 1 (batched): daily brainstorming ---------------------------
    def extract_impacts_batch(
        self,
        news_items: list[NewsItem],
        candidate_assets: list[str],
        max_items: int = 40,
    ) -> list[ImpactEdge]:
        """Extract impact edges for a whole day's news in ONE generate() call.

        On a large real corpus, one LLM call per article (`extract_impacts`) is
        prohibitively expensive — 31k articles would mean 31k generations. This
        aggregates a day's (up to `max_items`) headlines into a single numbered
        prompt and asks for one JSON array of impact edges; each edge carries a
        `news_idx` that maps it back to its NewsItem for id/timestamp. Edges with
        an out-of-range index are skipped.

        Callers should pass the most relevant items first; this simply caps the
        list at `max_items` (head/cap) so the prompt stays within budget.
        """
        items = news_items[:max_items]
        if not items:
            return []

        prompt = self._impact_batch_prompt(items, candidate_assets)
        raw = self.generate(prompt, max_new_tokens=1024)
        data = extract_json(raw) or []

        edges: list[ImpactEdge] = []
        for d in data if isinstance(data, list) else []:
            try:
                idx = int(d["news_idx"])
            except (KeyError, ValueError, TypeError):
                continue
            if idx < 0 or idx >= len(items):
                continue  # clamp/skip bad indices
            news = items[idx]
            try:
                edges.append(ImpactEdge(
                    subject=str(d["subject"]).upper(),
                    object=str(d["object"]).upper(),
                    polarity=1 if int(d.get("polarity", -1)) >= 0 else -1,
                    weight=max(0.0, min(1.0, float(d.get("weight", 0.5)))),
                    timestamp=news.timestamp,
                    source_news_id=news.id,
                    rationale=str(d.get("rationale", "")),
                ))
            except (KeyError, ValueError, TypeError):
                continue
        return edges

    # --- Phase 4: Reasoning -----------------------------------------------
    def predict_crash(self, tuples: list[tuple], context: str = "") -> tuple[float, str]:
        prompt = self._reason_prompt(tuples, context)
        raw = self.generate(prompt, max_new_tokens=512)
        data = extract_json(raw) or {}
        try:
            prob = float(data.get("crash_prob", 0.0))
        except (TypeError, ValueError):
            prob = 0.0
        prob = max(0.0, min(1.0, prob))
        return prob, str(data.get("rationale", ""))

    # --- prompt builders (shared by any generate()-based backend) ----------
    @staticmethod
    def _impact_prompt(news: NewsItem, candidate_assets: list[str]) -> str:
        assets = ", ".join(candidate_assets)
        return (
            "You are a financial analyst building an impact graph for crypto "
            "portfolio crash detection. Given a news item, list directed impact "
            "relations from the news/entities toward the portfolio assets.\n"
            f"Portfolio assets: {assets}\n"
            f"News ({news.timestamp:%Y-%m-%d}): {news.text()}\n\n"
            "Return ONLY a JSON array of objects with keys: subject, object, "
            "polarity (1 positive / -1 negative), weight (0..1), rationale. "
            "Use portfolio tickers for affected assets.\n"
        )

    @staticmethod
    def _impact_batch_prompt(news_items: list[NewsItem],
                             candidate_assets: list[str]) -> str:
        assets = ", ".join(candidate_assets)
        day = news_items[0].timestamp if news_items else None
        day_str = f"{day:%Y-%m-%d}" if day is not None else ""
        headlines = "\n".join(
            f"  [{i}] {item.text()}" for i, item in enumerate(news_items)
        )
        return (
            "You are a financial analyst building an impact graph for crypto "
            "portfolio crash detection. Below are the crypto news headlines for "
            f"a single day ({day_str}), each prefixed with a numeric index in "
            "square brackets.\n"
            f"Portfolio assets: {assets}\n\n"
            f"Headlines:\n{headlines}\n\n"
            "For EVERY headline that implies an impact on a portfolio asset, "
            "emit one or more directed impact relations toward that asset. "
            "Return ONLY a single JSON array of objects, each with keys: "
            "news_idx (the bracketed index of the source headline), subject, "
            "object, polarity (1 positive / -1 negative), weight (0..1), "
            "rationale. Use portfolio tickers for affected assets. Omit "
            "headlines with no portfolio impact.\n"
        )

    @staticmethod
    def _reason_prompt(tuples: list[tuple], context: str) -> str:
        lines = "\n".join(
            f"  ({t[0]:%Y-%m-%d}, {t[1]}, {'+' if t[2] >= 0 else '-'}, {t[3]})"
            for t in tuples
        )
        return (
            "You are forecasting whether the crypto portfolio (BTC, ETH, SOL, "
            "BNB, AVAX, DOGE) will CRASH (drop >8% over the next 3 days) from a "
            "graph of dated, directed news-impact relations (time, subject, "
            "polarity, object).\n"
            f"{context}\n"
            f"Impact tuples:\n{lines}\n\n"
            "CALIBRATION — read carefully:\n"
            "- Crypto news is negative on MOST days; routine bearish headlines do "
            "NOT mean a crash. The base rate of actual crashes is only ~13% of "
            "days, so the DEFAULT/typical answer is a LOW probability (~0.10-0.20).\n"
            "- Assign a HIGH probability (>0.6) ONLY when the impacts show "
            "SYSTEMIC, escalating, contagion-style stress concentrated in time — "
            "e.g. a major exchange/stablecoin failure, cascading liquidations, or "
            "insolvency spreading across multiple portfolio assets at once.\n"
            "- Distinguish a normal stream of negative chatter (LOW) from a sharp, "
            "broad escalation beyond the usual baseline (HIGH). Spread your "
            "probabilities across the 0..1 range; do not anchor every day high.\n"
            "Return ONLY JSON: {\"crash_prob\": 0..1, \"rationale\": \"...\"}.\n"
        )


class MockLLM(ReasoningLLM):
    """Deterministic heuristic backend for offline pipeline testing.

    No model: `extract_impacts` keys off a sentiment lexicon and any portfolio
    tickers mentioned; `predict_crash` aggregates the signed, weighted impacts.
    Fully deterministic so pipeline tests are stable.
    """

    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.0) -> str:
        return "{}"  # unused; semantic methods are overridden below

    def extract_impacts(self, news: NewsItem,
                        candidate_assets: list[str]) -> list[ImpactEdge]:
        words = set(re.findall(r"[a-z]+", news.text().lower()))
        neg = len(words & _NEG)
        pos = len(words & _POS)
        polarity = -1 if neg > pos else 1
        strength = min(1.0, 0.3 + 0.2 * abs(neg - pos))

        # Affected assets: those explicitly tagged, else any ticker named in text.
        text_up = news.text().upper()
        affected = [t for t in candidate_assets if t in news.assets or t in text_up]
        if not affected:
            # Market-wide news impacts BTC/ETH as the systemic anchors.
            affected = [t for t in ("BTC", "ETH") if t in candidate_assets]

        edges: list[ImpactEdge] = []
        for asset in affected:
            edges.append(ImpactEdge(
                subject=f"NEWS:{news.id}",
                object=asset,
                polarity=polarity,
                weight=strength,
                timestamp=news.timestamp,
                source_news_id=news.id,
                rationale=f"lexicon neg={neg} pos={pos}",
            ))
        return edges

    def extract_impacts_batch(
        self,
        news_items: list[NewsItem],
        candidate_assets: list[str],
        max_items: int = 40,
    ) -> list[ImpactEdge]:
        """Deterministic batched brainstorming: loop the per-item heuristic over
        the (capped) day's items and concatenate. Keeps the mock fast and stable
        while exercising the same batched code path as the real backend.
        """
        edges: list[ImpactEdge] = []
        for news in news_items[:max_items]:
            edges.extend(self.extract_impacts(news, candidate_assets))
        return edges

    def predict_crash(self, tuples: list[tuple], context: str = "") -> tuple[float, str]:
        if not tuples:
            return 0.0, "no impacts"
        neg = sum(1 for t in tuples if t[2] < 0)
        frac_neg = neg / len(tuples)
        # Logistic-ish squashing on negative concentration + volume.
        prob = max(0.0, min(1.0, 0.15 + 0.7 * frac_neg + 0.02 * min(neg, 10)))
        return prob, f"{neg}/{len(tuples)} negative impacts toward portfolio"


class HFReasoningLLM(ReasoningLLM):
    """Local HuggingFace causal-LM backend (e.g. NVIDIA Nemotron) — zero-shot.

    transformers/torch are imported lazily so this module stays importable on a
    box without them. Intended to run on the Kaggle RTX 6000 Pro (no internet:
    the model is pre-staged as a Kaggle model/dataset and loaded from disk).
    """

    def __init__(self, model_path: str, dtype: str = "bfloat16",
                 device: str = "cuda", max_input_tokens: int = 4096) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.max_input_tokens = max_input_tokens
        # Nemotron ships custom modeling code (model_type "nemotron_h"); it must
        # be trusted to load. The kernel runs non-interactively, so confirm here.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )
        torch_dtype = getattr(torch, dtype, torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = device

    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.0) -> str:
        torch = self._torch
        messages = [{"role": "user", "content": prompt}]
        attention_mask = None
        try:
            enc = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
                return_dict=True, truncation=True, max_length=self.max_input_tokens,
            )
            input_ids = enc["input_ids"].to(self.device)
            if "attention_mask" in enc:
                attention_mask = enc["attention_mask"].to(self.device)
        except Exception:
            enc = self.tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=self.max_input_tokens,
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            out = self.model.generate(input_ids, **gen_kwargs)
        gen = out[0][input_ids.shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

# ===================== trr/memory.py =====================

"""Phase 2 — Memory: a time-decaying store of past impact edges.

Each impact edge is remembered with the step (day index) at which it was seen.
Its relevance decays exponentially with age:

    R = exp(-(current_step - entry_step) * lambda)

so recent impacts dominate while old ones fade. On retrieval, entries below
`min_relevance` are dropped. A bounded history window keeps the store from
growing without limit. Fully deterministic.
"""

import math
from collections import deque



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



# ===================== trr/attention.py =====================

"""Phase 3 — Attention: PageRank-style ranking + pruning of the impact graph.

A power-iteration PageRank is run over the impact-edge node graph with the
teleport (personalization) vector biased toward the portfolio assets, so nodes
that are relationally close to the portfolio score highest. Each edge is then
scored by the importance of its endpoints times its absolute weight, and the
top_k edges are returned. Pure numpy/python and deterministic.
"""

import numpy as np



def _as_edges(graph_or_edges) -> list[ImpactEdge]:
    if isinstance(graph_or_edges, ImpactGraph):
        return list(graph_or_edges.edges)
    return list(graph_or_edges)


def pagerank_prune(
    graph_or_edges,
    portfolio: list[str] = PORTFOLIO,
    top_k: int = 30,
    damping: float = 0.85,
    iters: int = 30,
) -> list[ImpactEdge]:
    """Rank edges by portfolio-biased PageRank importance, return the top_k.

    Accepts either an ImpactGraph or a list[ImpactEdge].
    """
    edges = _as_edges(graph_or_edges)
    if not edges:
        return []

    # Stable node indexing in first-seen order for determinism.
    nodes: list[str] = []
    index: dict[str, int] = {}
    for e in edges:
        for n in (e.subject, e.object):
            if n not in index:
                index[n] = len(nodes)
                nodes.append(n)
    n = len(nodes)

    # Transition matrix M[j, i] = weight of edge i -> j (column-stochastic).
    M = np.zeros((n, n), dtype=float)
    for e in edges:
        i, j = index[e.subject], index[e.object]
        M[j, i] += abs(e.weight)
    col_sums = M.sum(axis=0)
    dangling = col_sums == 0.0
    M[:, ~dangling] /= col_sums[~dangling]

    # Personalization / teleport vector biased toward the portfolio nodes.
    teleport = np.zeros(n, dtype=float)
    for t in portfolio:
        if t in index:
            teleport[index[t]] = 1.0
    teleport = teleport / teleport.sum() if teleport.sum() > 0 else np.full(n, 1.0 / n)

    # Power iteration; dangling nodes redistribute their mass via teleport.
    rank = np.full(n, 1.0 / n, dtype=float)
    for _ in range(iters):
        dangle_mass = rank[dangling].sum()
        rank = (
            (1.0 - damping) * teleport
            + damping * (M @ rank + dangle_mass * teleport)
        )
        s = rank.sum()
        if s > 0:
            rank /= s

    # Score each edge by endpoint importance * |weight|; sort descending.
    scored = [
        (rank[index[e.subject]] + rank[index[e.object]]) * abs(e.weight)
        for e in edges
    ]
    order = sorted(range(len(edges)), key=lambda k: scored[k], reverse=True)
    return [edges[k] for k in order[:top_k]]

# ===================== trr/brainstorm.py =====================

"""Phase 1 — Brainstorming: build the directed impact graph G=(Z, A).

For each news item the LLM extracts directed "X impacts Y" edges with a polarity
(+1/-1) and a weight in [0, 1]. Edges chain the article through any intermediary
entities toward the portfolio assets. Following the paper's iterative expansion,
an edge whose object is a non-portfolio intermediary may be expanded another hop
(up to `max_hops`) so impact can propagate to a portfolio asset.
"""



def _expand_hop(
    edge: ImpactEdge,
    news: NewsItem,
    llm: ReasoningLLM,
    portfolio: list[str],
) -> list[ImpactEdge]:
    """Expand one intermediary edge toward the portfolio.

    Treats the edge's intermediary object as the new subject and asks the LLM how
    that entity impacts the portfolio assets, carrying the parent polarity so the
    sign of the impact propagates along the chain.
    """
    children: list[ImpactEdge] = []
    for child in llm.extract_impacts(news, portfolio):
        if child.object not in portfolio:
            continue
        # Re-root the child onto the intermediary and propagate the chain sign.
        children.append(ImpactEdge(
            subject=edge.object,
            object=child.object,
            polarity=edge.polarity * child.polarity,
            weight=edge.weight * child.weight,
            timestamp=edge.timestamp,
            source_news_id=edge.source_news_id,
            rationale=f"expanded via {edge.object}: {child.rationale}",
        ))
    return children


def build_impact_graph(
    news_items: list[NewsItem],
    llm: ReasoningLLM,
    portfolio: list[str] = PORTFOLIO,
    max_hops: int = 1,
    batch: bool = False,
    max_items: int = 40,
) -> ImpactGraph:
    """Build the directed impact graph for a set of news items.

    Calls `llm.extract_impacts` per item, adds the resulting edges, and — when an
    edge stops at a non-portfolio intermediary — expands further hops (up to
    `max_hops`) so the impact reaches a portfolio asset where possible.

    When ``batch=True`` the brainstorming for the whole set is done in ONE
    `llm.extract_impacts_batch(news_items, portfolio, max_items)` call instead of
    one call per item — the path used for the large real corpus on Kaggle, where
    a per-article call count is infeasible. Multi-hop expansion is not applied in
    the batched path (the batch prompt already asks for portfolio-targeted edges).
    """
    graph = ImpactGraph()
    portfolio_set = set(portfolio)

    if batch:
        for edge in llm.extract_impacts_batch(news_items, portfolio, max_items):
            graph.add_edge(edge)
        return graph

    for news in news_items:
        for edge in llm.extract_impacts(news, portfolio):
            graph.add_edge(edge)
            # Iterative expansion: chain intermediaries toward the portfolio.
            if max_hops > 1 and edge.object not in portfolio_set:
                for child in _expand_hop(edge, news, llm, portfolio):
                    graph.add_edge(child)

    return graph

# ===================== trr/reason.py =====================

"""Phase 4 — Reasoning: predict crash probability from the pruned sub-graph.

Converts the pruned impact edges to (time, subject, polarity, object) tuples and
asks the LLM for a crash probability. This thin layer is also where extra
context (e.g. a decayed-memory summary) is assembled into the prompt context.
"""



def reason_crash(
    edges: list[ImpactEdge],
    llm: ReasoningLLM,
    context: str = "",
) -> tuple[float, str]:
    """Predict (crash_prob, rationale) over the pruned impact edges."""
    tuples = [e.as_tuple() for e in edges]
    return llm.predict_crash(tuples, context=context)


def memory_context(decayed: list[tuple[ImpactEdge, float]], top: int = 5) -> str:
    """Summarise the most relevant decayed-memory edges as a prompt prefix."""
    if not decayed:
        return "No prior impacts in temporal memory."
    parts = [
        f"{e.subject}->{e.object}({'+' if e.polarity >= 0 else '-'},R={r:.2f})"
        for e, r in decayed[:top]
    ]
    return "Temporal memory (recent decayed impacts): " + ", ".join(parts)

# ===================== trr/pipeline.py =====================

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

from datetime import date, datetime

import pandas as pd



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
    ) -> None:
        self.llm = llm if llm is not None else MockLLM()
        self.lam = lam
        self.top_k = top_k
        self.label_threshold = label_threshold
        self.portfolio = portfolio
        # Batched brainstorming: one LLM call per day instead of per article. On
        # the real ~31k-article corpus this is the difference between ~1400 and
        # ~31k generations — the only feasible path on the GPU quota.
        self.batch = batch
        self.max_items_per_day = max_items_per_day
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
        prob, rationale = reason_crash(pruned, self.llm, context=context)

        return Prediction(
            timestamp=ts,
            crash_prob=prob,
            label=int(prob >= self.label_threshold),
            rationale=rationale,
            n_news=len(day_news),
            n_edges=len(pruned),
        )

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
    _mem_selftest()



# ===================== trr/news.py =====================

"""News-data layer for the TRR crypto crash-detection pipeline.

The pipeline reasons over financial NEWS to detect upcoming crashes in the
PORTFOLIO. This module is the ingestion front-end: it loads an arbitrary
crypto-news file (``.jsonl`` or ``.csv``), normalizes every row into the shared
``trr.schema.NewsItem`` contract, and buckets items by calendar day for the
per-day Brainstorming/Reasoning phases.

Because we have NO real news dataset locally, ``trr/sample_news.jsonl`` ships a
deterministic SYNTHETIC corpus whose negative-headline clusters are aligned with
the real portfolio crash windows (LUNA/Terra, FTX, 3AC/Celsius, the Jan-2022
selloff) so the whole pipeline can be demonstrated offline end-to-end. The
synthetic items are clearly fictional/illustrative (``source: "synthetic"``).

Plugging in a REAL dataset
--------------------------
Download a Kaggle crypto-news dataset (e.g. a CryptoPanic / crypto-headlines
CSV) and simply point the loader at it::

    news = load_news("/path/to/crypto_news.csv")
    by_day = group_by_day(news)

``load_news`` is schema-tolerant: the column-mapping below handles the common
header variants (timestamp/date/published_at, title/headline, body/content,
source/publisher, assets/tickers/currencies including CryptoPanic's list-of-
dicts ``currencies`` field), so most public datasets load with no extra work.
"""

import csv
import datetime as dt
import json
import os
from typing import Any, Iterable

import pandas as pd


# --- Column-name variants we accept (first present wins) ---------------------
_TIMESTAMP_KEYS = ["timestamp", "date", "published_at", "time", "created_at"]
_TITLE_KEYS = ["title", "headline", "text", "content"]
_BODY_KEYS = ["body", "content", "text", "description", "summary"]
_SOURCE_KEYS = ["source", "source_title", "publisher", "domain"]
_ASSET_KEYS = ["assets", "tickers", "currencies", "symbols", "coins"]

_PORTFOLIO_SET = set(PORTFOLIO)

# Map common aliases / names to PORTFOLIO tickers. Anything that resolves to a
# portfolio ticker is kept; LUNA/UST etc. are kept verbatim (uppercased) because
# they carry crash signal even though they're not in the portfolio.
_ASSET_ALIASES = {
    "BITCOIN": "BTC",
    "XBT": "BTC",
    "BTC": "BTC",
    "ETHEREUM": "ETH",
    "ETHER": "ETH",
    "ETH": "ETH",
    "SOLANA": "SOL",
    "SOL": "SOL",
    "BINANCE": "BNB",
    "BNB": "BNB",
    "AVALANCHE": "AVAX",
    "AVAX": "AVAX",
    "DOGECOIN": "DOGE",
    "DOGE": "DOGE",
}


def _first_present(row: dict[str, Any], keys: Iterable[str]) -> Any:
    """Return the value of the first key in ``keys`` that has a usable value."""
    for key in keys:
        if key in row:
            val = row[key]
            if val is None:
                continue
            if isinstance(val, float) and pd.isna(val):
                continue
            if isinstance(val, str) and not val.strip():
                continue
            return val
    return None


def normalize_asset(raw: str) -> str | None:
    """Normalize a single asset token to a ticker, or ``None`` if unusable."""
    if raw is None:
        return None
    token = str(raw).strip().upper()
    if not token:
        return None
    # Strip common quote-pair suffixes (BTCUSDT -> BTC, ETH-USD -> ETH).
    for sep in ("/", "-", "_"):
        if sep in token:
            token = token.split(sep)[0]
    for quote in ("USDT", "USD", "USDC", "BUSD"):
        if token.endswith(quote) and len(token) > len(quote):
            token = token[: -len(quote)]
    return _ASSET_ALIASES.get(token, token)


def _parse_assets(raw: Any) -> list[str]:
    """Parse the assets field from a list, delimited string, or list of dicts.

    Accepts CryptoPanic-style ``[{"code": "BTC", "title": "Bitcoin"}, ...]``,
    plain lists, and comma / ``|`` / ``;``-separated strings. Portfolio tickers
    are surfaced first (deduped, order-preserving), other resolved tokens kept.
    """
    if raw is None:
        return []
    tokens: list[str] = []

    if isinstance(raw, str):
        s = raw.strip()
        # A stringified JSON list (common in CSV exports).
        if s.startswith("[") and s.endswith("]"):
            try:
                raw = json.loads(s)
            except (ValueError, TypeError):
                raw = None
                for sep in (",", "|", ";"):
                    s = s.replace(sep, ",")
                tokens = [t for t in s.strip("[]").split(",")]
        if isinstance(raw, str):
            for sep in ("|", ";"):
                raw = raw.replace(sep, ",")
            tokens = raw.split(",")

    if isinstance(raw, (list, tuple)):
        for elem in raw:
            if isinstance(elem, dict):
                tokens.append(str(elem.get("code") or elem.get("title") or ""))
            else:
                tokens.append(str(elem))

    out: list[str] = []
    portfolio_hits: list[str] = []
    for tok in tokens:
        norm = normalize_asset(tok)
        if not norm:
            continue
        if norm in _PORTFOLIO_SET:
            if norm not in portfolio_hits:
                portfolio_hits.append(norm)
        elif norm not in out:
            out.append(norm)
    return portfolio_hits + [t for t in out if t not in portfolio_hits]


def _parse_timestamp(raw: Any) -> dt.datetime | None:
    """Parse a timestamp to a UTC-naive ``datetime``, or ``None``."""
    if raw is None:
        return None
    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    if ts is pd.NaT or pd.isna(ts):
        ts = pd.to_datetime(raw, errors="coerce")
        if ts is pd.NaT or pd.isna(ts):
            return None
        return ts.to_pydatetime().replace(tzinfo=None)
    return ts.tz_convert(None).to_pydatetime()


def _row_to_item(row: dict[str, Any], index: int) -> NewsItem | None:
    """Normalize one raw record into a NewsItem (or ``None`` if unusable)."""
    timestamp = _parse_timestamp(_first_present(row, _TIMESTAMP_KEYS))
    title = _first_present(row, _TITLE_KEYS)
    if timestamp is None or title is None:
        return None
    title = str(title).strip()
    if not title:
        return None

    body_raw = _first_present(row, _BODY_KEYS)
    body = str(body_raw).strip() if body_raw is not None else ""
    # Don't duplicate the title into the body.
    if body == title:
        body = ""

    source_raw = _first_present(row, _SOURCE_KEYS)
    source = str(source_raw).strip() if source_raw is not None else ""

    assets = _parse_assets(_first_present(row, _ASSET_KEYS))

    item_id = row.get("id")
    if item_id is None or (isinstance(item_id, str) and not item_id.strip()):
        item_id = str(index)
    else:
        item_id = str(item_id)

    return NewsItem(
        id=item_id,
        timestamp=timestamp,
        title=title,
        body=body,
        source=source,
        assets=assets,
    )


def _read_records(path: str) -> list[dict[str, Any]]:
    """Read raw records from a ``.jsonl`` or ``.csv`` file."""
    lower = path.lower()
    records: list[dict[str, Any]] = []
    if lower.endswith(".jsonl") or lower.endswith(".ndjson"):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    elif lower.endswith(".json"):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        records = data if isinstance(data, list) else data.get("results", [data])
    elif lower.endswith(".csv") or lower.endswith(".tsv"):
        delimiter = "\t" if lower.endswith(".tsv") else ","
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            records = [dict(r) for r in reader]
    else:
        raise ValueError(f"Unsupported news file type: {path!r} (use .jsonl/.csv)")
    return records


def load_news(path: str) -> list[NewsItem]:
    """Load a ``.jsonl`` or ``.csv`` news file into NewsItems, sorted by time.

    Robust to column-name variants and asset-field formats (see module docstring).
    Rows without a usable title or timestamp are skipped; ids are generated from
    the row index when absent.
    """
    records = _read_records(path)
    items: list[NewsItem] = []
    for index, row in enumerate(records):
        item = _row_to_item(row, index)
        if item is not None:
            items.append(item)
    items.sort(key=lambda it: it.timestamp)
    return items


def group_by_day(news: list[NewsItem]) -> dict[dt.date, list[NewsItem]]:
    """Bucket items by calendar day, chronological within each day."""
    by_day: dict[dt.date, list[NewsItem]] = {}
    for item in sorted(news, key=lambda it: it.timestamp):
        by_day.setdefault(item.timestamp.date(), []).append(item)
    return by_day


def load_sample_news() -> list[NewsItem]:
    """Load the bundled synthetic demo corpus (``trr/sample_news.jsonl``)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_news.jsonl")
    return load_news(path)


# --- Lightweight sentiment for the demo cross-check -------------------------
_NEG_WORDS = {
    "collapse", "insolvent", "insolvency", "contagion", "hack", "hacked",
    "exploit", "liquidation", "liquidated", "depeg", "depegs", "bankruptcy",
    "bankrupt", "plunge", "plunges", "selloff", "sell-off", "ban", "lawsuit",
    "sue", "sues", "fraud", "halt", "halts", "freeze", "freezes", "default",
    "crash", "crashes", "delist", "rout", "panic", "withdrawals", "probe",
}


def is_negative(item: NewsItem) -> bool:
    """Cheap keyword sentiment used only by the demo cross-check."""
    text = item.text().lower()
    return any(word in text for word in _NEG_WORDS)


def _main() -> None:
    news = load_sample_news()
    by_day = group_by_day(news)
    days = sorted(by_day)

    print(f"[news] loaded {len(news)} sample items")
    print(f"[news] date range: {days[0]} -> {days[-1]}")
    print(f"[news] distinct days with news: {len(days)}")

    print("[news] example items:")
    for item in news[:3]:
        print(f"    {item.timestamp.date()}  {item.assets}  "
              f"({item.source})  {item.title}")

    # Cross-check the negative-news clusters against the REAL crash labels.
    try:

        labels = crash_labels()
        crash_days = {ts.date() for ts in labels.index[labels["crash"] == 1]}
    except Exception as exc:  # pragma: no cover - price data may be absent
        print(f"[news] crash cross-check skipped (labels unavailable): {exc}")
        return

    neg_days = sorted(d for d, items in by_day.items() if any(is_negative(i) for i in items))
    print(f"[news] days with NEGATIVE news: {len(neg_days)}")

    def near_crash(day: dt.date, window: int = 1) -> bool:
        return any(
            (day + dt.timedelta(days=delta)) in crash_days
            for delta in range(-window, window + 1)
        )

    on_or_near = [d for d in neg_days if near_crash(d)]
    print(f"[news] negative-news days on/near a real crash day (+-1d): "
          f"{len(on_or_near)} / {len(neg_days)}")

    def covered(lo: str, hi: str) -> list[dt.date]:
        lo_d = dt.date.fromisoformat(lo)
        hi_d = dt.date.fromisoformat(hi)
        return [d for d in neg_days if lo_d <= d <= hi_d]

    luna = covered("2022-05-06", "2022-05-12")
    ftx = covered("2022-11-05", "2022-11-10")
    print(f"[news] LUNA/Terra window (2022-05-06..12): {len(luna)} neg-news days {luna}")
    print(f"[news] FTX window      (2022-11-05..10): {len(ftx)} neg-news days {ftx}")



# ===================== trr/labels.py =====================

"""Portfolio crash labels from price data — the TRR ground truth.

Builds an equal-weight crypto portfolio from the daily closes of the PORTFOLIO
assets and labels each day as a "crash" if the portfolio's forward return over
the next `horizon` days breaches `-threshold` (a sharp drawdown). This mirrors
the crash-detection target of arXiv:2410.17266, adapted to crypto.

Crash events are intentionally imbalanced (rare) — hence AUROC for evaluation.
"""

import os

import numpy as np
import pandas as pd


# A day is a "crash" if the equal-weight portfolio falls more than this over the
# forward window. 8% over 3 days is a severe multi-asset crypto drawdown.
DEFAULT_THRESHOLD = 0.08
DEFAULT_HORIZON = 3  # days


def _load_daily_close(symbol: str, hist_dir: str) -> pd.Series:
    path = os.path.join(hist_dir, f"{symbol}_5min_long.csv")
    df = pd.read_csv(path, usecols=["timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    s = df.set_index("timestamp")["close"].sort_index()
    # Last close of each calendar day.
    return s.resample("1D").last().dropna()


def build_portfolio(hist_dir: str = None) -> pd.DataFrame:
    """Equal-weight daily portfolio level + per-asset daily returns."""
    hist_dir = hist_dir or config.HISTORICAL_DIR
    closes = {}
    for ticker in PORTFOLIO:
        closes[ticker] = _load_daily_close(SYMBOLS[ticker], hist_dir)
    px = pd.DataFrame(closes).dropna(how="all").sort_index()
    px = px.ffill().dropna()

    rets = px.pct_change().fillna(0.0)
    # Equal-weight portfolio daily return -> cumulative level.
    port_ret = rets.mean(axis=1)
    port_level = (1.0 + port_ret).cumprod()

    out = rets.add_suffix("_ret")
    out["portfolio_ret"] = port_ret
    out["portfolio_level"] = port_level
    return out


def crash_labels(
    hist_dir: str = None,
    threshold: float = DEFAULT_THRESHOLD,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """Return a frame indexed by day with a `crash` 0/1 column and the forward
    return used to derive it.
    """
    port = build_portfolio(hist_dir)
    level = port["portfolio_level"]

    # Forward return over the next `horizon` days: min level ahead / today - 1.
    fwd_min = level.shift(-1).rolling(horizon, min_periods=1).min().shift(-(horizon - 1))
    # Simpler, robust forward drawdown: lowest close within the next horizon days.
    fwd_low = (
        level.iloc[::-1].rolling(horizon, min_periods=1).min().iloc[::-1].shift(-1)
    )
    fwd_ret = fwd_low / level - 1.0

    port = port.copy()
    port["fwd_ret"] = fwd_ret
    port["crash"] = (fwd_ret <= -threshold).astype("Int64")
    # Drop the tail where the forward window is undefined.
    port = port.dropna(subset=["fwd_ret"]).copy()
    port["crash"] = port["crash"].astype(int)
    return port




# =========================================================================== #
# Kernel orchestration (self-contained — no code dataset needed).
# =========================================================================== #
KAGGLE_WORKING = "/kaggle/working"
SMOKE_OUT_DIR = "/tmp/trr_smoke_out"
LOCAL_NEWS_CSV = "data/news_raw/oliviervha/cryptonews.csv"
DEFAULT_START, DEFAULT_END = "2022-10-01", "2022-12-15"
SMOKE_START, SMOKE_END = "2022-11-05", "2022-11-12"
MAX_ITEMS_PER_DAY = 40


def _is_smoke():
    return os.environ.get("SMOKE", "0") == "1"


def _glob1(*patterns):
    for p in patterns:
        hits = sorted(glob.glob(p, recursive=True))
        if hits:
            return hits[0]
    return None


def _gpu_gate():
    import torch
    if not torch.cuda.is_available():
        print("[gpu] CUDA not available (CPU/SMOKE).", flush=True)
        return "float32"
    major, minor = torch.cuda.get_device_capability(0)
    print(f"[gpu] {torch.cuda.get_device_name(0)}  sm_{major}{minor}  torch={torch.__version__}", flush=True)
    if (major, minor) == (6, 0):
        print("[gpu] FATAL: P100/sm_60 fallback — the three-field RTX 6000 Pro gate failed.", flush=True)
        sys.exit(1)
    return "bfloat16" if major >= 8 else "float16"


def _find_model_dir():
    for cfg in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
        d = os.path.dirname(cfg)
        if any(os.path.exists(os.path.join(d, t)) for t in
               ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")):
            return d
    return None


def _save_outputs(pred_df, metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pred_df.to_csv(os.path.join(out_dir, "trr_predictions.csv"))
    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(range(len(pred_df)), pred_df["crash_prob"], label="crash_prob")
        crash = pred_df["label"].to_numpy()
        for i, c in enumerate(crash):
            if c == 1:
                ax.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.15)
        ax.set_title("TRR crash probability vs actual crash days (shaded)")
        ax.legend()
        fig.savefig(os.path.join(out_dir, "trr_timeline.png"), dpi=120, bbox_inches="tight")
    except Exception as exc:
        print(f"[plot] skipped: {exc}", flush=True)


def _evaluate(pred_df, out_dir):
    """AUROC / PR-AUC of TRR crash_prob vs price-derived labels + baselines."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = crash_labels()
    lab = labels.copy()
    lab.index = pd.to_datetime(lab.index).date
    df = pred_df.copy()
    df["crash"] = [int(lab["crash"].get(d, 0)) for d in df.index]

    y = df["crash"].to_numpy()
    metrics = {"summary": {"n_days": int(len(df)), "n_crash_days": int(y.sum()),
                           "base_rate": float(y.mean()),
                           "date_start": str(df.index.min()),
                           "date_end": str(df.index.max())},
               "metrics": {}}
    if 0 < y.sum() < len(y):
        metrics["metrics"]["TRR"] = {
            "auroc": float(roc_auc_score(y, df["crash_prob"])),
            "pr_auc": float(average_precision_score(y, df["crash_prob"])),
        }
        # news-volume baseline: more news ~ more attention ~ risk
        if "n_news" in df:
            metrics["metrics"]["news_volume"] = {
                "auroc": float(roc_auc_score(y, df["n_news"])),
                "pr_auc": float(average_precision_score(y, df["n_news"])),
            }
    else:
        metrics["summary"]["single_class_window"] = True
    _save_outputs(df, metrics, out_dir)
    return metrics


def main():
    smoke = _is_smoke()
    print(f"[kernel] BUILD={BUILD_TAG} mode={'SMOKE' if smoke else 'KAGGLE'}", flush=True)

    # price data dir -> HISTORICAL_DIR
    price = _glob1("/kaggle/input/**/BTCUSDT_5min_long.csv",
                   os.path.join(config.HISTORICAL_DIR, "BTCUSDT_5min_long.csv"))
    if smoke and not price:
        price = "/home/nduong/eth-alpha/data/BTCUSDT_5min_long.csv"
    config.HISTORICAL_DIR = os.path.dirname(price)
    print(f"[kernel] HISTORICAL_DIR={config.HISTORICAL_DIR}", flush=True)

    # news
    news_csv = _glob1("/kaggle/input/**/*cryptonews*.csv", "/kaggle/input/**/*crypto*news*.csv")
    if smoke and not news_csv and os.path.exists(LOCAL_NEWS_CSV):
        news_csv = LOCAL_NEWS_CSV
    print(f"[kernel] news={news_csv}", flush=True)
    news = load_news(news_csv)

    start = os.environ.get("TRR_START", SMOKE_START if smoke else DEFAULT_START)
    end = os.environ.get("TRR_END", SMOKE_END if smoke else DEFAULT_END)
    out_dir = SMOKE_OUT_DIR if smoke else (KAGGLE_WORKING if os.path.isdir(KAGGLE_WORKING) else "/tmp")

    dtype = _gpu_gate()
    if smoke:
        llm = MockLLM()
    else:
        model_dir = _find_model_dir()
        print(f"[kernel] model dir: {model_dir}", flush=True)
        llm = HFReasoningLLM(model_path=model_dir, dtype=dtype)

    print(f"[kernel] window {start}..{end}  news_items={len(news)}", flush=True)
    pipe = TRRPipeline(llm=llm, batch=True, max_items_per_day=MAX_ITEMS_PER_DAY, lam=0.6)
    pred = pipe.run(group_by_day(news), start=start, end=end)
    print(f"[kernel] predicted {len(pred)} days", flush=True)

    metrics = _evaluate(pred, out_dir)
    print(f"[kernel] metrics: {json.dumps(metrics.get('metrics', {}))}", flush=True)
    print(f"[kernel] wrote outputs -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
