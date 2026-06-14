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
from __future__ import annotations

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
