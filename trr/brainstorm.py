"""Phase 1 — Brainstorming: build the directed impact graph G=(Z, A).

For each news item the LLM extracts directed "X impacts Y" edges with a polarity
(+1/-1) and a weight in [0, 1]. Edges chain the article through any intermediary
entities toward the portfolio assets. Following the paper's iterative expansion,
an edge whose object is a non-portfolio intermediary may be expanded another hop
(up to `max_hops`) so impact can propagate to a portfolio asset.
"""
from __future__ import annotations

from trr.llm import ReasoningLLM
from trr.schema import PORTFOLIO, ImpactEdge, ImpactGraph, NewsItem


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
) -> ImpactGraph:
    """Build the directed impact graph for a set of news items.

    Calls `llm.extract_impacts` per item, adds the resulting edges, and — when an
    edge stops at a non-portfolio intermediary — expands further hops (up to
    `max_hops`) so the impact reaches a portfolio asset where possible.
    """
    graph = ImpactGraph()
    portfolio_set = set(portfolio)

    for news in news_items:
        for edge in llm.extract_impacts(news, portfolio):
            graph.add_edge(edge)
            # Iterative expansion: chain intermediaries toward the portfolio.
            if max_hops > 1 and edge.object not in portfolio_set:
                for child in _expand_hop(edge, news, llm, portfolio):
                    graph.add_edge(child)

    return graph
