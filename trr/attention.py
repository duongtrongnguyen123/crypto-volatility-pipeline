"""Phase 3 — Attention: PageRank-style ranking + pruning of the impact graph.

A power-iteration PageRank is run over the impact-edge node graph with the
teleport (personalization) vector biased toward the portfolio assets, so nodes
that are relationally close to the portfolio score highest. Each edge is then
scored by the importance of its endpoints times its absolute weight, and the
top_k edges are returned. Pure numpy/python and deterministic.
"""
from __future__ import annotations

import numpy as np

from trr.schema import PORTFOLIO, ImpactEdge, ImpactGraph


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
