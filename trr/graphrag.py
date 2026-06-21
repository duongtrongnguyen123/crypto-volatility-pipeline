"""Multi-hop Graph-RAG — relational retrieval beyond single edges.

The base TRR pipeline feeds the LLM single (subject -> object) impact edges. Real
contagion is multi-hop: e.g. OIL_PRICE -> AIRLINES -> the portfolio. This module
walks the accumulated impact graph (today's edges + decayed memory) and extracts
the strongest directed CHAINS (length <= max_hops) terminating at a portfolio
asset, surfacing them as explicit causal paths in the reasoning context.

In-process (networkx) — the same idea as the comparison project's Neo4j Graph-RAG,
but with no external database. Deterministic.
"""
from __future__ import annotations

import networkx as nx


def _polarity_str(p) -> str:
    return "+" if p >= 0 else "-"


def multi_hop_chains(edges, portfolio, max_hops: int = 2, top: int = 8) -> list[str]:
    """Return up to `top` strongest causal chains ending at a portfolio asset.

    `edges` is a list of ImpactEdge (or tuples) with subject, object, polarity,
    weight. Chains are ranked by the product of their edge weights (a path is
    only as strong as its weakest, multiplicatively-combined link).
    """
    g = nx.DiGraph()
    for e in edges:
        subj = getattr(e, "subject", None)
        obj = getattr(e, "object", None)
        if subj is None:  # tuple form (time, subject, polarity, object/weight)
            continue
        w = float(getattr(e, "weight", 0.5))
        pol = getattr(e, "polarity", 1)
        # keep the strongest parallel edge
        if g.has_edge(subj, obj) and g[subj][obj]["weight"] >= w:
            continue
        g.add_edge(subj, obj, weight=w, polarity=pol)

    targets = {a for a in portfolio if a in g}
    chains: list[tuple[float, str]] = []
    seen: set[tuple] = set()
    for tgt in targets:
        # walk backwards from the target up to max_hops
        for src in g.nodes:
            if src == tgt:
                continue
            try:
                paths = nx.all_simple_paths(g, src, tgt, cutoff=max_hops)
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                continue
            for path in paths:
                if len(path) < 3:  # only genuine multi-hop (>=2 edges)
                    continue
                key = tuple(path)
                if key in seen:
                    continue
                seen.add(key)
                strength = 1.0
                parts = [path[0]]
                for u, v in zip(path[:-1], path[1:]):
                    d = g[u][v]
                    strength *= d["weight"]
                    parts.append(f"--({_polarity_str(d['polarity'])})-->{v}")
                chains.append((strength, " ".join(parts)))
    chains.sort(key=lambda x: -x[0])
    return [c for _, c in chains[:top]]


def shared_drivers(edges, portfolio, min_assets: int = 2, top: int = 6):
    """Find single drivers that hit MULTIPLE portfolio assets at once.

    A node with negative edges into >= min_assets portfolio assets is a systemic
    / contagion signal — the relational pattern that distinguishes a broad shock
    from an isolated one. Returns (driver, [assets], mean_weight, polarity).
    """
    g = nx.DiGraph()
    for e in edges:
        subj = getattr(e, "subject", None)
        obj = getattr(e, "object", None)
        if subj is None:
            continue
        g.add_edge(subj, obj, weight=float(getattr(e, "weight", 0.5)),
                   polarity=getattr(e, "polarity", 1))
    pset = set(portfolio)
    out = []
    for node in g.nodes:
        hit = [(v, g[node][v]) for v in g.successors(node) if v in pset]
        if len(hit) >= min_assets:
            mean_w = sum(d["weight"] for _, d in hit) / len(hit)
            pol = -1 if sum(d["polarity"] for _, d in hit) < 0 else 1
            out.append((node, [v for v, _ in hit], mean_w, pol))
    out.sort(key=lambda x: (-len(x[1]), -x[2]))
    return out[:top]


def chains_context(edges, portfolio, max_hops: int = 2, top: int = 8) -> str:
    """Reasoning-context block: multi-hop chains + shared systemic drivers."""
    parts = []
    chains = multi_hop_chains(edges, portfolio, max_hops, top)
    if chains:
        parts.append("MULTI-HOP CAUSAL CHAINS (indirect contagion paths into the "
                     "portfolio; weigh propagated effects, not just direct hits):\n"
                     + "\n".join(f"  - {c}" for c in chains))
    drivers = shared_drivers(edges, portfolio)
    if drivers:
        lines = [f"  - {n} hits {','.join(a)} ({_polarity_str(p)}, "
                 f"w~{w:.2f})" for n, a, w, p in drivers]
        parts.append("SHARED SYSTEMIC DRIVERS (one event hitting MULTIPLE "
                     "portfolio assets at once — breadth signals contagion):\n"
                     + "\n".join(lines))
    return ("\n".join(parts) + "\n") if parts else ""


if __name__ == "__main__":
    from datetime import datetime

    from trr.schema import ImpactEdge
    E = [
        ImpactEdge("OIL", "AIRLINES", -1, 0.8, datetime(2026, 1, 1), "n1"),
        ImpactEdge("AIRLINES", "BTC", -1, 0.6, datetime(2026, 1, 1), "n2"),
        ImpactEdge("FED", "BTC", -1, 0.7, datetime(2026, 1, 1), "n3"),
        ImpactEdge("HACK", "ETH", -1, 0.9, datetime(2026, 1, 1), "n4"),
        ImpactEdge("ETH", "BTC", -1, 0.5, datetime(2026, 1, 1), "n5"),
    ]
    chains = multi_hop_chains(E, ["BTC", "ETH"])
    print(f"[graphrag] {len(chains)} multi-hop chains:")
    for c in chains:
        print("  ", c)
    assert any("AIRLINES" in c and "BTC" in c for c in chains), chains
    assert any("HACK" in c and "ETH" in c and "BTC" in c for c in chains), chains
    print("[graphrag] self-test passed")
