"""Pure (Streamlit-free) data + figure builders for the TRR web platform.

Everything in this module is importable and runnable headless — no Streamlit, no
global UI state. The Streamlit UI (``app.py``) is a thin shell over these
functions, and the ``__main__`` block at the bottom doubles as a smoke test:

    python -m webapp.lib          # or:  python webapp/lib.py

runs every public function against the repo's bundled data and prints a summary.

The TRR pipeline (``trr/``) reasons over financial NEWS to predict portfolio
CRASH probability via four phases — Brainstorm (news -> impact graph), Memory
(time-decay), Attention (PageRank prune), Reason (LLM -> crash prob). These
helpers surface the pipeline's outputs (per-day prediction CSVs, a live demo
impact graph from the heuristic MockLLM, the campaign results table) as plain
DataFrames / dicts / Plotly figures.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Optional

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

# --- Repo layout ------------------------------------------------------------
# webapp/ lives directly under the repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAGGLE_DIR = os.path.join(REPO_ROOT, "kaggle")
PRICES_DIR = os.path.join(REPO_ROOT, "data", "fnspid", "prices")
RESULTS_MD = os.path.join(REPO_ROOT, "reports", "RESULTS_TRR.md")

# Stocks that have local price series (data/fnspid/prices/*.csv).
PRICE_TICKERS = ["AAPL", "AMZN", "GOOGL", "NVDA", "TSLA", "NFLX"]

# Human labels for the per-day prediction runs we ship in the UI dropdown.
# Maps a kaggle/out_* slug -> a short description shown to the user.
RUN_LABELS = {
    "s9": "COVID crash (2020, stocks)",
    "main": "Main campaign run",
    "stock_base": "Stock TRR — base",
    "stock_rag": "Stock TRR — RAG few-shot",
    "f1": "Fold 1",
    "f2": "Fold 2",
    "f3": "Fold 3",
    "f4": "Fold 4",
    "f5": "Fold 5",
}


# ===========================================================================
# (a) Load a per-day prediction CSV into a tidy DataFrame.
# ===========================================================================
def list_prediction_runs() -> list[dict]:
    """Discover every ``kaggle/out_*/crash/trr_predictions.csv`` run.

    Returns a list of ``{"slug", "label", "path"}`` dicts sorted with the
    best-known demo runs first, then the rest alphabetically. Pure filesystem
    scan — safe to call headless.
    """
    runs: list[dict] = []
    if not os.path.isdir(KAGGLE_DIR):
        return runs
    for name in os.listdir(KAGGLE_DIR):
        if not name.startswith("out_"):
            continue
        path = os.path.join(KAGGLE_DIR, name, "crash", "trr_predictions.csv")
        if os.path.isfile(path):
            slug = name[len("out_"):]
            runs.append(
                {
                    "slug": slug,
                    "label": RUN_LABELS.get(slug, slug),
                    "path": path,
                }
            )

    # Stable, friendly ordering: featured slugs first, in the order declared.
    featured = list(RUN_LABELS.keys())

    def sort_key(run: dict):
        slug = run["slug"]
        return (featured.index(slug) if slug in featured else len(featured), slug)

    return sorted(runs, key=sort_key)


def load_predictions(path: str) -> pd.DataFrame:
    """Load a TRR per-day predictions CSV into a DataFrame.

    Columns produced (always present, filled with NaN if absent in the file):
        date (index, datetime), crash_prob, label, n_news, n_edges,
        rationale, label_true.

    The on-disk index column is named ``day``; we parse it to datetime and
    rename the index to ``date`` for the UI.
    """
    df = pd.read_csv(path)
    # The first column is the date index ("day" in the campaign CSVs).
    index_col = df.columns[0]
    df[index_col] = pd.to_datetime(df[index_col], errors="coerce")
    df = df.set_index(index_col)
    df.index.name = "date"

    for col in ("crash_prob", "label", "n_news", "n_edges", "rationale", "label_true"):
        if col not in df.columns:
            df[col] = pd.NA

    numeric = ["crash_prob", "label", "n_news", "n_edges", "label_true"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["rationale"] = df["rationale"].fillna("").astype(str)
    return df.sort_index()


def summarize_predictions(df: pd.DataFrame) -> dict:
    """Headline numbers for a predictions DataFrame (pure, no plotting).

    Returns latest crash_prob, peak crash_prob (+ its date), the number of
    actual-crash days, total days, and a simple AUROC if both crash_prob and
    label_true are available and the labels are not all one class.
    """
    out: dict = {
        "n_days": int(len(df)),
        "latest_date": None,
        "latest_prob": None,
        "peak_prob": None,
        "peak_date": None,
        "n_true_crash": None,
        "auroc": None,
    }
    if df.empty:
        return out

    probs = df["crash_prob"].dropna()
    if not probs.empty:
        out["latest_date"] = df.index[-1]
        out["latest_prob"] = float(df["crash_prob"].iloc[-1])
        out["peak_prob"] = float(probs.max())
        out["peak_date"] = probs.idxmax()

    truth = df["label_true"].dropna()
    if not truth.empty:
        out["n_true_crash"] = int((truth == 1).sum())

    out["auroc"] = compute_auroc(df)
    return out


def compute_auroc(df: pd.DataFrame) -> Optional[float]:
    """Rank-based AUROC of crash_prob vs label_true; None if not computable.

    Implemented from scratch (Mann-Whitney rank statistic) so the webapp has no
    scikit-learn dependency. Returns None when label_true is missing or the
    labels are a single class.
    """
    sub = df[["crash_prob", "label_true"]].dropna()
    if sub.empty:
        return None
    y = (sub["label_true"] == 1).to_numpy()
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    scores = sub["crash_prob"].to_numpy()
    order = scores.argsort(kind="mergesort")
    ranks = pd.Series(range(1, len(scores) + 1), dtype=float)
    # Average ranks for ties so the statistic is exact.
    s_sorted = pd.Series(scores[order])
    avg_ranks = s_sorted.groupby(s_sorted).transform(
        lambda g: ranks[g.index].mean()
    )
    rank_of = pd.Series(index=order, data=avg_ranks.to_numpy()).sort_index()
    sum_pos_ranks = float(rank_of.to_numpy()[y].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# ===========================================================================
# (b) Timeline figure: crash_prob over time + shaded actual-crash days.
# ===========================================================================
def build_timeline_figure(df: pd.DataFrame, title: str = "") -> go.Figure:
    """Plotly timeline of crash_prob with actual-crash days shaded red.

    - Blue line + fill: predicted crash probability per day.
    - Red translucent vertical bands: days where label_true == 1.
    - Dashed line: the 0.5 alert threshold.
    Pure — returns a Figure, renders nothing.
    """
    fig = go.Figure()

    if not df.empty and df["crash_prob"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["crash_prob"],
                mode="lines",
                name="Predicted crash prob.",
                line=dict(color="#2563eb", width=2),
                fill="tozeroy",
                fillcolor="rgba(37,99,235,0.12)",
                hovertemplate="%{x|%Y-%m-%d}<br>crash prob = %{y:.2f}<extra></extra>",
            )
        )

    # Shade contiguous runs of actual-crash days as vrects.
    truth = df["label_true"]
    if truth.notna().any():
        crash_idx = list(df.index[truth == 1])
        for d in crash_idx:
            fig.add_vrect(
                x0=d,
                x1=d,
                line_width=6,
                line_color="rgba(220,38,38,0.22)",
                layer="below",
            )
        # Single legend proxy for the shaded crash days.
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color="rgba(220,38,38,0.5)", symbol="square"),
                name="Actual crash day",
            )
        )

    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="#9ca3af",
        annotation_text="alert threshold",
        annotation_position="top left",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Crash probability",
        yaxis=dict(range=[0, 1.02]),
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    return fig


def load_price_series(ticker: str) -> Optional[pd.DataFrame]:
    """Load a local price series (date, close) for a stock ticker, or None.

    Pure helper used by the price overlay in the UI; safe headless.
    """
    path = os.path.join(PRICES_DIR, f"{ticker}.csv")
    if not os.path.isfile(path):
        return None
    px = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return px


# ===========================================================================
# (c) Live demo impact graph from TRRPipeline(MockLLM) over sample news.
# ===========================================================================
@lru_cache(maxsize=8)
def _sample_news_by_day():
    """Cached load of the bundled synthetic demo corpus, grouped by day."""
    from trr.news import group_by_day, load_sample_news

    return group_by_day(load_sample_news())


def list_demo_days() -> list:
    """Calendar days that have sample news, busiest first (for a day picker)."""
    by_day = _sample_news_by_day()
    return sorted(by_day, key=lambda d: (-len(by_day[d]), d))


def build_impact_graph_data(day=None, top_k: int = 30) -> dict:
    """Run the TRR Brainstorm phase (MockLLM) and return graph data for viz.

    Builds the directed impact graph for one demo day's news using
    ``TRRPipeline``'s components with the heuristic ``MockLLM`` (no GPU), prunes
    it with the same PageRank attention the pipeline uses, lays it out with
    networkx, and returns plain dicts ready for Plotly:

        {
          "day": date,
          "n_news": int, "n_edges": int,
          "nodes": [{"id","label","kind","x","y","score"}...],
          "edges": [{"src","dst","polarity","weight"}...],
          "crash_prob": float, "rationale": str,
        }

    ``kind`` is one of: "portfolio" (a portfolio asset), "news" (a source
    article node), or "entity" (an intermediary). Pure: returns data only.
    """
    from trr.attention import pagerank_prune
    from trr.brainstorm import build_impact_graph
    from trr.llm import MockLLM
    from trr.reason import reason_crash
    from trr.schema import PORTFOLIO

    by_day = _sample_news_by_day()
    days = sorted(by_day)
    if day is None:
        # Default to the busiest day so the demo graph is non-trivial.
        day = list_demo_days()[0]
    elif day not in by_day:
        # Tolerate a string/date that isn't present: fall back to the first day.
        day = days[0]

    day_news = by_day[day]
    llm = MockLLM()
    graph = build_impact_graph(day_news, llm, PORTFOLIO)
    pruned = pagerank_prune(list(graph.edges), PORTFOLIO, top_k=top_k)

    # Reason over the pruned subgraph for the headline crash probability.
    crash_prob, rationale = reason_crash(pruned, llm, context="", universe=PORTFOLIO)

    portfolio_set = set(PORTFOLIO)
    g = nx.DiGraph()
    for e in pruned:
        g.add_edge(e.subject, e.object, polarity=int(e.polarity), weight=float(e.weight))
    # Ensure isolated portfolio anchors still show if referenced.
    for node in graph.nodes:
        if node not in g:
            g.add_node(node)

    # PageRank score for sizing nodes (undirected fallback for tiny graphs).
    try:
        pr = nx.pagerank(g) if g.number_of_edges() else {n: 1.0 for n in g.nodes}
    except Exception:
        pr = {n: 1.0 for n in g.nodes}

    pos = nx.spring_layout(g, seed=42, k=0.9) if g.number_of_nodes() else {}

    def kind_of(node: str) -> str:
        if node in portfolio_set:
            return "portfolio"
        if node.startswith("NEWS:"):
            return "news"
        return "entity"

    def label_of(node: str) -> str:
        return node[len("NEWS:"):] if node.startswith("NEWS:") else node

    nodes = []
    for node in g.nodes:
        x, y = pos.get(node, (0.0, 0.0))
        nodes.append(
            {
                "id": node,
                "label": label_of(node),
                "kind": kind_of(node),
                "x": float(x),
                "y": float(y),
                "score": float(pr.get(node, 0.0)),
            }
        )

    edges = [
        {
            "src": u,
            "dst": v,
            "polarity": int(d.get("polarity", 0)),
            "weight": float(d.get("weight", 0.0)),
        }
        for u, v, d in g.edges(data=True)
    ]

    return {
        "day": day,
        "n_news": len(day_news),
        "n_edges": len(pruned),
        "nodes": nodes,
        "edges": edges,
        "crash_prob": float(crash_prob),
        "rationale": rationale,
    }


def build_impact_graph_figure(data: dict) -> go.Figure:
    """Render the impact-graph data dict (from build_impact_graph_data) as a
    Plotly network figure. Pure — returns a Figure.

    Node colour encodes kind; node size encodes PageRank. Edge colour encodes
    polarity (green = positive impact, red = negative).
    """
    nodes = {n["id"]: n for n in data["nodes"]}
    fig = go.Figure()

    # Edges first (drawn underneath). One trace per polarity for a clean legend.
    for polarity, colour, name in (
        (-1, "rgba(220,38,38,0.55)", "Negative impact"),
        (1, "rgba(22,163,74,0.55)", "Positive impact"),
    ):
        xs, ys = [], []
        for e in data["edges"]:
            if int(e["polarity"]) != polarity:
                continue
            s, t = nodes.get(e["src"]), nodes.get(e["dst"])
            if not s or not t:
                continue
            xs += [s["x"], t["x"], None]
            ys += [s["y"], t["y"], None]
        if xs:
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(color=colour, width=1.5),
                    hoverinfo="skip", name=name,
                )
            )

    kind_style = {
        "portfolio": ("#1d4ed8", "circle", "Portfolio asset"),
        "news": ("#f59e0b", "square", "News source"),
        "entity": ("#6b7280", "diamond", "Intermediary"),
    }
    for kind, (colour, symbol, name) in kind_style.items():
        members = [n for n in data["nodes"] if n["kind"] == kind]
        if not members:
            continue
        scores = [m["score"] for m in members]
        smax = max(scores) if scores else 1.0
        sizes = [16 + 34 * (s / smax if smax else 0) for s in scores]
        fig.add_trace(
            go.Scatter(
                x=[m["x"] for m in members],
                y=[m["y"] for m in members],
                mode="markers+text",
                text=[m["label"] for m in members],
                textposition="top center",
                textfont=dict(size=11),
                marker=dict(color=colour, size=sizes, symbol=symbol,
                            line=dict(color="white", width=1.5)),
                name=name,
                hovertemplate="%{text}<br>PageRank=%{customdata:.3f}<extra></extra>",
                customdata=scores,
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=520,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# ===========================================================================
# (d) Parse the campaign results tables from reports/RESULTS_TRR.md.
# ===========================================================================
def parse_results_tables(path: str = RESULTS_MD) -> list[dict]:
    """Parse every Markdown table in the results report into structured data.

    Returns a list of ``{"caption", "columns", "rows"}`` dicts in document
    order. ``caption`` is the nearest preceding heading/bold line; cells have
    Markdown bold/asterisk emphasis stripped. Pure file read.
    """
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    def clean(cell: str) -> str:
        cell = cell.strip()
        cell = re.sub(r"\*\*(.+?)\*\*", r"\1", cell)  # bold
        cell = cell.replace("**", "").replace("`", "")
        return cell.strip()

    tables: list[dict] = []
    i = 0
    last_heading = ""
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("#"):
            last_heading = stripped.lstrip("#").strip()
        elif re.match(r"^\*\*.+\*\*\s*$", stripped):
            last_heading = clean(stripped)

        # A table row starts with "|" and the NEXT line is a separator "|---|".
        is_row = stripped.startswith("|")
        next_sep = (
            i + 1 < len(lines)
            and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]) is not None
        )
        if is_row and next_sep:
            header = [clean(c) for c in stripped.strip("|").split("|")]
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                cells = [clean(c) for c in lines[j].strip().strip("|").split("|")]
                # Pad/trim to header width.
                cells = (cells + [""] * len(header))[: len(header)]
                rows.append(cells)
                j += 1
            tables.append({"caption": last_heading, "columns": header, "rows": rows})
            i = j
            continue
        i += 1
    return tables


def results_table_to_df(table: dict) -> pd.DataFrame:
    """Convert one parsed results table dict into a DataFrame (pure helper)."""
    return pd.DataFrame(table["rows"], columns=table["columns"])


# ===========================================================================
# Headless smoke test.
# ===========================================================================
def _smoke() -> int:
    """Exercise every public function against the repo's data. Returns 0 on OK."""
    print("== webapp.lib smoke test ==")

    runs = list_prediction_runs()
    print(f"[a] list_prediction_runs -> {len(runs)} runs; first: {runs[0] if runs else None}")
    assert runs, "no prediction runs found under kaggle/out_*/crash/"

    df = load_predictions(runs[0]["path"])
    print(f"[a] load_predictions({runs[0]['slug']}) -> {df.shape}, cols={list(df.columns)}")
    assert {"crash_prob", "label_true", "rationale"} <= set(df.columns)

    summ = summarize_predictions(df)
    print(f"[a] summarize_predictions -> latest={summ['latest_prob']}, "
          f"peak={summ['peak_prob']}, true_crash={summ['n_true_crash']}, auroc={summ['auroc']}")

    fig = build_timeline_figure(df, title="smoke")
    print(f"[b] build_timeline_figure -> Figure with {len(fig.data)} traces")
    assert len(fig.data) >= 1

    px = load_price_series(PRICE_TICKERS[0])
    print(f"[b] load_price_series({PRICE_TICKERS[0]}) -> "
          f"{None if px is None else px.shape}")

    days = list_demo_days()
    print(f"[c] list_demo_days -> {len(days)} days; busiest={days[0]}")
    gdata = build_impact_graph_data()
    print(f"[c] build_impact_graph_data -> day={gdata['day']}, "
          f"nodes={len(gdata['nodes'])}, edges={len(gdata['edges'])}, "
          f"crash_prob={gdata['crash_prob']:.2f}")
    assert gdata["nodes"], "impact graph produced no nodes"
    gfig = build_impact_graph_figure(gdata)
    print(f"[c] build_impact_graph_figure -> Figure with {len(gfig.data)} traces")
    assert len(gfig.data) >= 1

    tables = parse_results_tables()
    print(f"[d] parse_results_tables -> {len(tables)} tables")
    assert tables, "no results tables parsed"
    print(f"[d] first table caption='{tables[0]['caption']}', "
          f"cols={tables[0]['columns']}, rows={len(tables[0]['rows'])}")
    rdf = results_table_to_df(tables[0])
    print(f"[d] results_table_to_df -> {rdf.shape}")

    print("== ALL OK ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke())
