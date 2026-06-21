"""TRR Stock-Crash-Prediction — Streamlit web platform.

A thin UI shell over ``webapp.lib`` (all data/figure logic lives there, pure and
headless-testable). Sections:

  1. Crash-risk gauge / big number for the selected run + latest day.
  2. Live impact-graph viz (Brainstorm -> Attention) from MockLLM over sample news.
  3. Historical crash-probability timeline vs actual crash days (+ price overlay).
  4. Per-day rationale browser (the LLM's reasoning text).
  5. Campaign / backtest results tables parsed from reports/RESULTS_TRR.md.

Run:  cd /home/nduong/dev/bigdata && .venv/bin/streamlit run webapp/app.py
"""
from __future__ import annotations

import os
import sys

# Make the repo root importable (so `import trr` and `import webapp.lib` work
# regardless of how Streamlit is launched).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from webapp import lib

st.set_page_config(
    page_title="TRR — Stock Crash Radar",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Light custom styling ---------------------------------------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem;}
      .stMetric {background: #f8fafc; border: 1px solid #e2e8f0;
                 border-radius: 12px; padding: 0.6rem 0.9rem;}
      h1, h2, h3 {letter-spacing: -0.01em;}
      .trr-pill {display:inline-block; padding:2px 10px; border-radius:999px;
                 font-size:0.78rem; font-weight:600; margin-right:6px;}
      .trr-phase {background:#eef2ff; color:#3730a3;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Cached wrappers around the pure lib functions --------------------------
@st.cache_data(show_spinner=False)
def _load_predictions(path: str) -> pd.DataFrame:
    return lib.load_predictions(path)


@st.cache_data(show_spinner=False)
def _results_tables():
    return lib.parse_results_tables()


@st.cache_data(show_spinner=False)
def _demo_days():
    return lib.list_demo_days()


@st.cache_data(show_spinner=True)
def _graph_data(day, top_k: int):
    return lib.build_impact_graph_data(day=day, top_k=top_k)


@st.cache_data(show_spinner=False)
def _price(ticker: str):
    return lib.load_price_series(ticker)


# ===========================================================================
# Sidebar — run + day selection.
# ===========================================================================
runs = lib.list_prediction_runs()

st.sidebar.title("📉 TRR Crash Radar")
st.sidebar.caption(
    "Zero-shot LLM pipeline that reads financial **news** and predicts portfolio "
    "**crash** probability."
)

if not runs:
    st.error("No prediction runs found under kaggle/out_*/crash/. Run the TRR "
             "pipeline first.")
    st.stop()

run_labels = [f"{r['label']}  ·  out_{r['slug']}" for r in runs]
sel = st.sidebar.selectbox("Prediction run", options=range(len(runs)),
                           format_func=lambda i: run_labels[i], index=0)
run = runs[sel]
df = _load_predictions(run["path"])
summ = lib.summarize_predictions(df)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Four phases**  \n"
    "<span class='trr-pill trr-phase'>1 Brainstorm</span>"
    "<span class='trr-pill trr-phase'>2 Memory</span>  \n"
    "<span class='trr-pill trr-phase'>3 Attention</span>"
    "<span class='trr-pill trr-phase'>4 Reason</span>",
    unsafe_allow_html=True,
)
st.sidebar.caption(
    "News → impact graph → time-decay memory → PageRank prune → LLM crash prob."
)

# ===========================================================================
# Header + headline gauge.
# ===========================================================================
st.title("Stock Crash Radar")
st.markdown(
    "Temporal Relational Reasoning (TRR) — an LLM reasons over financial news to "
    "estimate the probability of a near-term portfolio crash. "
    f"Showing run **{run['label']}** (`out_{run['slug']}`)."
)

# ---- LIVE monitor: auto-fetches + auto-refreshes (yfinance, needs internet) ----
st.header("🔴 Live market monitor")
st.caption("Auto-pulls current prices + headlines (yfinance) and runs TRR live, "
           "refreshing on an interval. NOTE: a *runnability/deployment* demo — live "
           "news is sparse/noisy/unlabeled, so the rigorous AUROC numbers come from "
           "the batch evaluation on curated, labeled historical news, not this feed.")

_lc1, _lc2 = st.columns([1, 2])
_interval = _lc1.selectbox("Auto-refresh", [30, 60, 120, 300], index=1,
                           format_func=lambda s: f"every {s}s")
_lc2.caption("The panel below fetches live data automatically and updates itself. "
             "Use the local 7B button at the bottom for a real-LLM (slower) read.")


@st.fragment(run_every=_interval)
def _live_panel():
    from webapp import live as _live
    try:
        snap = _live.read_daemon_snapshot()              # prefer the live daemon if running
        if snap is None:
            snap = _live.live_snapshot(use_local_7b=False)  # else fetch inline (instant heuristic)
        sig = snap["signal"]
        prob = sig["crash_prob"]
        st.metric("🔴 Crash probability — next ~3 trading days", f"{prob:.0%}",
                  delta=f"{snap['portfolio_move']:+.2%} portfolio (1d)",
                  delta_color="inverse")
        st.caption("Horizon is fixed at ~3 trading days (the trained/validated "
                   "target). Frequent polling only refreshes this same 3-day-ahead "
                   "estimate as new headlines arrive — it is NOT a minute/intraday "
                   "forecast (the method is daily-resolution).")
        st.progress(min(prob, 1.0))
        st.caption(f"updated {sig['asof']} · backend {sig.get('backend','?')} · "
                   f"{sig['n_news']} headlines · {sig['n_edges']} impact edges · "
                   f"{sig['rationale'][:110]}")
        pc = st.columns(6)
        for i, (tk, row) in enumerate(snap["prices"].items()):
            pc[i % 6].metric(tk, row["price"],
                             f"{row['ret_1d']:+.2%}" if row["ret_1d"] is not None else "—")
        with st.expander(f"Live headlines ({len(snap['headlines'])})"):
            for h in snap["headlines"]:
                st.write(f"**[{h['ticker']}]** {h['title']}")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Live fetch unavailable (need internet / yfinance): {exc}")


_live_panel()

if st.button("↻ Run once with local Qwen-7B (real LLM on the 2060, ~1–3 min)"):
    with st.spinner("Loading 7B-AWQ + reasoning over live news…"):
        try:
            from webapp import live as _live
            sig = _live.run_live(_live.fetch_live_headlines(), use_local_7b=True, use_rag=True)
            st.success(f"7B live crash_prob {sig['crash_prob']:.0%} · "
                       f"{sig['n_edges']} edges · {sig['backend']}")
            st.caption(sig["rationale"][:200])
        except Exception as exc:  # noqa: BLE001
            st.warning(f"7B run failed: {exc}")
st.markdown("---")

st.header("📋 Daily advisory")
st.caption("The feasible product: a once-a-day TRR analysis (run on Kaggle 32B or "
           "local 7B) — risk level, most-exposed assets, key drivers, cautions. "
           "Research/analysis, not financial advice.")
_dr_path = os.path.join(_REPO_ROOT, "data", "live", "daily_report.json")
if os.path.exists(_dr_path):
    import json as _json
    adv = _json.load(open(_dr_path))
    _c = {"HIGH": "#dc2626", "ELEVATED": "#d97706", "LOW": "#16a34a"}.get(adv["risk_level"], "#666")
    st.markdown(f"### Risk: <span style='color:{_c}'>{adv['risk_level']}</span> "
                f"· crash prob {adv['crash_prob']:.0%} · {adv.get('horizon','')}",
                unsafe_allow_html=True)
    st.caption(f"as of {adv.get('asof','?')} · backend {adv.get('backend','?')} · "
               f"{adv.get('n_headlines','?')} headlines")
    if adv.get("at_risk_assets"):
        st.write("**Most exposed:** " + ", ".join(
            f"{a['ticker']} ({a['exposure']})" for a in adv["at_risk_assets"]))
    if adv.get("top_drivers"):
        st.write("**Key drivers:** " + "; ".join(
            f"{d['subject']}→{d['object']}" for d in adv["top_drivers"]))
    for c in adv.get("cautions", []):
        st.write(f"- {c}")
    if adv.get("rationale"):
        st.caption("Rationale: " + str(adv["rationale"])[:300])
    st.caption(adv.get("disclaimer", ""))
else:
    st.info("No daily report yet — run `.venv/bin/python -m scripts.daily_report "
            "[--backend 7b]` (cron it once a day).")
st.markdown("---")

st.header("📊 Historical backtest (labeled evaluation)")
st.caption(f"This section is NOT live — it replays a past run ({run['label']}, "
           f"`out_{run['slug']}`) with KNOWN outcomes so we can score it (AUROC "
           "needs labels). This is where the rigorous numbers come from; the live "
           "monitor above is the real-time deployment.")

latest_prob = summ["latest_prob"] or 0.0
peak_prob = summ["peak_prob"] or 0.0

g_col, m_col = st.columns([1.1, 1.5])

with g_col:
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=latest_prob * 100,
            number={"suffix": "%", "font": {"size": 46}},
            title={"text": "Latest crash probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1d4ed8"},
                "steps": [
                    {"range": [0, 30], "color": "#dcfce7"},
                    {"range": [30, 60], "color": "#fef9c3"},
                    {"range": [60, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#dc2626", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )
    gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10))
    st.plotly_chart(gauge, width='stretch')

with m_col:
    c1, c2 = st.columns(2)
    latest_date = summ["latest_date"]
    c1.metric("Latest day", latest_date.date().isoformat() if latest_date is not None else "—")
    c1.metric("Days evaluated", summ["n_days"])
    peak_date = summ["peak_date"]
    c2.metric(
        "Peak crash prob.",
        f"{peak_prob:.0%}",
        help=f"on {peak_date.date()}" if peak_date is not None else None,
    )
    if summ["n_true_crash"] is not None:
        c2.metric("Actual crash days", summ["n_true_crash"])
    if summ["auroc"] is not None:
        st.metric("AUROC (crash_prob vs truth)", f"{summ['auroc']:.3f}",
                  help="Rank AUROC over this run's days. 0.5 = chance.")
    risk = ("HIGH" if latest_prob >= 0.6 else "ELEVATED" if latest_prob >= 0.3
            else "LOW")
    colour = {"HIGH": "#dc2626", "ELEVATED": "#d97706", "LOW": "#16a34a"}[risk]
    st.markdown(
        f"#### Risk on this run's last day ({latest_date.date().isoformat() if latest_date is not None else '—'}): "
        f"<span style='color:{colour}'>{risk}</span>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ===========================================================================
# Section 2 — Live impact graph (Brainstorm → Attention).
# ===========================================================================
st.header("Impact graph — live Brainstorm → Attention")
st.caption(
    "Built on the fly with the heuristic **MockLLM** (no GPU) over the bundled "
    "synthetic news corpus. Each news node fans out signed, weighted impact edges "
    "toward portfolio assets; PageRank attention keeps the most relevant subgraph."
)

demo_days = _demo_days()
gc1, gc2 = st.columns([1, 2])
with gc1:
    day = st.selectbox(
        "Demo news day", options=demo_days,
        format_func=lambda d: d.isoformat(), index=0,
    )
    top_k = st.slider("Attention top-k edges", 5, 40, 30, step=5)

gdata = _graph_data(day, top_k)

with gc1:
    st.metric("News items this day", gdata["n_news"])
    st.metric("Edges after prune", gdata["n_edges"])
    st.metric("Reasoned crash prob.", f"{gdata['crash_prob']:.0%}")
    st.caption(f"**Rationale:** {gdata['rationale']}")

with gc2:
    if gdata["nodes"]:
        st.plotly_chart(lib.build_impact_graph_figure(gdata),
                        width='stretch')
    else:
        st.info("No impact edges for this day.")

st.markdown("---")

# ===========================================================================
# Section 3 — Historical timeline.
# ===========================================================================
st.header("Historical crash-probability timeline")
st.caption("Predicted crash probability per day; red bands mark actual crash days.")

st.plotly_chart(
    lib.build_timeline_figure(df, title=f"out_{run['slug']} — crash probability"),
    width='stretch',
)

# Optional price overlay for stock runs.
with st.expander("Overlay a stock price series"):
    ticker = st.selectbox("Ticker", options=["(none)"] + lib.PRICE_TICKERS)
    if ticker != "(none)":
        px = _price(ticker)
        if px is None or px.empty:
            st.info(f"No local price series for {ticker}.")
        else:
            lo, hi = df.index.min(), df.index.max()
            win = px.loc[(px.index >= lo) & (px.index <= hi)]
            pfig = go.Figure()
            pfig.add_trace(go.Scatter(x=win.index, y=win["close"],
                                      mode="lines", name=f"{ticker} close",
                                      line=dict(color="#0f172a")))
            truth = df["label_true"]
            if truth.notna().any():
                for d in df.index[truth == 1]:
                    pfig.add_vrect(x0=d, x1=d, line_width=5,
                                   line_color="rgba(220,38,38,0.22)", layer="below")
            pfig.update_layout(template="plotly_white", height=340,
                               margin=dict(l=40, r=20, t=30, b=40),
                               yaxis_title=f"{ticker} close", xaxis_title="Date")
            st.plotly_chart(pfig, width='stretch')

st.markdown("---")

# ===========================================================================
# Section 4 — Rationale browser.
# ===========================================================================
st.header("Per-day reasoning")
st.caption("The LLM's natural-language rationale behind each day's crash call.")

rat = df[df["rationale"].astype(str).str.len() > 0]
if rat.empty:
    st.info("This run has no rationale text.")
else:
    pick = st.select_slider(
        "Day", options=list(rat.index),
        value=summ["peak_date"] if summ["peak_date"] in rat.index else rat.index[-1],
        format_func=lambda d: d.date().isoformat(),
    )
    row = df.loc[pick]
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Crash prob.", f"{float(row['crash_prob']):.0%}")
    rc2.metric("News items", int(row["n_news"]) if pd.notna(row["n_news"]) else "—")
    rc3.metric("Edges", int(row["n_edges"]) if pd.notna(row["n_edges"]) else "—")
    if pd.notna(row.get("label_true")):
        actual = "CRASH" if int(row["label_true"]) == 1 else "no crash"
        rc3.metric("Actual outcome", actual)
    st.markdown(f"> {row['rationale']}")

    st.dataframe(
        df[["crash_prob", "label", "n_news", "n_edges", "label_true"]],
        width='stretch', height=260,
    )

st.markdown("---")

# ===========================================================================
# Section 5 — Campaign / backtest results.
# ===========================================================================
st.header("Campaign & backtest results")
st.caption("Parsed from reports/RESULTS_TRR.md — the empirical study behind TRR.")

tables = _results_tables()
if not tables:
    st.info("reports/RESULTS_TRR.md not found.")
else:
    captions = [f"{i+1}. {t['caption'] or 'table'}" for i, t in enumerate(tables)]
    idx = st.selectbox("Result table", options=range(len(tables)),
                       format_func=lambda i: captions[i], index=0)
    table = tables[idx]
    st.subheader(table["caption"] or "Results")
    st.dataframe(lib.results_table_to_df(table), width='stretch',
                 hide_index=True)

st.markdown("---")

st.header("Research figures")
st.caption("Generated by train/figures.py (run `bash scripts/run_all.sh`).")
_FIGS = [
    ("campaign_auroc.png", "Campaign: TRR vs news-volume baseline across windows"),
    ("reliability.png", "Calibration — predicted vs observed crash frequency"),
    ("backtest_equity.png", "Economic backtest — de-risk overlay vs buy & hold"),
]
_figdir = os.path.join(_REPO_ROOT, "reports", "figures")
_present = [(f, cap) for f, cap in _FIGS if os.path.exists(os.path.join(_figdir, f))]
if not _present:
    st.info("No figures yet — run `bash scripts/run_all.sh` to generate them.")
else:
    for fname, cap in _present:
        st.image(os.path.join(_figdir, fname), caption=cap, width='stretch')

st.caption(
    "TRR — Temporal Relational Reasoning over financial news. "
    "Crash predictions are research artefacts, not investment advice."
)
