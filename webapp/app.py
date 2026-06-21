"""TRR Stock-Crash-Prediction — Streamlit web platform.

Live-first layout:
  Tab 1 "🔴 Live & Advisory"  — the MAIN view: live market status, crash gauge,
                                daily advisory (risk, exposed assets, drivers, cautions).
  Tab 2 "📊 Research & Backtest" — historical labeled runs, impact graph, timeline,
                                campaign results, figures (where the rigorous numbers live).
  Tab 3 "ℹ️ How it works"     — the 4-phase TRR method + honesty notes.

All data/figure logic lives in webapp.lib (pure, headless-testable).
Run:  cd /home/nduong/dev/bigdata && .venv/bin/streamlit run webapp/app.py
"""
from __future__ import annotations

import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from webapp import lib

st.set_page_config(page_title="TRR — Stock Crash Radar", page_icon="📉",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """<style>
      .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
      .stMetric {background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;
                 padding:0.5rem 0.8rem;}
      h1,h2,h3 {letter-spacing:-0.01em;}
    </style>""", unsafe_allow_html=True)

_RISK_COLOR = {"HIGH": "#dc2626", "ELEVATED": "#d97706", "LOW": "#16a34a"}
_DR_PATH = os.path.join(_REPO_ROOT, "data", "live", "daily_report.json")
_FIG_DIR = os.path.join(_REPO_ROOT, "reports", "figures")


# --- cached wrappers --------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_predictions(path):
    return lib.load_predictions(path)


@st.cache_data(show_spinner=False)
def _results_tables():
    return lib.parse_results_tables()


@st.cache_data(show_spinner=True)
def _graph_data(day, top_k):
    return lib.build_impact_graph_data(day=day, top_k=top_k)


def gauge(prob, title):
    p = max(0.0, min(1.0, prob)) * 100
    colour = "#dc2626" if p >= 60 else "#d97706" if p >= 30 else "#16a34a"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p, number={"suffix": "%"},
        title={"text": title, "font": {"size": 15}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": colour},
               "steps": [{"range": [0, 30], "color": "#dcfce7"},
                         {"range": [30, 60], "color": "#fef9c3"},
                         {"range": [60, 100], "color": "#fee2e2"}],
               "threshold": {"line": {"color": "#111", "width": 3},
                             "thickness": 0.75, "value": 50}}))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=50, b=10))
    return fig


# ===========================================================================
st.sidebar.title("📉 TRR Crash Radar")
st.sidebar.caption("Zero-shot LLM reads financial **news** → portfolio **crash** "
                   "probability over the next ~3 trading days.")
runs = lib.list_prediction_runs()
st.sidebar.markdown("---")
st.sidebar.caption("Live tab = current market (yfinance + local model). "
                   "Research tab = labeled historical backtests (where AUROC lives).")

st.title("Stock Crash Radar")
tab_live, tab_research, tab_how = st.tabs(
    ["🔴 Live & Advisory", "📊 Research & Backtest", "ℹ️ How it works"])

# ===========================================================================
# TAB 1 — LIVE & ADVISORY (the main view)
# ===========================================================================
with tab_live:
    # ---- Daily advisory hero ----
    st.subheader("📋 Daily advisory")
    if os.path.exists(_DR_PATH):
        adv = json.load(open(_DR_PATH))
        col_g, col_a = st.columns([1, 1.6])
        with col_g:
            st.plotly_chart(gauge(adv["crash_prob"], "Crash prob · next ~3d"),
                            width="stretch")
            lvl = adv["risk_level"]
            st.markdown(f"### Risk: <span style='color:{_RISK_COLOR.get(lvl,'#666')}'>"
                        f"{lvl}</span>", unsafe_allow_html=True)
            st.caption(f"as of {adv.get('asof','?')} · {adv.get('backend','?')}")
        with col_a:
            if adv.get("at_risk_assets"):
                ar = adv["at_risk_assets"]
                bar = go.Figure(go.Bar(
                    x=[a["exposure"] for a in ar], y=[a["ticker"] for a in ar],
                    orientation="h", marker_color="#dc2626"))
                bar.update_layout(height=180, title="Most-exposed assets",
                                  margin=dict(l=10, r=10, t=40, b=10),
                                  xaxis_title="negative-impact weight")
                st.plotly_chart(bar, width="stretch")
            if adv.get("top_drivers"):
                st.write("**Key drivers:** " + "  ·  ".join(
                    f"{d['subject']}→{d['object']}" for d in adv["top_drivers"]))
        st.markdown("**Cautions**")
        for c in adv.get("cautions", []):
            st.write(f"- {c}")
        if adv.get("rationale"):
            st.caption("Rationale: " + str(adv["rationale"])[:300])
        st.caption("⚠️ " + adv.get("disclaimer", "Research output — not financial advice."))
    else:
        st.info("No daily report yet — run `.venv/bin/python -m scripts.daily_report "
                "[--backend 7b]` (cron once a day).")

    st.markdown("---")
    st.subheader("📡 Live market monitor")
    st.caption("Auto-refreshes from yfinance + local model. Deployment demo — live "
               "news is unlabeled, so rigorous AUROC is in the Research tab.")
    _interval = st.selectbox("Auto-refresh", [30, 60, 120, 300], index=1,
                             format_func=lambda s: f"every {s}s")

    @st.fragment(run_every=_interval)
    def _live_panel():
        from webapp import live as _live
        try:
            snap = _live.read_daemon_snapshot() or _live.live_snapshot(use_local_7b=False)
            sig = snap["signal"]
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.plotly_chart(gauge(sig["crash_prob"], "Live crash prob"),
                                width="stretch")
            c2.metric("Portfolio move (1d)", f"{snap['portfolio_move']:+.2%}")
            c2.metric("Live headlines", sig["n_news"])
            c3.metric("Impact edges", sig["n_edges"])
            c3.caption(f"{sig.get('backend','?')} · {sig['asof']}")
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
    if st.button("↻ Run once with local Qwen-7B + RAG (real LLM, ~1–3 min)"):
        with st.spinner("Loading 7B-AWQ + reasoning over live news…"):
            try:
                from webapp import live as _live
                sig = _live.run_live(_live.fetch_live_headlines(),
                                     use_local_7b=True, use_rag=True)
                st.success(f"7B crash_prob {sig['crash_prob']:.0%} · "
                           f"{sig['n_edges']} edges · {sig['backend']}")
                st.caption(sig["rationale"][:200])
            except Exception as exc:  # noqa: BLE001
                st.warning(f"7B run failed: {exc}")

# ===========================================================================
# TAB 2 — RESEARCH & BACKTEST
# ===========================================================================
with tab_research:
    if not runs:
        st.info("No prediction runs under kaggle/out_*/crash/.")
    else:
        labels = [f"{r['label']}  ·  out_{r['slug']}" for r in runs]
        sel = st.selectbox("Backtest run (labeled, historical)",
                           options=range(len(runs)), format_func=lambda i: labels[i])
        run = runs[sel]
        df = _load_predictions(run["path"])
        summ = lib.summarize_predictions(df)
        st.caption("NOT live — a labeled past run; AUROC needs known outcomes.")
        c1, c2, c3, c4 = st.columns(4)
        ld = summ["latest_date"]
        c1.metric("Latest day", ld.date().isoformat() if ld is not None else "—")
        c2.metric("Days", summ["n_days"])
        c3.metric("Actual crash days", summ["n_true_crash"])
        c4.metric("AUROC", f"{summ['auroc']:.3f}" if summ.get("auroc") else "—")
        st.plotly_chart(lib.build_timeline_figure(df, run["label"]), width="stretch")

        with st.expander("🕸️ Live impact graph (Brainstorm → Attention, sample news)"):
            try:
                st.plotly_chart(lib.build_impact_graph_figure(_graph_data(None, 30)),
                                width="stretch")
            except Exception as exc:  # noqa: BLE001
                st.caption(f"graph unavailable: {exc}")

        with st.expander("🧠 Per-day rationale"):
            if "rationale" in df.columns:
                d = st.select_slider("Day", options=list(df.index),
                                     value=df.index[-1])
                row = df.loc[d]
                st.metric("crash_prob", f"{row['crash_prob']:.2f}")
                st.write(f"> {row.get('rationale','')}")

        st.markdown("### Campaign results")
        for table in _results_tables()[:3]:
            st.caption(table["caption"] or "")
            st.dataframe(lib.results_table_to_df(table), width="stretch",
                         hide_index=True)
        st.markdown("### Figures")
        for f in ["campaign_auroc.png", "reliability.png", "backtest_equity.png"]:
            p = os.path.join(_FIG_DIR, f)
            if os.path.exists(p):
                st.image(p, width="stretch")

# ===========================================================================
# TAB 3 — HOW IT WORKS
# ===========================================================================
with tab_how:
    st.subheader("Temporal Relational Reasoning (TRR)")
    st.markdown(
        "A zero-shot LLM reads financial **news** and predicts the probability of "
        "a portfolio **crash** over the next ~3 trading days, via four phases:\n\n"
        "1. **Brainstorm** — news → directed impact graph (entities → assets)\n"
        "2. **Memory** — time-decay `R=exp(-t·λ)` carries impacts across days (temporal)\n"
        "3. **Attention** — PageRank prunes to the portfolio-relevant sub-graph (relational)\n"
        "4. **Reason** — the LLM outputs a crash probability + rationale\n")
    st.markdown(
        "**Architecture:** heavy **Qwen2.5-32B offline on Kaggle** = the validated "
        "quality predictor; **local Qwen2.5-7B-AWQ on a 2060 + RAG** = the live "
        "deployment. **RAG** (retrieving similar labeled past days) is the most "
        "robust enhancement (+0.06–0.07 AUROC).")
    st.markdown(
        "**Honesty:** rigorous AUROC comes from labeled historical backtests "
        "(Research tab). Live news is unlabeled, so the Live tab proves *deployment*, "
        "not accuracy. Raw price/direction is ~chance (EMH); tail-risk/crash is the "
        "feasible target. Not financial advice.")
