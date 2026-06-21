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

# The 7B summarizer loads lazily in a background thread the first time the Live
# tab is viewed (so app restart stays fast). Until it's ready the panel shows the
# instant rule-based summary, then the LLM summary swaps in on a later refresh.

st.markdown(
    """<style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
      html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
      .block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1300px;}
      /* clean light metric cards with a green left edge */
      [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid #16a34a;
        border-radius: 12px; padding: 0.7rem 1rem;
        box-shadow: 0 1px 3px rgba(15,23,42,0.06);
      }
      [data-testid="stMetricValue"] { font-weight: 800; color:#0f172a; }
      h1 { font-weight: 800; letter-spacing:-0.02em; color:#0f172a; }
      h1 span.accent { color:#16a34a; }
      h2,h3 { letter-spacing:-0.01em; font-weight:700; color:#0f172a; }
      /* tabs: pill-like, green active underline */
      .stTabs [data-baseweb="tab-list"] { gap: 4px; }
      .stTabs [data-baseweb="tab"] { background:#f1f5f9; border-radius:10px 10px 0 0;
        padding: 8px 16px; }
      .stTabs [aria-selected="true"] { background:#ffffff;
        border-bottom: 3px solid #16a34a; font-weight:700; }
      .stButton>button { border-radius:10px; border:1px solid #16a34a;
        background:#16a34a; color:#fff; font-weight:600; }
      .stButton>button:hover { background:#15803d; border-color:#15803d; }
      a { color:#15803d; }
      hr { border-color:#e2e8f0; }
      /* SMOOTH auto-refresh: never dim/blur stale content on rerun */
      [data-stale="true"], [data-stale="true"] * {
        opacity: 1 !important; filter: none !important; transition: none !important; }
      [data-testid="stStatusWidget"] { display: none !important; }   /* hide rerun 'Running…' (keep st.spinner for explicit heavy runs) */
      /* live content eases in instead of flashing */
      [data-testid="stMetric"], .stPlotlyChart { transition: opacity .5s ease; }
      @keyframes fadein { from { opacity: .55; } to { opacity: 1; } }
      /* LIVE pulsing indicator (shows the monitor is actively analysing) */
      .live-badge { display:inline-flex; align-items:center; gap:7px;
        background:#ecfdf5; color:#15803d; border:1px solid #86efac;
        border-radius:999px; padding:3px 12px; font-size:0.8rem; font-weight:700; }
      .live-dot { width:9px; height:9px; border-radius:50%; background:#16a34a;
        animation: pulse 1.3s infinite; }
      @keyframes pulse {
        0%   { box-shadow:0 0 0 0 rgba(22,163,74,.65); }
        70%  { box-shadow:0 0 0 9px rgba(22,163,74,0); }
        100% { box-shadow:0 0 0 0 rgba(22,163,74,0); } }
      /* spinning gear for heavy (7B / Kaggle) analysis */
      .analyzing { display:inline-flex; align-items:center; gap:7px; color:#b45309;
        font-weight:600; }
      .spin { display:inline-block; animation: spin 1s linear infinite; }
      @keyframes spin { to { transform: rotate(360deg); } }
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
        mode="gauge+number", value=p, number={"suffix": "%", "font": {"color": "#0f172a"}},
        title={"text": title, "font": {"size": 14, "color": "#64748b"}},
        gauge={"axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
               "bar": {"color": colour},
               "bgcolor": "rgba(0,0,0,0)",
               "steps": [{"range": [0, 30], "color": "rgba(22,163,74,0.15)"},
                         {"range": [30, 60], "color": "rgba(217,119,6,0.15)"},
                         {"range": [60, 100], "color": "rgba(220,38,38,0.15)"}],
               "threshold": {"line": {"color": "#0f172a", "width": 3},
                             "thickness": 0.75, "value": 50}}))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": "#0f172a"})
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
    st.subheader("📡 Live market monitor — giám sát & tóm tắt tin")
    st.caption("**Descriptive, not a prediction.** This panel monitors + summarizes the "
               "live news flow (recency-weighted) — it does NOT output a 3-day crash "
               "probability. The crash **prediction** is the **Daily advisory** above "
               "(daily cadence = the 3-day horizon). Deployment demo (live news unlabeled).")
    _interval = st.selectbox("Auto-refresh", [30, 60, 120, 300], index=1,
                             format_func=lambda s: f"every {s}s")
    _stress_color = {"High": "#dc2626", "Elevated": "#d97706", "Low": "#16a34a"}

    @st.fragment(run_every=_interval)
    def _live_panel():
        import datetime as _dt

        from webapp import live as _live
        try:
            now = _dt.datetime.now(_dt.timezone.utc).strftime("%H:%M:%S")
            prices, pmove = _live.fetch_live_prices()
            # Read the daemon's cached summary (instant — the 7B lives in the
            # long-running daemon, never in the web). Fall back to an instant
            # rule-based summary only if the daemon isn't running.
            snap = _live.read_live_summary()
            if snap:
                summary, source = snap["summary"], snap["source"]
                stress, neg, recent = snap["stress"], snap["neg_ratio"], snap["recent_count"]
                nheads, tickers, top_recent = snap["n_headlines"], snap["top_tickers"], snap["top_recent"]
                if snap.get("stale"):
                    source += " · stale"
            else:
                heads = _live.fetch_live_headlines(
                    _live.FEED_TICKERS, max_per=6, include_macro=True,
                    include_crypto=True, include_world=True)
                res = _live.summarize_live_news(heads, use_llm=False)
                s = res["signals"]
                summary, source = res["summary"], "rule-based (daemon offline)"
                stress, neg, recent = s["stress"], s["neg_ratio"], s["recent_count"]
                nheads, tickers = len(heads), s["top_tickers"]
                top_recent = [{"ticker": (it.assets[0] if it.assets else "—"),
                               "title": it.title} for it in s["top_recent"]]
            st.markdown(
                f"<span class='live-badge'><span class='live-dot'></span>LIVE</span> "
                f"<span style='color:#64748b;font-size:0.82rem'>monitoring news · "
                f"auto-refresh {_interval}s · updated {now} UTC</span>",
                unsafe_allow_html=True)
            st.markdown(f"**📝 News summary (7B, recency-weighted):** {summary}")
            col = _stress_color.get(stress, "#475569")
            st.markdown(
                f"News-stress: <b style='color:{col}'>{stress}</b> "
                f"<span style='color:#64748b'>(neg {neg:.0%}; source: {source})</span>",
                unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Portfolio move (1d)", f"{pmove:+.2%}")
            c2.metric("Live headlines", nheads)
            c3.metric("Recent (≤6h)", recent)
            if tickers:
                st.caption("Focus (recency-weighted): "
                           + " · ".join(str(t[0]) for t in tickers))
            pc = st.columns(6)
            for i, (tk, row) in enumerate(prices.items()):
                pc[i % 6].metric(tk, row["price"],
                                 f"{row['ret_1d']:+.2%}" if row["ret_1d"] is not None else "—")
            with st.expander("Latest headlines (salient, recency-weighted)"):
                for h in top_recent:
                    st.write(f"**[{h['ticker']}]** {h['title']}")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Live fetch unavailable (need internet / yfinance): {exc}")

    _live_panel()

    st.markdown("#### 📰 Live news feed")
    st.caption("Company + **macro** headlines (Fed, rates, VIX, geopolitics), "
               "newest first, sentiment-tinted, with 🆕 badges. **Display only** — "
               "high-frequency news shown for context; the crash *prediction* stays "
               "the daily 3-day advisory above (we don't predict at this cadence).")
    _NEG = ("crash slump plunge fall drop fear hack ban lawsuit selloff sell-off "
            "tumble sink slide warn cut loss recession halt panic downgrade").split()
    _POS = ("surge soar rally gain jump rise beat upgrade record high boom approve "
            "win growth profit strong buy bullish").split()
    fc1, fc2, fc3 = st.columns([1, 1, 1])
    _feed_rate = fc1.selectbox("Feed refresh", [10, 15, 30, 60], index=1,
                               format_func=lambda s: f"every {s}s", key="feedrate")
    _filter = fc2.selectbox("Filter", ["All", "🏢 Companies", "🌐 Macro", "₿ Crypto",
                                       "🌍 World"], key="feedfilter")
    _show_n = fc3.slider("Show last", 20, 500, 200, 20, key="feedshow",
                         help="Display only — does not affect the prediction "
                              "(the model uses ~20–40/day regardless).")

    @st.fragment(run_every=_feed_rate)
    def _news_feed():
        from webapp import live as _live
        try:
            heads = _live.fetch_live_headlines(tickers=_live.FEED_TICKERS,
                                               include_macro=True, include_crypto=True,
                                               include_world=True, max_per=10)
            # ACCUMULATE across refreshes: new headlines append into a persistent
            # store (keyed by title), newest kept on top, capped at 500.
            store = st.session_state.get("feed_store", {})
            fresh = 0
            for h in heads:
                k = h.title
                if k not in store:
                    fresh += 1
                    store[k] = {"ts": h.timestamp, "tag": h.assets[0],
                                "title": h.title, "src": h.source, "new": True}
                else:
                    store[k]["new"] = False
            items = sorted(store.values(), key=lambda r: r["ts"], reverse=True)[:500]
            st.session_state["feed_store"] = {r["title"]: r for r in items}
            def _kind(tag):
                return ("MACRO" if tag.startswith("MACRO") else
                        "CRYPTO" if tag.startswith("CRYPTO") else
                        "WORLD" if tag == "WORLD" else "TICKER")
            view = items
            fmap = {"🌐 Macro": "MACRO", "₿ Crypto": "CRYPTO", "🌍 World": "WORLD",
                    "🏢 Companies": "TICKER"}
            if _filter in fmap:
                view = [r for r in items if _kind(r["tag"]) == fmap[_filter]]
            st.caption(f"showing {min(_show_n, len(view))} of {len(view)} accumulated "
                       f"· 🆕 {fresh} new this refresh · source: Yahoo Finance (yfinance) "
                       f"· display only (model predicts on ~20–40/day)")
            _STYLE = {"MACRO": ("🌐", "#fef3c7", "#b45309"),
                      "CRYPTO": ("₿", "#fae8ff", "#a21caf"),
                      "WORLD": ("🌍", "#dbeafe", "#1d4ed8"),
                      "TICKER": ("🏢", "#eef2ff", "#3730a3")}
            for r in view[:_show_n]:
                tag = r["tag"]; icon, bg, col = _STYLE[_kind(tag)]
                low = r["title"].lower()
                senti = ("#dc2626" if any(w in low for w in _NEG) else
                         "#16a34a" if any(w in low for w in _POS) else "#334155")
                badge = ("<span style='background:#dc2626;color:#fff;border-radius:4px;"
                         "padding:0 5px;font-size:0.66rem'>🆕</span> " if r.get("new") else "")
                t = r["ts"].strftime("%m-%d %H:%M")
                st.markdown(
                    f"<div style='padding:3px 0;border-bottom:1px solid #eef;"
                    f"animation:fadein .5s ease'>"
                    f"<span style='color:#94a3b8;font-size:0.76rem'>{t}</span> "
                    f"<span style='background:{bg};color:{col};border-radius:6px;"
                    f"padding:1px 7px;font-size:0.7rem;font-weight:600'>"
                    f"{icon} {tag}</span> {badge}"
                    f"<span style='color:{senti}'>{r['title']}</span> "
                    f"<span style='color:#cbd5e1;font-size:0.72rem'>· {r['src']}</span></div>",
                    unsafe_allow_html=True)
        except Exception as exc:  # noqa: BLE001
            st.caption(f"news feed unavailable: {exc}")

    _news_feed()

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

    st.markdown("---")
    st.subheader("🔬 Try it — type a headline, watch TRR react")
    st.caption("Type any market news; the pipeline extracts an impact graph and "
               "predicts crash risk on YOUR input (heuristic MockLLM, instant).")
    _txt = st.text_area("News headline(s), one per line",
                        "Major exchange hacked; contagion fears; cascading "
                        "liquidations hit BTC and ETH", height=70)
    _assets = st.multiselect("Portfolio assets",
                             ["AAPL", "AMZN", "GOOGL", "NVDA", "TSLA", "NFLX",
                              "BTC", "ETH", "SOL", "BNB"], default=["BTC", "ETH"])
    if st.button("⚡ Analyze"):
        from datetime import datetime, timezone
        from trr.schema import NewsItem
        items = [NewsItem(id=f"u{i}", timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
                          title=t.strip(), assets=_assets or ["BTC", "ETH"])
                 for i, t in enumerate(_txt.splitlines()) if t.strip()]
        if items:
            d = lib.build_impact_graph_data(news_items=items)
            cc1, cc2 = st.columns([1, 1.5])
            with cc1:
                st.plotly_chart(gauge(d["crash_prob"], "Crash prob (your input)"),
                                width="stretch")
            with cc2:
                try:
                    st.plotly_chart(lib.build_impact_graph_figure(d), width="stretch")
                except Exception as exc:  # noqa: BLE001
                    st.caption(f"graph: {exc}")
            st.caption(f"{d['n_edges']} impact edges · rationale: "
                       + str(d.get("rationale", ""))[:200])

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
        st.plotly_chart(lib.build_animated_timeline_figure(df, run["label"]),
                        width="stretch")
        st.caption("▶ Press Play to replay the crash radar day by day.")

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
        st.markdown("### Figures (interactive)")
        _fd = lib.load_fig_data()
        if _fd:
            st.plotly_chart(lib.build_campaign_figure(_fd), width="stretch")
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                st.plotly_chart(lib.build_reliability_figure(_fd), width="stretch")
            with fcol2:
                st.plotly_chart(lib.build_backtest_figure(_fd), width="stretch")
        else:
            st.info("Run `.venv/bin/python -m train.figures` to generate figure data.")

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
