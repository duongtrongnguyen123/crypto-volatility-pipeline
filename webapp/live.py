"""Live market monitor — pulls CURRENT prices + CURRENT news (yfinance) and runs
the TRR pipeline live to produce a real-time crash signal. Needs internet.

Pure functions (no Streamlit) so they're importable/testable headless. The
reasoning backend is MockLLM by default (no GPU/network) — the same 4-phase TRR
logic that the offline 32B uses, just a lighter reasoner for live use.
"""
from __future__ import annotations

from datetime import datetime, timezone

TICKERS = ["AAPL", "AMZN", "GOOGL", "NVDA", "TSLA", "NFLX"]


def fetch_live_headlines(tickers=TICKERS, max_per: int = 6):
    """Current headlines per ticker via yfinance -> list[NewsItem] (today)."""
    import yfinance as yf
    from trr.schema import NewsItem
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    items, seen = [], set()
    for t in tickers:
        try:
            news = getattr(yf.Ticker(t), "news", []) or []
        except Exception:  # noqa: BLE001
            news = []
        for i, it in enumerate(news[:max_per]):
            c = it.get("content", it) if isinstance(it, dict) else {}
            title = (c.get("title") or it.get("title") or "").strip()
            if not title or title in seen:
                continue
            seen.add(title)
            # use the ARTICLE's real publish time so a multi-day window can form
            ts = now
            pub = c.get("pubDate") or c.get("displayTime")
            if pub:
                try:
                    ts = datetime.fromisoformat(str(pub).replace("Z", "+00:00")).replace(tzinfo=None)
                except ValueError:
                    pass
            items.append(NewsItem(id=f"{t}-{i}", timestamp=ts, title=title,
                                  source="yfinance", assets=[t]))
    return items


def fetch_live_prices(tickers=TICKERS):
    """Latest close + 1-day return per ticker + equal-weight portfolio move."""
    import yfinance as yf
    rows, rets = {}, []
    for t in tickers:
        try:
            h = yf.Ticker(t).history(period="5d")["Close"]
            last, prev = float(h.iloc[-1]), float(h.iloc[-2])
            r = last / prev - 1.0
            rows[t] = {"price": round(last, 2), "ret_1d": r}
            rets.append(r)
        except Exception:  # noqa: BLE001
            rows[t] = {"price": None, "ret_1d": None}
    port = sum(rets) / len(rets) if rets else 0.0
    return rows, port


_LLM_CACHE = {}


def _get_llm(use_local_7b: bool):
    """MockLLM (instant) or the local Qwen2.5-7B-AWQ on the 2060 (cached)."""
    if not use_local_7b:
        from trr.llm import MockLLM
        return MockLLM(), False
    if "hf" not in _LLM_CACHE:
        import os
        from trr.llm import HFReasoningLLM
        model = os.environ.get("SMALL_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
        _LLM_CACHE["hf"] = HFReasoningLLM(model_path=model, dtype="float16",
                                         device="cuda", max_input_tokens=1536,
                                         batch_size=4)
    return _LLM_CACHE["hf"], True


_RAG_BANK = {}


def _get_rag_bank():
    """Labeled historical analogue bank (FNSPID news days + realized crash labels)
    for LIVE RAG — retrieve 'today looks like past day X, which crashed'."""
    if "bank" not in _RAG_BANK:
        try:
            from trr.news import group_by_day, load_news
            from trr.prices import crash_labels_daily
            from trr.rag import CausalRAG, day_text
            bd = group_by_day(load_news("data/fnspid/stocknews.csv"))
            cl = crash_labels_daily("data/fnspid/prices", TICKERS)["crash"]
            cl.index = [d for d in cl.index]
            dates = sorted(bd)
            texts = [day_text(bd[d]) for d in dates]
            labels = [int(cl.get(d, 0)) for d in dates]
            _RAG_BANK["bank"] = (CausalRAG(k=5).fit(texts, dates), labels)
        except Exception:  # noqa: BLE001
            _RAG_BANK["bank"] = (None, [])
    return _RAG_BANK["bank"]


def run_live(headlines, use_local_7b: bool = False, use_rag: bool = False):
    """Run one TRR step over the live headlines -> crash_prob, edges, rationale.

    use_local_7b loads the local Qwen2.5-7B-AWQ on the 2060 (real LLM, ~1-3 min);
    default MockLLM is instant. use_rag retrieves analogues from the LABELED
    historical bank and injects them into the reasoning context.
    """
    from trr.attention import pagerank_prune
    from trr.reason import reason_crash
    llm, real = _get_llm(use_local_7b)
    cap = headlines[:8] if real else headlines  # cap items for the slow 7B path
    btok = 640 if real else 768
    edges = llm.brainstorm_multi([cap], TICKERS, max_new_tokens=btok)[0] if cap else []
    pruned = pagerank_prune(edges, TICKERS, top_k=30)
    ctx = ""
    if use_rag:
        bank, labels = _get_rag_bank()
        if bank is not None:
            ctx = bank.fewshot_for_query(" ".join(h.title for h in headlines), labels)
    prob, rationale = (reason_crash(pruned, llm, context=ctx, universe=TICKERS)
                       if pruned else (0.0, "no impacts extracted from live news"))
    return {
        "crash_prob": float(prob),
        "rationale": rationale,
        "n_news": len(headlines),
        "n_edges": len(pruned),
        "edges": [{"subject": e.subject, "object": e.object,
                   "polarity": e.polarity, "weight": round(e.weight, 2)}
                  for e in pruned[:20]],
        "asof": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": ("Qwen2.5-7B-AWQ (local 2060)" if real else "MockLLM (heuristic)")
                   + (" +RAG" if (use_rag and ctx) else ""),
        "rag_analogues": bool(ctx),
    }


def run_live_window(items, use_local_7b: bool = False):
    """Feed the WHOLE multi-day window through the full TRR pipeline (brainstorm
    per day -> decaying memory -> reason), the way TRR is meant to consume
    history — not as one flat prompt. `items` is a list of NewsItem spanning days.
    Returns the latest day's signal + how many days were processed.
    """
    from trr.news import group_by_day
    from trr.pipeline import TRRPipeline
    llm, real = _get_llm(use_local_7b)
    by_day = group_by_day(items)
    pipe = TRRPipeline(llm=llm, portfolio=TICKERS, batch=True, cross_batch=True,
                       max_items_per_day=12 if real else 40,
                       reason_max_new_tokens=160 if real else 256,
                       brainstorm_max_new_tokens=400 if real else 768)
    pred = pipe.run(by_day)
    if len(pred) == 0:
        return {"crash_prob": 0.0, "rationale": "no news", "n_news": 0,
                "n_edges": 0, "days": 0, "edges": [],
                "asof": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "backend": "Qwen2.5-7B-AWQ (local 2060)" if real else "MockLLM (heuristic)"}
    last = pred.iloc[-1]
    return {
        "crash_prob": float(last["crash_prob"]),
        "rationale": str(last["rationale"]),
        "n_news": int(pred["n_news"].sum()),
        "n_edges": int(last["n_edges"]),
        "days": int(len(pred)),
        "edges": [],
        "asof": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": "Qwen2.5-7B-AWQ (local 2060)" if real else "MockLLM (heuristic)",
    }


def compose_advisory(sig: dict) -> dict:
    """Turn a raw TRR signal into a structured DAILY ADVISORY: risk level, the
    most-exposed assets, the key driving events, and cautions — actionable
    analysis, not just a probability. (Research output, not financial advice.)
    """
    prob = sig.get("crash_prob", 0.0)
    level = "HIGH" if prob >= 0.6 else "ELEVATED" if prob >= 0.3 else "LOW"
    edges = sig.get("edges", [])
    # per-asset exposure: summed negative-edge weight pointing at each ticker
    risk = {t: 0.0 for t in TICKERS}
    for e in edges:
        if e["object"] in risk and e["polarity"] < 0:
            risk[e["object"]] += e["weight"]
    at_risk = sorted([(t, w) for t, w in risk.items() if w > 0],
                     key=lambda x: -x[1])[:3]
    # top driving events = strongest negative edges
    drivers = sorted([e for e in edges if e["polarity"] < 0],
                     key=lambda e: -e["weight"])[:3]
    cautions = []
    if level != "LOW":
        if at_risk:
            cautions.append("Elevated downside concentrated in "
                            + ", ".join(f"{t}" for t, _ in at_risk)
                            + " — monitor exposure there first.")
        if drivers:
            d = drivers[0]
            cautions.append(f"Primary driver: {d['subject']} → {d['object']} "
                            "(negative). Watch for follow-through / contagion.")
        cautions.append("Consider risk-reduction sizing if the signal persists "
                        "over consecutive days (3-day horizon).")
    else:
        cautions.append("No systemic stress detected; routine monitoring.")
    return {
        "risk_level": level,
        "crash_prob": prob,
        "horizon": "next ~3 trading days",
        "at_risk_assets": [{"ticker": t, "exposure": round(w, 2)} for t, w in at_risk],
        "top_drivers": drivers,
        "cautions": cautions,
        "rationale": sig.get("rationale", ""),
        "backend": sig.get("backend", "?"),
        "asof": sig.get("asof"),
        "disclaimer": "Research/analysis output — NOT financial advice.",
    }


def daily_report(use_local_7b: bool = False, use_rag: bool = True) -> dict:
    """Build today's advisory report from current news and save it for the web."""
    import json
    import os
    heads = fetch_live_headlines()
    sig = run_live(heads, use_local_7b=use_local_7b, use_rag=use_rag)
    prices, port_move = fetch_live_prices()
    adv = compose_advisory(sig)
    adv["portfolio_move_1d"] = port_move
    adv["prices"] = prices
    adv["n_headlines"] = len(heads)
    os.makedirs("data/live", exist_ok=True)
    with open("data/live/daily_report.json", "w") as f:
        json.dump(adv, f, indent=2)
    return adv


def read_daemon_snapshot(max_age_s: int = 300):
    """If scripts.live_daemon is running, return its latest signal+prices snapshot
    (so the heavy 7B can run continuously in the daemon and the UI just displays
    it). Returns None if no fresh daemon output exists."""
    import json
    import os
    sp, pp = "data/live/signal.json", "data/live/prices.json"
    if not os.path.exists(sp):
        return None
    try:
        sig = json.load(open(sp))
        t = datetime.fromisoformat(sig["asof"])
        if (datetime.now(timezone.utc) - t).total_seconds() > max_age_s:
            return None
        pr = json.load(open(pp)) if os.path.exists(pp) else {"prices": {}, "portfolio_move": 0.0}
        return {"signal": sig, "prices": pr.get("prices", {}),
                "portfolio_move": pr.get("portfolio_move", 0.0),
                "headlines": [], "source": "daemon"}
    except Exception:  # noqa: BLE001
        return None


def live_snapshot(use_local_7b: bool = False):
    """One call -> everything the live monitor needs."""
    heads = fetch_live_headlines()
    prices, port_move = fetch_live_prices()
    sig = run_live(heads, use_local_7b=use_local_7b)
    return {"signal": sig, "prices": prices, "portfolio_move": port_move,
            "headlines": [{"ticker": h.assets[0], "title": h.title} for h in heads]}


if __name__ == "__main__":
    snap = live_snapshot()
    s = snap["signal"]
    print(f"[live] asof {s['asof']}  crash_prob={s['crash_prob']:.2f}  "
          f"news={s['n_news']} edges={s['n_edges']}  port_move={snap['portfolio_move']:+.2%}")
    print(f"[live] rationale: {s['rationale'][:120]}")
    print(f"[live] {len(snap['headlines'])} live headlines, e.g.:")
    for h in snap["headlines"][:4]:
        print(f"    [{h['ticker']}] {h['title'][:70]}")
