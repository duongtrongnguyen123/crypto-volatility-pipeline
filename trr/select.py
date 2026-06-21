"""Salience-based per-day news selection — keeps LLM cost bounded as volume grows.

Adding news volume (more tickers, full corpora) must NOT increase LLM cost: the
pipeline reasons over at most `k` items/day. Naive `items[:k]` wastes that budget
on near-duplicates and trivia. This selects the `k` most INFORMATIVE headlines:

  1. exact-dedup by normalized title,
  2. score by salience = sentiment extremity (crash/▲ lexicon) + portfolio-ticker
     mention + length,
  3. take the top candidates, then greedily drop near-duplicates (token Jaccard).

O(n) scoring + O(k²) dedup on the shortlist — fast even at thousands/day, so the
ingest layer can scale to GBs while the LLM input stays at ~40/day.
"""
from __future__ import annotations

import re

_NEG = set("crash slump plunge plunges fall falls drop drops fear hack ban lawsuit "
           "selloff sell-off tumble sink slide warn cut loss recession halt panic "
           "downgrade bankruptcy default contagion liquidation crackdown probe".split())
_POS = set("surge soar rally gain gains jump rise beat upgrade record boom approve "
           "win growth profit strong bullish breakout".split())


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def _tokens(t: str) -> set:
    return set(_norm(t).split())


def _salience(item, portfolio: set) -> float:
    toks = _tokens(item.title)
    senti = sum(1 for w in toks if w in _NEG) * 1.5 + sum(1 for w in toks if w in _POS)
    has_asset = 1.0 if (set(getattr(item, "assets", []) or []) & portfolio
                        or toks & {a.lower() for a in portfolio}) else 0.0
    return senti + has_asset + min(len(toks), 25) / 25.0


def select_salient(items, k: int, portfolio=None, jaccard: float = 0.7):
    """Return the k most salient, de-duplicated items for one day."""
    if len(items) <= k:
        return items
    pset = {a.upper() for a in (portfolio or [])}
    seen, uniq = set(), []
    for it in items:                       # 1. exact-dedup
        key = _norm(it.title)
        if key and key not in seen:
            seen.add(key); uniq.append(it)
    uniq.sort(key=lambda it: _salience(it, pset), reverse=True)   # 2. rank
    out, out_tok = [], []                  # 3. greedy near-dup drop on the shortlist
    for it in uniq:
        tk = _tokens(it.title)
        if any(len(tk & o) / max(1, len(tk | o)) >= jaccard for o in out_tok):
            continue
        out.append(it); out_tok.append(tk)
        if len(out) >= k:
            break
    return out


def select_relevant(items, query: str, k: int, portfolio=None):
    """RAG-style selection: from a LARGE daily pool, retrieve the k headlines most
    RELEVANT to a crash/portfolio query (TF-IDF cosine), then drop near-dups.

    This lets the underlying news corpus scale (thousands/day, many tickers) while
    the LLM reads only the bounded, most-relevant slice — the LLM effectively
    'draws from' the big pool via retrieval. Falls back to salience if TF-IDF is
    unavailable or the day is tiny.
    """
    if len(items) <= k:
        return items
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception:  # noqa: BLE001
        return select_salient(items, k, portfolio)
    docs = [_norm(it.title) or "__empty__" for it in items]
    vec = TfidfVectorizer(max_features=4096, stop_words="english", ngram_range=(1, 2))
    m = vec.fit_transform(docs).astype(np.float32)
    q = vec.transform([_norm(query)]).astype(np.float32)
    sims = np.asarray((m @ q.T).todense()).ravel()
    order = np.argsort(-sims)
    out, out_tok = [], []
    for j in order:
        it = items[j]; tk = _tokens(it.title)
        if any(len(tk & o) / max(1, len(tk | o)) >= 0.7 for o in out_tok):
            continue
        out.append(it); out_tok.append(tk)
        if len(out) >= k:
            break
    return out


# default crash-relevance query for the portfolio (used by select_mode="rag")
def crash_query(portfolio) -> str:
    return (" ".join(portfolio) + " crash selloff plunge tumble fall fear hack ban "
            "lawsuit fed rates inflation recession contagion liquidation downgrade "
            "sanctions war geopolitical bankruptcy default panic")


if __name__ == "__main__":
    import time
    from datetime import datetime

    from trr.schema import NewsItem
    base = ["Fed hikes rates, markets tumble", "AAPL upgraded to buy",
            "Exchange hacked, contagion fears", "Quiet trading session",
            "NVDA earnings beat estimates", "Recession warning from analysts"]
    for n in (40, 400, 4000):
        items = [NewsItem(id=str(i), timestamp=datetime(2026, 1, 1),
                          title=base[i % len(base)] + (f" #{i}" if i % 7 else ""),
                          assets=["AAPL"]) for i in range(n)]
        t0 = time.time()
        sel = select_salient(items, 40, ["AAPL", "NVDA", "TSLA"])
        print(f"  {n:5} headlines -> {len(sel)} selected in {1000*(time.time()-t0):.1f} ms")
    assert len(select_salient(items, 40, ["AAPL"])) == 40
    print("[select] bounded output as volume grows — LLM cost stays O(days*k)")
    # RAG-relevance selection from a large pool
    rel = select_relevant(items, crash_query(["AAPL", "NVDA"]), 40, ["AAPL", "NVDA"])
    print(f"[select] RAG-relevance: {len(items)} -> {len(rel)} most crash-relevant")
