"""LLM backend abstraction for the TRR pipeline.

The four TRR phases never call a model directly — they call the two semantic
methods on a `ReasoningLLM`:

    extract_impacts(news, candidate_assets) -> list[ImpactEdge]   # Brainstorming
    predict_crash(tuples, context)          -> (crash_prob, rationale)  # Reasoning

This keeps the phases identical whether the backend is:
    - MockLLM       : deterministic heuristic, for fast offline pipeline tests
    - HFReasoningLLM: a local HuggingFace causal LM (e.g. NVIDIA Nemotron) run
                      zero-shot on the Kaggle RTX 6000 Pro (no internet).

Both implement the same interface, so swapping backends changes nothing else.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from datetime import datetime

from trr.schema import PORTFOLIO, ImpactEdge, NewsItem

# Lightweight lexicons for the deterministic mock backend.
_NEG = {
    # base + common variants (matching is exact-word, so include plural/past forms)
    "crash", "crashes", "crashed", "plunge", "plunges", "plunged", "plunging",
    "collapse", "collapses", "collapsed", "hack", "hacks", "hacked", "exploit",
    "exploited", "bankruptcy", "bankrupt", "default", "defaults", "lawsuit",
    "sue", "sues", "sued", "ban", "bans", "banned", "selloff", "sell-off",
    "liquidation", "liquidations", "liquidated", "fraud", "fear", "fears",
    "dump", "dumps", "dumped", "halt", "halts", "halted", "insolvent",
    "insolvency", "contagion", "delist", "delisted", "sec", "fud", "rug",
    "depeg", "panic", "tumble", "tumbles", "tumbled", "slump", "slumps",
    "sink", "sinks", "sinking", "slide", "slides", "rout", "freeze", "frozen",
    "downgrade", "downgraded", "recession", "warn", "warns", "warning", "loss",
    "losses", "cut", "cuts", "weak", "drop", "drops", "fall", "falls", "probe",
    "bearish", "sanction", "sanctions", "war", "crisis", "meltdown",
}
_POS = {
    "surge", "surges", "surged", "rally", "rallies", "rallied", "approval",
    "approve", "approved", "etf", "adoption", "partnership", "upgrade",
    "upgraded", "bullish", "record", "records", "inflow", "inflows", "halving",
    "breakout", "gain", "gains", "gained", "soar", "soars", "soared", "jump",
    "jumps", "jumped", "rise", "rises", "beat", "beats", "profit", "profits",
    "strong", "growth", "boom", "win", "wins", "rebound", "recover", "recovers",
}


def extract_json(text: str):
    """Best-effort extraction of the first JSON object/array in `text`."""
    # Reasoning models (R1 / QwQ) emit a <think>...</think> trace before the
    # answer — parse only what comes after it.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    # Try fenced block first.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text
    # Parse the balanced span of whichever bracket appears FIRST — so a JSON
    # array "[{...},{...}]" is read as the array, not just its first object.
    pos_obj = candidate.find("{")
    pos_arr = candidate.find("[")
    spans = []
    if pos_arr != -1:
        spans.append((pos_arr, "[", "]"))
    if pos_obj != -1:
        spans.append((pos_obj, "{", "}"))
    spans.sort()
    for start, opener, closer in spans:
        depth = 0
        for i in range(start, len(candidate)):
            if candidate[i] == opener:
                depth += 1
            elif candidate[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(candidate[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


class ReasoningLLM(ABC):
    """Backend interface. Subclasses implement `generate`; the two semantic
    methods have prompt-based default implementations that call it.
    """

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.0) -> str:
        ...

    # --- Phase 1: Brainstorming -------------------------------------------
    def extract_impacts(self, news: NewsItem,
                        candidate_assets: list[str]) -> list[ImpactEdge]:
        prompt = self._impact_prompt(news, candidate_assets)
        raw = self.generate(prompt, max_new_tokens=512)
        data = extract_json(raw) or []
        edges: list[ImpactEdge] = []
        for d in data if isinstance(data, list) else []:
            try:
                edges.append(ImpactEdge(
                    subject=str(d["subject"]).upper(),
                    object=str(d["object"]).upper(),
                    polarity=1 if int(d.get("polarity", -1)) >= 0 else -1,
                    weight=max(0.0, min(1.0, float(d.get("weight", 0.5)))),
                    timestamp=news.timestamp,
                    source_news_id=news.id,
                    rationale=str(d.get("rationale", "")),
                ))
            except (KeyError, ValueError, TypeError):
                continue
        return edges

    # --- Phase 1 (batched): daily brainstorming ---------------------------
    def extract_impacts_batch(
        self,
        news_items: list[NewsItem],
        candidate_assets: list[str],
        max_items: int = 40,
    ) -> list[ImpactEdge]:
        """Extract impact edges for a whole day's news in ONE generate() call.

        On a large real corpus, one LLM call per article (`extract_impacts`) is
        prohibitively expensive — 31k articles would mean 31k generations. This
        aggregates a day's (up to `max_items`) headlines into a single numbered
        prompt and asks for one JSON array of impact edges; each edge carries a
        `news_idx` that maps it back to its NewsItem for id/timestamp. Edges with
        an out-of-range index are skipped.

        Callers should pass the most relevant items first; this simply caps the
        list at `max_items` (head/cap) so the prompt stays within budget.
        """
        items = news_items[:max_items]
        if not items:
            return []

        prompt = self._impact_batch_prompt(items, candidate_assets)
        raw = self.generate(prompt, max_new_tokens=1024)
        data = extract_json(raw) or []

        edges: list[ImpactEdge] = []
        for d in data if isinstance(data, list) else []:
            try:
                idx = int(d["news_idx"])
            except (KeyError, ValueError, TypeError):
                continue
            if idx < 0 or idx >= len(items):
                continue  # clamp/skip bad indices
            news = items[idx]
            try:
                edges.append(ImpactEdge(
                    subject=str(d["subject"]).upper(),
                    object=str(d["object"]).upper(),
                    polarity=1 if int(d.get("polarity", -1)) >= 0 else -1,
                    weight=max(0.0, min(1.0, float(d.get("weight", 0.5)))),
                    timestamp=news.timestamp,
                    source_news_id=news.id,
                    rationale=str(d.get("rationale", "")),
                ))
            except (KeyError, ValueError, TypeError):
                continue
        return edges

    # --- Phase 4: Reasoning -----------------------------------------------
    def predict_crash(self, tuples: list[tuple], context: str = "",
                      universe: list[str] | None = None) -> tuple[float, str]:
        prompt = self._reason_prompt(tuples, context, universe)
        raw = self.generate(prompt, max_new_tokens=512)
        data = extract_json(raw) or {}
        try:
            prob = float(data.get("crash_prob", 0.0))
        except (TypeError, ValueError):
            prob = 0.0
        prob = max(0.0, min(1.0, prob))
        return prob, str(data.get("rationale", ""))

    # --- prompt builders (shared by any generate()-based backend) ----------
    @staticmethod
    def _impact_prompt(news: NewsItem, candidate_assets: list[str]) -> str:
        assets = ", ".join(candidate_assets)
        return (
            "You are a financial analyst building an impact graph for "
            "portfolio crash detection. Given a news item, list directed impact "
            "relations from the news/entities toward the portfolio assets.\n"
            f"Portfolio assets: {assets}\n"
            f"News ({news.timestamp:%Y-%m-%d}): {news.text()}\n\n"
            "Return ONLY a JSON array of objects with keys: subject, object, "
            "polarity (1 positive / -1 negative), weight (0..1), rationale. "
            "Use portfolio tickers for affected assets.\n"
        )

    @staticmethod
    def _impact_batch_prompt(news_items: list[NewsItem],
                             candidate_assets: list[str],
                             elicit_chains: bool = False) -> str:
        assets = ", ".join(candidate_assets)
        day = news_items[0].timestamp if news_items else None
        day_str = f"{day:%Y-%m-%d}" if day is not None else ""
        headlines = "\n".join(
            f"  [{i}] {item.text()}" for i, item in enumerate(news_items)
        )
        chain_hint = (
            "When an event acts THROUGH an intermediary (a sector, commodity, "
            "company or another asset) before reaching a portfolio asset, emit "
            "BOTH hops as separate edges (event->intermediary AND "
            "intermediary->asset) so multi-hop contagion chains form, e.g. "
            "OIL->AIRLINES and AIRLINES->AAPL. "
            if elicit_chains else ""
        )
        return (
            "You are a financial analyst building an impact graph for "
            "portfolio crash detection. Below are the news headlines for "
            f"a single day ({day_str}), each prefixed with a numeric index in "
            "square brackets.\n"
            f"Portfolio assets: {assets}\n\n"
            f"Headlines:\n{headlines}\n\n"
            "For EVERY headline that implies an impact on a portfolio asset, "
            "emit one or more directed impact relations toward that asset. "
            + chain_hint +
            "Return ONLY a single JSON array of objects, each with keys: "
            "news_idx (the bracketed index of the source headline), subject, "
            "object, polarity (1 positive / -1 negative), weight (0..1), "
            "rationale. Use portfolio tickers for affected assets. Omit "
            "headlines with no portfolio impact.\n"
        )

    @staticmethod
    def _reason_prompt(tuples: list[tuple], context: str,
                       universe: list[str] | None = None) -> str:
        lines = "\n".join(
            f"  ({t[0]:%Y-%m-%d}, {t[1]}, {'+' if t[2] >= 0 else '-'}, {t[3]})"
            for t in tuples
        )
        uni = ", ".join(universe) if universe else "BTC, ETH, SOL, BNB, AVAX, DOGE"
        return (
            f"You are forecasting whether the portfolio ({uni}) will CRASH (a "
            "large draw-down over the next 3 days) from a "
            "graph of dated, directed news-impact relations (time, subject, "
            "polarity, object).\n"
            f"{context}\n"
            f"Impact tuples:\n{lines}\n\n"
            "CALIBRATION — read carefully:\n"
            "- Financial news is negative on MANY days; routine bearish headlines do "
            "NOT mean a crash. The base rate of actual crashes is low (~5-13% of "
            "days), so the DEFAULT/typical answer is a LOW probability (~0.10-0.20).\n"
            "- Assign a HIGH probability (>0.6) ONLY when the impacts show "
            "SYSTEMIC, escalating, contagion-style stress concentrated in time — "
            "e.g. a macro shock, cascading sell-offs, panic, or distress "
            "spreading across multiple portfolio assets at once.\n"
            "- Distinguish a normal stream of negative chatter (LOW) from a sharp, "
            "broad escalation beyond the usual baseline (HIGH). Spread your "
            "probabilities across the 0..1 range; do not anchor every day to one "
            "value.\n\n"
            "FEW-SHOT EXAMPLES (learn the pattern; do not copy the numbers):\n"
            "Example A — routine bearish chatter, NO crash:\n"
            "  a few mild, scattered negative items (price dip commentary, one "
            "regulator quote), no single shock hitting multiple assets at once\n"
            "  -> {\"crash_prob\": 0.10, \"rationale\": \"ordinary negative noise; "
            "no systemic or contagion signal\"}\n"
            "Example B — broad escalation, ELEVATED:\n"
            "  several negative impacts clustering the SAME day across multiple assets "
            "(a sector shock, a regulatory crackdown) but contained to a sub-sector\n"
            "  -> {\"crash_prob\": 0.45, \"rationale\": \"notable stress but not yet "
            "portfolio-wide contagion\"}\n"
            "Example C — systemic contagion, CRASH imminent:\n"
            "  dense same-day negatives — a macro/systemic shock "
            "cascading across the whole portfolio at once with panic-selling language\n"
            "  -> {\"crash_prob\": 0.88, \"rationale\": \"simultaneous portfolio-wide "
            "failure; classic contagion cascade\"}\n\n"
            "Now assess TODAY from the tuples above.\n"
            "Return ONLY JSON: {\"crash_prob\": 0..1, \"rationale\": \"...\"}.\n"
        )

    # --- batched execution across many days (the speed path) --------------
    def generate_batch(self, prompts: list[str], max_new_tokens: int = 512,
                       temperature: float = 0.0) -> list[str]:
        """Generate for many prompts. Default: sequential. Subclasses with a
        real model override this to batch the forward pass."""
        return [self.generate(p, max_new_tokens, temperature) for p in prompts]

    @staticmethod
    def _parse_batch_edges(raw: str, items: list[NewsItem]) -> list[ImpactEdge]:
        data = extract_json(raw) or []
        edges: list[ImpactEdge] = []
        for d in data if isinstance(data, list) else []:
            try:
                idx = int(d["news_idx"])
            except (KeyError, ValueError, TypeError):
                continue
            if idx < 0 or idx >= len(items):
                continue
            news = items[idx]
            try:
                edges.append(ImpactEdge(
                    subject=str(d["subject"]).upper(),
                    object=str(d["object"]).upper(),
                    polarity=1 if int(d.get("polarity", -1)) >= 0 else -1,
                    weight=max(0.0, min(1.0, float(d.get("weight", 0.5)))),
                    timestamp=news.timestamp,
                    source_news_id=news.id,
                    rationale=str(d.get("rationale", "")),
                ))
            except (KeyError, ValueError, TypeError):
                continue
        return edges

    def brainstorm_multi(self, day_items_list: list[list[NewsItem]],
                        candidate_assets: list[str], max_items: int = 40,
                        max_new_tokens: int = 768,
                        elicit_chains: bool = False) -> list[list[ImpactEdge]]:
        """Batched Brainstorming: one prompt per day, all generated together."""
        capped = [items[:max_items] for items in day_items_list]
        prompts = [self._impact_batch_prompt(items, candidate_assets, elicit_chains)
                   for items in capped]
        raws = self.generate_batch(prompts, max_new_tokens=max_new_tokens)
        return [self._parse_batch_edges(raw, items) for items, raw in zip(capped, raws)]

    def reason_multi(self, tuples_list: list[list[tuple]], contexts: list[str],
                    max_new_tokens: int = 256, n_samples: int = 1,
                    temperature: float = 0.0,
                    universe: list[str] | None = None) -> list[tuple[float, str]]:
        """Batched Reasoning. With n_samples>1 and temperature>0 this does
        SELF-CONSISTENCY: sample several reasoning traces per day and average the
        crash probabilities (test-time compute scaling)."""
        prompts = [self._reason_prompt(t, c, universe) for t, c in zip(tuples_list, contexts)]

        def parse(raw):
            data = extract_json(raw) or {}
            try:
                return max(0.0, min(1.0, float(data.get("crash_prob", 0.0)))), str(data.get("rationale", ""))
            except (TypeError, ValueError):
                return 0.0, ""

        if n_samples <= 1:
            return [parse(r) for r in self.generate_batch(prompts, max_new_tokens=max_new_tokens)]

        # Self-consistency: K sampled passes, average the probabilities.
        sample_probs = [[] for _ in prompts]
        last_rat = ["" for _ in prompts]
        for _ in range(n_samples):
            raws = self.generate_batch(prompts, max_new_tokens=max_new_tokens,
                                       temperature=temperature)
            for i, r in enumerate(raws):
                p, rat = parse(r)
                sample_probs[i].append(p)
                if rat:
                    last_rat[i] = rat
        return [(sum(ps) / len(ps) if ps else 0.0, last_rat[i])
                for i, ps in enumerate(sample_probs)]

    # --- Phase 4 (alternative target): price-direction reasoning ----------
    def reason_multi_direction(
        self,
        tuples_list: list[list[tuple]],
        contexts: list[str],
        max_new_tokens: int = 256,
        n_samples: int = 1,
        temperature: float = 0.0,
        universe: list[str] | None = None,
    ) -> list[tuple[float, str]]:
        """Batched price-direction Reasoning.

        Mirrors `reason_multi` but asks the LLM for `up_prob` — the probability
        the portfolio price RISES over the next day — instead of a crash
        probability. Supports the same self-consistency sampling (n_samples>1 at
        temperature>0 averages the up-probabilities). Returns one
        (up_prob, rationale) per day.
        """
        prompts = [self._reason_prompt_direction(t, c, universe)
                   for t, c in zip(tuples_list, contexts)]

        def parse(raw):
            data = extract_json(raw) or {}
            try:
                return max(0.0, min(1.0, float(data.get("up_prob", 0.5)))), str(data.get("rationale", ""))
            except (TypeError, ValueError):
                return 0.5, ""

        if n_samples <= 1:
            return [parse(r) for r in self.generate_batch(prompts, max_new_tokens=max_new_tokens)]

        sample_probs = [[] for _ in prompts]
        last_rat = ["" for _ in prompts]
        for _ in range(n_samples):
            raws = self.generate_batch(prompts, max_new_tokens=max_new_tokens,
                                       temperature=temperature)
            for i, r in enumerate(raws):
                p, rat = parse(r)
                sample_probs[i].append(p)
                if rat:
                    last_rat[i] = rat
        return [(sum(ps) / len(ps) if ps else 0.5, last_rat[i])
                for i, ps in enumerate(sample_probs)]

    @staticmethod
    def _reason_prompt_direction(tuples: list[tuple], context: str,
                                 universe: list[str] | None = None) -> str:
        lines = "\n".join(
            f"  ({t[0]:%Y-%m-%d}, {t[1]}, {'+' if t[2] >= 0 else '-'}, {t[3]})"
            for t in tuples
        )
        uni = ", ".join(universe) if universe else "BTC, ETH, SOL, BNB, AVAX, DOGE"
        return (
            "You are forecasting the next-day PRICE DIRECTION of an equal-weight "
            f"portfolio ({uni}) from a graph of "
            "dated, directed news-impact relations (time, subject, polarity, "
            "object).\n"
            f"{context}\n"
            f"Impact tuples:\n{lines}\n\n"
            "CALIBRATION — read carefully:\n"
            "- Estimate up_prob = P(portfolio closes HIGHER tomorrow than today).\n"
            "- Daily price direction is near a coin-flip, so the DEFAULT is "
            "~0.50. Move away from 0.50 only when the impacts lean clearly one "
            "way: a preponderance of POSITIVE impacts (strong earnings, upgrades, "
            "inflows) pushes up_prob ABOVE 0.50; a preponderance of NEGATIVE "
            "impacts (downgrades, shocks, sell-offs) pushes it BELOW 0.50.\n"
            "- Weight by breadth and strength: many strong same-day impacts hitting "
            "multiple assets justify a larger move from 0.50 than one mild item. "
            "Spread your probabilities; do not anchor every day to 0.50.\n\n"
            "FEW-SHOT EXAMPLES (learn the pattern; do not copy the numbers):\n"
            "Example A — mild mixed chatter, no clear lean:\n"
            "  a couple of small positives and negatives roughly cancelling\n"
            "  -> {\"up_prob\": 0.50, \"rationale\": \"balanced impacts; no edge\"}\n"
            "Example B — broad positive flow:\n"
            "  several strong positive impacts (upgrades, strong results, inflows) "
            "across multiple assets the same day, few negatives\n"
            "  -> {\"up_prob\": 0.68, \"rationale\": \"broad bullish catalysts lean up\"}\n"
            "Example C — broad negative flow:\n"
            "  dense same-day negatives (a macro shock and broad downgrades) hitting "
            "multiple assets together\n"
            "  -> {\"up_prob\": 0.30, \"rationale\": \"broad bearish shock leans down\"}\n\n"
            "Now assess TODAY from the tuples above.\n"
            "Return ONLY JSON: {\"up_prob\": 0..1, \"rationale\": \"...\"}.\n"
        )

    def reason_multi_per_asset(self, tuples_list: list[list[tuple]], contexts: list[str],
                              assets: list[str], max_new_tokens: int = 320) -> list[dict]:
        """Batched per-asset Reasoning: one crash probability PER portfolio asset."""
        prompts = [self._reason_prompt_per_asset(t, c, assets)
                   for t, c in zip(tuples_list, contexts)]
        raws = self.generate_batch(prompts, max_new_tokens=max_new_tokens)
        out: list[dict] = []
        for raw in raws:
            data = extract_json(raw) or {}
            d = {}
            for a in assets:
                v = data.get(a, data.get(a.upper(), data.get(a.lower(), 0.0)))
                try:
                    d[a] = max(0.0, min(1.0, float(v)))
                except (TypeError, ValueError):
                    d[a] = 0.0
            out.append(d)
        return out

    @staticmethod
    def _reason_prompt_per_asset(tuples: list[tuple], context: str,
                                assets: list[str]) -> str:
        lines = "\n".join(
            f"  ({t[0]:%Y-%m-%d}, {t[1]}, {'+' if t[2] >= 0 else '-'}, {t[3]})"
            for t in tuples
        )
        keys = ", ".join(f'"{a}": 0..1' for a in assets)
        return (
            "You are forecasting, for EACH crypto asset, the probability it "
            "CRASHES (drops >12% over the next 3 days), from a graph of dated, "
            "directed news-impact relations (time, subject, polarity, object).\n"
            f"{context}\n"
            f"Impact tuples:\n{lines}\n\n"
            "Reason per asset: an asset's risk rises with negative impacts "
            "directed AT it AND with broad market contagion. Most days are calm "
            "for most assets (base rate ~5-10%), so default LOW; assign high "
            "probability only to assets under specific, escalating stress. "
            "Different assets can have very different probabilities.\n"
            f"Return ONLY JSON mapping each ticker to its crash probability: "
            f"{{{keys}}}.\n"
        )


class MockLLM(ReasoningLLM):
    """Deterministic heuristic backend for offline pipeline testing.

    No model: `extract_impacts` keys off a sentiment lexicon and any portfolio
    tickers mentioned; `predict_crash` aggregates the signed, weighted impacts.
    Fully deterministic so pipeline tests are stable.
    """

    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.0) -> str:
        return "{}"  # unused; semantic methods are overridden below

    def extract_impacts(self, news: NewsItem,
                        candidate_assets: list[str]) -> list[ImpactEdge]:
        words = set(re.findall(r"[a-z]+", news.text().lower()))
        neg = len(words & _NEG)
        pos = len(words & _POS)
        polarity = -1 if neg > pos else 1
        strength = min(1.0, 0.3 + 0.2 * abs(neg - pos))

        # Affected assets: those explicitly tagged, else any ticker named in text.
        text_up = news.text().upper()
        affected = [t for t in candidate_assets if t in news.assets or t in text_up]
        if not affected:
            # Market-wide news impacts BTC/ETH as the systemic anchors.
            affected = [t for t in ("BTC", "ETH") if t in candidate_assets]

        edges: list[ImpactEdge] = []
        for asset in affected:
            edges.append(ImpactEdge(
                subject=f"NEWS:{news.id}",
                object=asset,
                polarity=polarity,
                weight=strength,
                timestamp=news.timestamp,
                source_news_id=news.id,
                rationale=f"lexicon neg={neg} pos={pos}",
            ))
        return edges

    def extract_impacts_batch(
        self,
        news_items: list[NewsItem],
        candidate_assets: list[str],
        max_items: int = 40,
    ) -> list[ImpactEdge]:
        """Deterministic batched brainstorming: loop the per-item heuristic over
        the (capped) day's items and concatenate. Keeps the mock fast and stable
        while exercising the same batched code path as the real backend.
        """
        edges: list[ImpactEdge] = []
        for news in news_items[:max_items]:
            edges.extend(self.extract_impacts(news, candidate_assets))
        return edges

    def predict_crash(self, tuples: list[tuple], context: str = "",
                      universe: list[str] | None = None) -> tuple[float, str]:
        if not tuples:
            return 0.0, "no impacts"
        neg = sum(1 for t in tuples if t[2] < 0)
        frac_neg = neg / len(tuples)
        # Graded heuristic: blend negative CONCENTRATION (frac_neg) with negative
        # VOLUME (count) so severity spreads the score across a range instead of
        # snapping to one value — 1 neg edge ≈ 0.64, scaling up with more/heavier
        # negatives; non-negative news stays low (~0.12).
        prob = max(0.0, min(1.0, 0.12 + 0.45 * frac_neg + 0.07 * min(neg, 7)))
        return prob, f"{neg}/{len(tuples)} negative impacts toward portfolio"

    # Use the deterministic heuristics (not the stub generate) in batched mode.
    def brainstorm_multi(self, day_items_list, candidate_assets, max_items=40,
                        max_new_tokens=512, elicit_chains=False):
        return [self.extract_impacts_batch(items, candidate_assets, max_items)
                for items in day_items_list]

    def reason_multi(self, tuples_list, contexts, max_new_tokens=256,
                    n_samples=1, temperature=0.0, universe=None):
        return [self.predict_crash(t, c) for t, c in zip(tuples_list, contexts)]

    def predict_direction(self, tuples: list[tuple], context: str = "") -> tuple[float, str]:
        """Heuristic next-day up-probability: more POSITIVE impacts -> higher
        up_prob, more NEGATIVE -> lower. Centred at 0.5 with no impacts.
        """
        if not tuples:
            return 0.5, "no impacts"
        pos = sum(1 for t in tuples if t[2] >= 0)
        neg = len(tuples) - pos
        # Net positive fraction in [-1, 1] -> up_prob in [~0.1, ~0.9].
        net = (pos - neg) / len(tuples)
        prob = max(0.0, min(1.0, 0.5 + 0.4 * net))
        return prob, f"{pos}/{len(tuples)} positive impacts toward portfolio"

    def reason_multi_direction(self, tuples_list, contexts, max_new_tokens=256,
                               n_samples=1, temperature=0.0, universe=None):
        return [self.predict_direction(t, c) for t, c in zip(tuples_list, contexts)]

    def reason_multi_per_asset(self, tuples_list, contexts, assets, max_new_tokens=320):
        out = []
        for tuples in tuples_list:
            d = {}
            for a in assets:
                at = [t for t in tuples if t[3] == a]
                neg = sum(1 for t in at if t[2] < 0)
                d[a] = (max(0.0, min(1.0, 0.1 + 0.7 * (neg / max(len(at), 1))
                                     + 0.02 * min(neg, 10))) if at else 0.1)
            out.append(d)
        return out


class HFReasoningLLM(ReasoningLLM):
    """Local HuggingFace causal-LM backend (e.g. NVIDIA Nemotron) — zero-shot.

    transformers/torch are imported lazily so this module stays importable on a
    box without them. Intended to run on the Kaggle RTX 6000 Pro (no internet:
    the model is pre-staged as a Kaggle model/dataset and loaded from disk).
    """

    def __init__(self, model_path: str, dtype: str = "bfloat16",
                 device: str = "cuda", max_input_tokens: int = 4096,
                 batch_size: int = 24) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.max_input_tokens = max_input_tokens
        self.batch_size = batch_size  # smaller for larger models (VRAM)
        # Nemotron ships custom modeling code (model_type "nemotron_h"); it must
        # be trusted to load. The kernel runs non-interactively, so confirm here.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )
        torch_dtype = getattr(torch, dtype, torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = device

    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.0) -> str:
        torch = self._torch
        messages = [{"role": "user", "content": prompt}]
        attention_mask = None
        try:
            enc = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
                return_dict=True, truncation=True, max_length=self.max_input_tokens,
            )
            input_ids = enc["input_ids"].to(self.device)
            if "attention_mask" in enc:
                attention_mask = enc["attention_mask"].to(self.device)
        except Exception:
            enc = self.tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=self.max_input_tokens,
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            out = self.model.generate(input_ids, **gen_kwargs)
        gen = out[0][input_ids.shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def generate_batch(self, prompts: list[str], max_new_tokens: int = 512,
                       temperature: float = 0.0, batch_size: int = None) -> list[str]:
        """True batched generation with left padding, chunked to bound VRAM."""
        torch = self._torch
        tok = self.tokenizer
        batch_size = batch_size or self.batch_size
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        results: list[str] = []
        n_chunks = (len(prompts) + batch_size - 1) // batch_size
        for ci, s in enumerate(range(0, len(prompts), batch_size)):
            chunk = prompts[s : s + batch_size]
            texts = []
            for p in chunk:
                try:
                    texts.append(tok.apply_chat_template(
                        [{"role": "user", "content": p}],
                        add_generation_prompt=True, tokenize=False))
                except Exception:
                    texts.append(p)
            old_side = tok.padding_side
            tok.padding_side = "left"
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=self.max_input_tokens)
            tok.padding_side = old_side
            enc = {k: v.to(self.device) for k, v in enc.items()}
            gk = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                      pad_token_id=tok.pad_token_id or tok.eos_token_id)
            if temperature > 0:
                gk["temperature"] = temperature
            with torch.no_grad():
                out = self.model.generate(**enc, **gk)
            in_len = enc["input_ids"].shape[1]
            for i in range(out.shape[0]):
                results.append(tok.decode(out[i][in_len:], skip_special_tokens=True))
            print(f"[gen] batch {ci + 1}/{n_chunks} ({len(chunk)} prompts) done", flush=True)
        return results
