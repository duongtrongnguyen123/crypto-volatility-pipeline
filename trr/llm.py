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
    "crash", "plunge", "collapse", "hack", "exploit", "bankruptcy", "default",
    "lawsuit", "ban", "selloff", "liquidation", "fraud", "fear", "dump",
    "halt", "insolvent", "contagion", "delist", "sec", "fud", "rug", "depeg",
}
_POS = {
    "surge", "rally", "approval", "etf", "adoption", "partnership", "upgrade",
    "bullish", "record", "inflow", "halving", "breakout", "gain", "soar",
}


def extract_json(text: str):
    """Best-effort extraction of the first JSON object/array in `text`."""
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
    def predict_crash(self, tuples: list[tuple], context: str = "") -> tuple[float, str]:
        prompt = self._reason_prompt(tuples, context)
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
            "You are a financial analyst building an impact graph for crypto "
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
                             candidate_assets: list[str]) -> str:
        assets = ", ".join(candidate_assets)
        day = news_items[0].timestamp if news_items else None
        day_str = f"{day:%Y-%m-%d}" if day is not None else ""
        headlines = "\n".join(
            f"  [{i}] {item.text()}" for i, item in enumerate(news_items)
        )
        return (
            "You are a financial analyst building an impact graph for crypto "
            "portfolio crash detection. Below are the crypto news headlines for "
            f"a single day ({day_str}), each prefixed with a numeric index in "
            "square brackets.\n"
            f"Portfolio assets: {assets}\n\n"
            f"Headlines:\n{headlines}\n\n"
            "For EVERY headline that implies an impact on a portfolio asset, "
            "emit one or more directed impact relations toward that asset. "
            "Return ONLY a single JSON array of objects, each with keys: "
            "news_idx (the bracketed index of the source headline), subject, "
            "object, polarity (1 positive / -1 negative), weight (0..1), "
            "rationale. Use portfolio tickers for affected assets. Omit "
            "headlines with no portfolio impact.\n"
        )

    @staticmethod
    def _reason_prompt(tuples: list[tuple], context: str) -> str:
        lines = "\n".join(
            f"  ({t[0]:%Y-%m-%d}, {t[1]}, {'+' if t[2] >= 0 else '-'}, {t[3]})"
            for t in tuples
        )
        return (
            "You are forecasting whether the crypto portfolio (BTC, ETH, SOL, "
            "BNB, AVAX, DOGE) will CRASH (drop >8% over the next 3 days) from a "
            "graph of dated, directed news-impact relations (time, subject, "
            "polarity, object).\n"
            f"{context}\n"
            f"Impact tuples:\n{lines}\n\n"
            "CALIBRATION — read carefully:\n"
            "- Crypto news is negative on MOST days; routine bearish headlines do "
            "NOT mean a crash. The base rate of actual crashes is only ~13% of "
            "days, so the DEFAULT/typical answer is a LOW probability (~0.10-0.20).\n"
            "- Assign a HIGH probability (>0.6) ONLY when the impacts show "
            "SYSTEMIC, escalating, contagion-style stress concentrated in time — "
            "e.g. a major exchange/stablecoin failure, cascading liquidations, or "
            "insolvency spreading across multiple portfolio assets at once.\n"
            "- Distinguish a normal stream of negative chatter (LOW) from a sharp, "
            "broad escalation beyond the usual baseline (HIGH). Spread your "
            "probabilities across the 0..1 range; do not anchor every day high.\n"
            "Return ONLY JSON: {\"crash_prob\": 0..1, \"rationale\": \"...\"}.\n"
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

    def predict_crash(self, tuples: list[tuple], context: str = "") -> tuple[float, str]:
        if not tuples:
            return 0.0, "no impacts"
        neg = sum(1 for t in tuples if t[2] < 0)
        frac_neg = neg / len(tuples)
        # Logistic-ish squashing on negative concentration + volume.
        prob = max(0.0, min(1.0, 0.15 + 0.7 * frac_neg + 0.02 * min(neg, 10)))
        return prob, f"{neg}/{len(tuples)} negative impacts toward portfolio"


class HFReasoningLLM(ReasoningLLM):
    """Local HuggingFace causal-LM backend (e.g. NVIDIA Nemotron) — zero-shot.

    transformers/torch are imported lazily so this module stays importable on a
    box without them. Intended to run on the Kaggle RTX 6000 Pro (no internet:
    the model is pre-staged as a Kaggle model/dataset and loaded from disk).
    """

    def __init__(self, model_path: str, dtype: str = "bfloat16",
                 device: str = "cuda", max_input_tokens: int = 4096) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.max_input_tokens = max_input_tokens
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
