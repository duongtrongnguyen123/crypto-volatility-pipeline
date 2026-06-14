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
    # Find the first balanced { } or [ ] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        if start == -1:
            continue
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
    def _reason_prompt(tuples: list[tuple], context: str) -> str:
        lines = "\n".join(
            f"  ({t[0]:%Y-%m-%d}, {t[1]}, {'+' if t[2] >= 0 else '-'}, {t[3]})"
            for t in tuples
        )
        return (
            "You are detecting an imminent crypto portfolio crash from a graph "
            "of dated, directed impact relations (time, subject, polarity, "
            "object).\n"
            f"{context}\n"
            f"Impact tuples:\n{lines}\n\n"
            "Reason over the temporal accumulation and relational spread of "
            "negative impacts toward the portfolio. Return ONLY JSON: "
            '{"crash_prob": 0..1, "rationale": "..."}.\n'
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        torch_dtype = getattr(torch, dtype, torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device,
        )
        self.model.eval()
        self.device = device

    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.0) -> str:
        torch = self._torch
        messages = [{"role": "user", "content": prompt}]
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
                truncation=True, max_length=self.max_input_tokens,
            ).to(self.device)
        except Exception:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=self.max_input_tokens,
            ).input_ids.to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        return text
