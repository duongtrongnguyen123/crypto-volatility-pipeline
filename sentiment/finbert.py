"""LLM sentiment module — FinBERT (ProsusAI/finbert).

Exposes `score_sentiment(text) -> float` in [-1, 1].

The model is downloaded from HuggingFace on first use and cached locally; every
subsequent call runs fully offline. The model is loaded lazily and cached at
module scope so it is materialized at most once per process (e.g. once per Spark
executor).

Mapping from FinBERT's {positive, negative, neutral} probabilities to a scalar:
    score = P(positive) - P(negative)
which naturally lands in [-1, 1] (neutral mass pulls the score toward 0).
"""
from __future__ import annotations

import threading
from typing import Optional

_MODEL_NAME = "ProsusAI/finbert"

_lock = threading.Lock()
_tokenizer = None
_model = None
_torch = None


def _ensure_loaded() -> None:
    """Load tokenizer + model once, thread-safely."""
    global _tokenizer, _model, _torch
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _torch = torch
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        _model.eval()
        # Use GPU (RTX 2060 Super) when available.
        if torch.cuda.is_available():
            _model.to("cuda")


def score_sentiment(text: Optional[str]) -> float:
    """Return a sentiment score in [-1, 1] for `text`.

    Empty/None input returns 0.0 (neutral).
    """
    if not text or not str(text).strip():
        return 0.0

    _ensure_loaded()
    torch = _torch

    inputs = _tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    )
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()

    # ProsusAI/finbert label order: id2label = {0: positive, 1: negative, 2: neutral}
    id2label = {int(k): v.lower() for k, v in _model.config.id2label.items()}
    p = {id2label[i]: float(probs[i]) for i in range(len(probs))}
    score = p.get("positive", 0.0) - p.get("negative", 0.0)
    return max(-1.0, min(1.0, score))


if __name__ == "__main__":
    samples = [
        "Bitcoin surges to a new all-time high as institutional demand soars",
        "Crypto market crashes amid regulatory crackdown fears",
        "Bitcoin price holds steady around 60k",
    ]
    for s in samples:
        print(f"{score_sentiment(s):+.3f}  {s}")
