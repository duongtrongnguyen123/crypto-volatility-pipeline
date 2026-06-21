"""Retrieval-augmented few-shot (case-based reasoning) for TRR.

For each day, retrieve the most SIMILAR PAST days and inject their realized
crash/no-crash outcomes as dynamic few-shot examples into the reasoning context.
This grounds the LLM's probability in real historical analogues instead of the
generic hand-written examples — answering "is today like a known crash
precursor?", a *similarity* question that the recency-based decay memory cannot.

Retriever: TF-IDF over each day's news text + cosine similarity. Zero extra
model dependencies (sklearn is already on the Kaggle image) and fully offline.

Causality / leakage control: when retrieving analogues for day t we only ever
consider days with index < t - embargo, where embargo >= the label horizon. The
outcome of day t (and of any day whose 3-day forward window overlaps t) is never
visible to its own prediction. TF-IDF vocabulary is fit on the day texts; this is
standard practice and leaks no labels.
"""
from __future__ import annotations

import numpy as np


class CausalRAG:
    """Case-based few-shot retriever over per-day news text."""

    def __init__(self, embargo: int = 5, k: int = 5, min_sim: float = 0.05) -> None:
        # embargo: days to skip before the query day (>= label horizon, no overlap).
        self.embargo = embargo
        self.k = k
        self.min_sim = min_sim
        self._matrix = None       # (n_days, vocab) TF-IDF, L2-normalised rows
        self._dates: list = []

    def fit(self, day_texts: list[str], dates: list) -> "CausalRAG":
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._dates = list(dates)
        vec = TfidfVectorizer(max_features=4096, stop_words="english",
                              ngram_range=(1, 2), min_df=2)
        # Empty docs would make TF-IDF choke; substitute a single sentinel token.
        docs = [t if t and t.strip() else "__empty__" for t in day_texts]
        m = vec.fit_transform(docs).astype(np.float32)
        # L2-normalise rows so cosine = dot product.
        norms = np.sqrt(m.multiply(m).sum(axis=1))
        norms[norms == 0] = 1.0
        self._matrix = m.multiply(1.0 / norms).tocsr()
        return self

    def analogue_crash_rate(self, day_idx: int, labels: list[int]) -> float:
        """Numeric meta-feature: the fraction of the retrieved similar PAST days
        that actually crashed (0.0 if no eligible analogue). Causal — only days
        older than the embargo are considered.
        """
        if self._matrix is None:
            return 0.0
        cutoff = day_idx - self.embargo
        if cutoff <= 0:
            return 0.0
        q = self._matrix[day_idx]
        sims = np.asarray((self._matrix[:cutoff] @ q.T).todense()).ravel()
        order = np.argsort(-sims)[: self.k]
        picks = [j for j in order if sims[j] >= self.min_sim]
        if not picks:
            return 0.0
        return float(np.mean([labels[j] for j in picks]))

    def fewshot(self, day_idx: int, labels: list[int]) -> str:
        """Build the analogue few-shot block for the day at `day_idx`.

        `labels[j]` is the realized crash outcome (0/1) of day j. Only days with
        index < day_idx - embargo are eligible. Returns "" when no analogue
        exists (early days), so the prompt falls back to the static few-shot.
        """
        if self._matrix is None:
            return ""
        cutoff = day_idx - self.embargo
        if cutoff <= 0:
            return ""
        q = self._matrix[day_idx]
        # cosine against all eligible past rows (rows already L2-normalised).
        sims = np.asarray((self._matrix[:cutoff] @ q.T).todense()).ravel()
        order = np.argsort(-sims)[: self.k]
        picks = [(j, float(sims[j])) for j in order if sims[j] >= self.min_sim]
        if not picks:
            return ""
        lines = []
        for j, s in picks:
            outcome = "CRASHED (next 3d)" if labels[j] == 1 else "no crash"
            lines.append(f"  - {self._dates[j]} (similarity {s:.2f}): {outcome}")
        n_crash = sum(1 for j, _ in picks if labels[j] == 1)
        return (
            "HISTORICAL ANALOGUES — the most similar PAST days by news content and "
            f"what actually happened ({n_crash}/{len(picks)} of them crashed):\n"
            + "\n".join(lines)
            + "\nWeight these analogues: if today closely resembles prior CRASH "
            "days, lean higher; if it resembles calm days, lean lower.\n"
        )


def day_text(day_news: list) -> str:
    """Concatenate a day's news into one document for TF-IDF."""
    return " ".join(item.text() for item in day_news) if day_news else ""


if __name__ == "__main__":
    # Self-test: causality (no future leakage) + sensible retrieval.
    from datetime import date

    texts = [
        "exchange hacked funds stolen panic",      # 0
        "calm market mild gains tech",             # 1
        "regulator lawsuit crackdown selloff",     # 2
        "exchange insolvency contagion hack panic", # 3  ~ like 0
        "quiet day small moves",                   # 4  ~ like 1
    ]
    dates = [date(2022, 1, d) for d in (1, 2, 3, 4, 5)]
    labels = [1, 0, 1, 1, 0]
    rag = CausalRAG(embargo=1, k=2, min_sim=0.0).fit(texts, dates)
    # Day 3 ("exchange insolvency...") should retrieve day 0 ("exchange hacked...").
    block = rag.fewshot(3, labels)
    assert "2022-01-01" in block, block
    # Day 0 has no eligible past (cutoff <= 0) -> empty.
    assert rag.fewshot(0, labels) == ""
    print("[rag] causal self-test passed:\n" + block)
