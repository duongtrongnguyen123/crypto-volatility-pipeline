"""Phase 4 — Reasoning: predict crash probability from the pruned sub-graph.

Converts the pruned impact edges to (time, subject, polarity, object) tuples and
asks the LLM for a crash probability. This thin layer is also where extra
context (e.g. a decayed-memory summary) is assembled into the prompt context.
"""
from __future__ import annotations

from trr.llm import ReasoningLLM
from trr.schema import ImpactEdge


def reason_crash(
    edges: list[ImpactEdge],
    llm: ReasoningLLM,
    context: str = "",
    universe: list[str] | None = None,
) -> tuple[float, str]:
    """Predict (crash_prob, rationale) over the pruned impact edges."""
    tuples = [e.as_tuple() for e in edges]
    return llm.predict_crash(tuples, context=context, universe=universe)


def memory_context(decayed: list[tuple[ImpactEdge, float]], top: int = 5) -> str:
    """Summarise the most relevant decayed-memory edges as a prompt prefix."""
    if not decayed:
        return "No prior impacts in temporal memory."
    parts = [
        f"{e.subject}->{e.object}({'+' if e.polarity >= 0 else '-'},R={r:.2f})"
        for e, r in decayed[:top]
    ]
    return "Temporal memory (recent decayed impacts): " + ", ".join(parts)
