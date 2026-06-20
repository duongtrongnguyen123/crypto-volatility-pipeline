"""Pluggable reasoning-LLM backends for the LOCAL live-serving tier.

THREE-TIER ARCHITECTURE
=======================
This crash-prediction system is split across three tiers so the heavy training
stays offline while serving stays cheap, online, and always-available:

    1. Kaggle offline lab  (RTX 6000 Pro, NO internet, batch-only)
       The big LLM (e.g. Nemotron / Qwen-32B) is run zero-shot, and a small
       LoRA adapter is fine-tuned on the historical news->crash corpus. This box
       cannot serve anything live — it has no network and runs in batch kernels.

    2. Adapter artifact  (the hand-off)
       The trained LoRA/merged weights are exported from Kaggle and downloaded to
       the local serving box. That directory is pointed at by the env var
       `TRR_MODEL_DIR`. This is the ONLY thing that crosses the air-gap.

    3. Local serving  (this code; has internet + a small/optional GPU)
       FastAPI + the TRR pipeline run continuously here. The backend is
       PLUGGABLE so the exact same pipeline serves with:
         * "heuristic"  -> trr.llm.MockLLM  (CPU, deterministic, always works;
                           the safe default for live serving and for tests)
         * "finetuned"  -> trr.llm.HFReasoningLLM on the downloaded adapter dir
         * "api"        -> a hosted LLM over HTTP (the local box has internet)

`get_backend(name)` returns a `trr.llm.ReasoningLLM`. Every tier degrades
gracefully to the heuristic MockLLM when its heavier dependency or artifact is
missing, so the service NEVER fails to come up.
"""
from __future__ import annotations

import logging
import os

from trr.llm import MockLLM, ReasoningLLM

logger = logging.getLogger("serving.model_backend")

# Backend names this factory understands.
BACKENDS = ("heuristic", "finetuned", "api")


class APIReasoningLLM(ReasoningLLM):
    """Thin backend that calls a hosted LLM over HTTP.

    The local serving box has internet, so we can offload the reasoning to a
    hosted chat-completions endpoint (OpenAI-compatible) instead of loading a
    model locally. Configuration is read from the environment:

        LLM_API_URL   chat-completions URL (e.g. https://api.openai.com/v1/chat/completions)
        LLM_API_KEY   bearer token
        LLM_MODEL     model id (default "gpt-4o-mini")

    `generate()` builds the standard OpenAI chat payload and POSTs it. If either
    `LLM_API_URL` or `LLM_API_KEY` is unset, the instance is *not configured*
    and transparently degrades to the deterministic MockLLM heuristic — so this
    backend is fully testable offline with no network and no secrets.
    """

    def __init__(self, url: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout: float = 30.0) -> None:
        self.url = url or os.getenv("LLM_API_URL", "")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        # Fallback used whenever we are not configured or the call fails.
        self._fallback = MockLLM()
        self.configured = bool(self.url and self.api_key)
        if not self.configured:
            logger.warning(
                "APIReasoningLLM: LLM_API_URL/LLM_API_KEY unset; degrading to "
                "the heuristic MockLLM (offline-safe)."
            )

    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.0) -> str:
        if not self.configured:
            # Not configured: behave exactly like the heuristic backend so the
            # default ReasoningLLM prompt-parsing path still produces a result.
            return self._fallback.generate(prompt, max_new_tokens, temperature)
        try:
            import httpx
        except ImportError:  # pragma: no cover - httpx is a serving dependency
            logger.warning("httpx missing; falling back to heuristic backend.")
            return self._fallback.generate(prompt, max_new_tokens, temperature)

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = httpx.post(self.url, json=payload, headers=headers,
                              timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:  # network / shape errors -> safe degradation
            logger.warning("API call failed (%s); using heuristic fallback.", exc)
            return self._fallback.generate(prompt, max_new_tokens, temperature)


def _get_finetuned() -> ReasoningLLM:
    """Load the fine-tuned adapter from `TRR_MODEL_DIR`, else degrade to Mock.

    `TRR_MODEL_DIR` points at the LoRA/merged model directory exported from the
    Kaggle offline lab and downloaded to this box. If the env var is unset, the
    directory is missing, or transformers/torch can't load it, we log a warning
    and return the deterministic MockLLM so serving still comes up.
    """
    model_dir = os.getenv("TRR_MODEL_DIR", "")
    if not model_dir or not os.path.isdir(model_dir):
        logger.warning(
            "TRR_MODEL_DIR unset or not a directory (%r); falling back to the "
            "heuristic MockLLM. Download the Kaggle adapter and set TRR_MODEL_DIR "
            "to enable the fine-tuned backend.", model_dir,
        )
        return MockLLM()
    try:
        from trr.llm import HFReasoningLLM

        device = os.getenv("TRR_MODEL_DEVICE", "cuda")
        return HFReasoningLLM(model_dir, device=device)
    except Exception as exc:  # transformers/torch/model missing or load failure
        logger.warning(
            "Could not load fine-tuned model from %r (%s); falling back to the "
            "heuristic MockLLM.", model_dir, exc,
        )
        return MockLLM()


def get_backend(name: str = "heuristic") -> ReasoningLLM:
    """Return a `ReasoningLLM` for the requested backend.

    name:
        "heuristic" (default) -> trr.llm.MockLLM. CPU-only, deterministic,
            no model/network. The safe live default and the test backend.
        "finetuned" -> trr.llm.HFReasoningLLM on `TRR_MODEL_DIR` (the Kaggle
            adapter downloaded locally); degrades to MockLLM if unavailable.
        "api" -> APIReasoningLLM (hosted LLM over HTTP via LLM_API_URL/KEY);
            degrades to MockLLM if unconfigured.

    Unknown names raise ValueError.
    """
    name = (name or "heuristic").lower()
    if name == "heuristic":
        return MockLLM()
    if name == "finetuned":
        return _get_finetuned()
    if name == "api":
        return APIReasoningLLM()
    raise ValueError(f"unknown backend {name!r}; choose one of {BACKENDS}")


def backend_label(backend: ReasoningLLM) -> str:
    """Human-readable name of an instantiated backend (for /health)."""
    if isinstance(backend, APIReasoningLLM):
        return "api" if backend.configured else "api(degraded->heuristic)"
    if isinstance(backend, MockLLM):
        return "heuristic"
    return type(backend).__name__


def is_real_model(backend: ReasoningLLM) -> bool:
    """True if a real (non-heuristic) model is actually serving requests."""
    if isinstance(backend, MockLLM):
        return False
    if isinstance(backend, APIReasoningLLM):
        return backend.configured
    return True
