"""The chat model under test, built from Settings with the options recorded in docs/decisions.md.

Every role (orchestrator and retrieval agent) uses this one factory with the same options, so Ollama never
reloads the model between calls.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from agentic_earth_observation.config import Settings

OLLAMA_CONTEXT = 16384  # Ollama's default context is 4k on this hardware and truncates silently
SEED = 42


def build_chat_model(settings: Settings) -> BaseChatModel:
    """The configured chat model: deterministic options, thinking off, bounded time and output."""
    if settings.llm_provider == "ollama":
        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_url,
            num_ctx=OLLAMA_CONTEXT,
            temperature=0,
            seed=SEED,
            reasoning=False,
            keep_alive="30m",
            num_predict=settings.num_predict,
            client_kwargs={"timeout": settings.model_timeout_s},
        )
    raise NotImplementedError(
        "the Mistral provider arrives with the paid evaluation runs; set EO_LLM_PROVIDER=ollama for now"
    )
