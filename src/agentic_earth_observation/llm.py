"""The chat model under test, built from Settings with the options recorded in docs/decisions.md.

Every role (orchestrator and retrieval agent) uses this one factory with the same options, so Ollama never
reloads the model between calls. The two functions at the end say how to read the model server's errors.
"""

from __future__ import annotations

import ollama
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from agentic_earth_observation.config import Settings

OLLAMA_CONTEXT = 16384  # Ollama's default context is 4k on this hardware and truncates silently
SEED = 42
OLLAMA_BUSY = 503  # Ollama's answer when its queue of requests is full
# How Ollama (0.40.1) reports a reply its parser could not read: HTTP 500 if nothing was sent yet, otherwise
# an error inside the stream, which the client gives the status -1.
OLLAMA_UNREADABLE = (500, -1)


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


def model_server_busy(err: BaseException) -> bool:
    """Whether the model server turned the call away because it was busy: infrastructure, not the model."""
    return isinstance(err, ollama.ResponseError) and err.status_code == OLLAMA_BUSY


def unreadable_reply(err: BaseException) -> str | None:
    """The model server's message if it could not read the model's reply, otherwise None.

    For both local models Ollama parses the model's tool calls itself, and reports one it cannot read
    (malformed XML or JSON, or for ministral-3 a tool it was not given) as an error. A crash of the model
    server gives the same status, so these errors count as the model's failure, with the message kept for the
    error analysis (decision of 2026-11-02): the reverse would exclude a model's own failures from its score.
    """
    if isinstance(err, ollama.ResponseError) and err.status_code in OLLAMA_UNREADABLE:
        return " ".join(str(err.error).split())[:300]
    return None
