"""The chat model under test, built from Settings with the options recorded in docs/decisions.md.

Every role (orchestrator and retrieval agent) uses this one factory with the same options, so Ollama never
reloads the model between calls. The two functions at the end say how to read the model server's errors.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import ollama
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from agentic_earth_observation.config import Settings, is_local_url

OLLAMA_CONTEXT = 16384  # Ollama's default context can be as small as 4k, and it truncates silently
SEED = 42
OLLAMA_KEEP_ALIVE = "30m"  # how long Ollama keeps the model loaded after a request
OLLAMA_CHECKED = "0.40.1"  # the version whose errors the functions at the end of this module were read from
OLLAMA_BUSY = 503  # Ollama's answer when its queue of requests is full
# How Ollama (0.40.1) reports a reply its parser could not read: HTTP 500 if nothing was sent yet, otherwise
# an error inside the stream, which the client gives the status -1.
OLLAMA_UNREADABLE = (500, -1)
# Ollama's words for its own trouble, in errors of the same statuses: no memory, a runner that died or never
# started, or no free slot, whatever the model wrote. From the Ollama 0.40.1 binary (decision of 2026-11-07).
OLLAMA_OWN_TROUBLE = (
    "process no longer running",
    "process has terminated",
    "waiting for llama-server to start",
    "too large for system memory",
    "no slots available",
)
PARSER_ERROR = "error parsing tool call"  # quotes the model's own text, which could hold those words


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
            keep_alive=OLLAMA_KEEP_ALIVE,
            num_predict=settings.num_predict,
            client_kwargs={
                "timeout": settings.model_timeout_s,
                "trust_env": not is_local_url(settings.ollama_url),
            },
        )
    raise NotImplementedError(
        "the Mistral provider arrives with the paid evaluation runs; set EO_LLM_PROVIDER=ollama for now"
    )


def model_seconds(metadata: Mapping[str, Any]) -> tuple[float, float]:
    """From a reply's metadata, the seconds Ollama spent on the model's work, and on loading the model.

    Ollama reports them in nanoseconds; a reply from elsewhere has none.
    """

    def seconds(*names: str) -> float:
        return sum(value for name in names if isinstance(value := metadata.get(name), int)) / 1e9

    return seconds("prompt_eval_duration", "eval_duration"), seconds("load_duration")


def model_server_failed(err: BaseException) -> bool:
    """Whether the model server was busy or failed by itself: infrastructure, not the model."""
    if not isinstance(err, ollama.ResponseError):
        return False
    return err.status_code == OLLAMA_BUSY or _own_trouble(err)


def _own_trouble(err: ollama.ResponseError) -> bool:
    words = str(err.error)
    return (
        err.status_code in OLLAMA_UNREADABLE
        and PARSER_ERROR not in words
        and any(trouble in words for trouble in OLLAMA_OWN_TROUBLE)
    )


def unreadable_reply(err: BaseException) -> str | None:
    """The model server's message if it could not read the model's reply, otherwise None.

    For both local models Ollama parses the model's tool calls itself, and reports one it cannot read
    (malformed XML or JSON, or for ministral-3 a tool it was not given) as an error. A crash of the model
    server gives the same status, so these errors count as the model's failure, with the message kept for the
    error analysis (decision of 2026-11-02): the reverse would exclude a model's own failures from its score.
    """
    if (
        isinstance(err, ollama.ResponseError)
        and err.status_code in OLLAMA_UNREADABLE
        and not _own_trouble(err)
    ):
        return " ".join(str(err.error).split())[:300]
    return None
