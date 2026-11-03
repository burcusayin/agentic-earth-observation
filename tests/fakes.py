"""Shared test doubles: a fake catalogue (no network), a scripted chat model and a scripted Ollama server (no
LLM), and sample data."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from pydantic import Field

from agentic_earth_observation.catalog.stac import CatalogError, SceneSummary
from agentic_earth_observation.domain import Product, SearchRequest
from agentic_earth_observation.retrieval_agent.models import RetrievalRequest, RetrievalResult

SCENE = SceneSummary(
    id="S2A_T33TTG_20250730T100227_L2A",
    product=Product.SENTINEL_2_L2A,
    acquired=datetime(2025, 7, 30, 10, 9, 29, tzinfo=UTC),
    cloud_cover=4.1,
    platform="sentinel-2a",
)
ROME_JULY = {
    "product": "sentinel-2-l2a",
    "bbox": [12.2344, 41.6556, 12.8558, 42.141],
    "start_date": "2025-07-01",
    "end_date": "2025-07-31",
    "max_cloud_cover": 10,
}


class FakeCatalog:
    """Answers from a fixed list of scenes and records every search request it receives."""

    def __init__(self, scenes: list[SceneSummary] | None = None, *, fail: bool = False) -> None:
        self.scenes = scenes if scenes is not None else [SCENE]
        self.fail = fail
        self.requests: list[SearchRequest] = []
        self.lookups: list[str] = []

    def search(self, request: SearchRequest) -> list[SceneSummary]:
        self.requests.append(request)
        if self.fail:
            raise CatalogError("scene search failed: the catalogue did not answer (Read timed out)")
        return self.scenes[: request.limit]

    def get_scene(self, product: Product, scene_id: str) -> SceneSummary | None:
        self.lookups.append(scene_id)
        if self.fail:
            raise CatalogError("scene lookup failed: the catalogue did not answer (Read timed out)")
        return next((s for s in self.scenes if s.id == scene_id and s.product is product), None)


class BlockingCatalog(FakeCatalog):
    """Holds every call until `release` is set, so that a test can time out or cancel a call in progress."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def search(self, request: SearchRequest) -> list[SceneSummary]:
        self.started.set()
        self.release.wait(timeout=5)
        return super().search(request)

    def get_scene(self, product: Product, scene_id: str) -> SceneSummary | None:
        self.started.set()
        self.release.wait(timeout=5)
        return super().get_scene(product, scene_id)


class ScriptedModel(GenericFakeChatModel):
    """A chat model that gives the scripted replies in order and keeps every list of messages it was sent.

    Binding tools returns the model itself, so a graph can bind tools to it as it would to a real model.
    """

    seen: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,  # noqa: ANN401 - the signature of BaseChatModel.bind_tools
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,  # noqa: ANN401 - the signature of BaseChatModel._generate
    ) -> ChatResult:
        self.seen.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def tool_call(tool: str, call_id: str = "call-1", /, **args: object) -> AIMessage:
    """A model reply that calls one tool. Positional-only, so a tool argument may be called `name` too."""
    return AIMessage(
        content="", tool_calls=[{"name": tool, "args": args, "id": call_id, "type": "tool_call"}]
    )


MID_STREAM = -1  # the status the Ollama client gives an error that arrives inside a stream


@dataclass(frozen=True)
class OllamaError:
    """An error from Ollama: an HTTP status before any output, or MID_STREAM after some text."""

    status: int
    message: str


CONNECTION_REFUSED = "connection refused"
# A reply that starts and never ends, as from a model that keeps generating: no silence long enough for the
# model server's timeout, so only a deadline stops it.
NEVER_FINISHES = "never finishes"
OllamaStep = (
    dict[str, Any] | OllamaError | str
)  # a reply message, an error, CONNECTION_REFUSED or NEVER_FINISHES


def ollama_tool_call(tool: str, /, **args: object) -> dict[str, Any]:
    """A reply message that calls one tool, as Ollama streams it."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": tool, "arguments": args}}],
    }


def ollama_text(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


class FakeOllama:
    """An Ollama server that answers chat requests from a script, reached through httpx's MockTransport.

    The real ChatOllama client talks to it, so tests see the replies and errors exactly as that client turns
    them into messages and exceptions. Each reply reports 100 input and 10 output tokens.
    """

    def __init__(self, *steps: OllamaStep) -> None:
        self.steps = list(steps)
        self.requests: list[dict[str, Any]] = []

    def model(self) -> ChatOllama:
        return ChatOllama(model="qwen3.5:9b", async_client_kwargs={"transport": httpx.MockTransport(self)})

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat", request.url
        self.requests.append(json.loads(request.content))
        step = self.steps.pop(0)
        if step == CONNECTION_REFUSED:
            raise httpx.ConnectError("All connection attempts failed", request=request)
        if step == NEVER_FINISHES:
            return httpx.Response(200, content=_never_finishing())
        if isinstance(step, OllamaError) and step.status != MID_STREAM:
            return httpx.Response(step.status, json={"error": step.message})
        if isinstance(step, OllamaError):
            lines = [_chunk(ollama_text("Let me"), done=False), {"error": step.message}]
        else:
            assert isinstance(step, dict)
            lines = [_chunk(step, done=False), _chunk(ollama_text(""), done=True)]
        return httpx.Response(200, content="".join(json.dumps(line) + "\n" for line in lines).encode())


async def _never_finishing() -> AsyncIterator[bytes]:
    while True:
        yield (json.dumps(_chunk(ollama_text("Let me"), done=False)) + "\n").encode()
        await asyncio.sleep(0.05)


def _chunk(message: dict[str, Any], *, done: bool) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "model": "qwen3.5:9b",
        "created_at": "2025-12-15T10:00:00Z",
        "message": message,
        "done": done,
    }
    if done:
        chunk |= {"done_reason": "stop", "prompt_eval_count": 100, "eval_count": 10}
    return chunk


class FakeClient:
    """Stands in for the A2A client: records each call, then returns or raises the next scripted outcome."""

    def __init__(self, *outcomes: RetrievalResult | Exception) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[RetrievalRequest, str]] = []

    async def retrieve(self, request: RetrievalRequest, trace_id: str) -> RetrievalResult:
        self.calls.append((request, trace_id))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
