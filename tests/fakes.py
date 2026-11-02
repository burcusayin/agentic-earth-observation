"""Shared test doubles: a fake catalogue (no network), a scripted chat model (no LLM) and sample data."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field

from agentic_earth_observation.catalog.stac import CatalogError, SceneSummary
from agentic_earth_observation.domain import Product, SearchRequest

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
