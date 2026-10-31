"""Probe how each local model calls tools and produces structured output. Needs Ollama; makes no other call.

    uv run python scripts/probe_models.py                 # both local models
    uv run python scripts/probe_models.py qwen3.5:9b      # one model

The MCP server runs in-process with the real tool schemas and an offline catalogue that finds nothing, so no
request leaves the machine. Only dev-split questions are used. Models are probed one at a time and unloaded
afterwards, so two never share the GPU.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any, Literal

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import BaseModel

from agentic_earth_observation.catalog.gazetteer import Gazetteer
from agentic_earth_observation.catalog.stac import SceneSummary
from agentic_earth_observation.catalog_mcp.server import build_server
from agentic_earth_observation.config import Settings
from agentic_earth_observation.domain import Product, SearchRequest
from agentic_earth_observation.llm import build_chat_model

MODELS = ["qwen3.5:9b", "ministral-3:8b"]
QUESTIONS = [
    "Show me Sentinel-2 scenes of Milan from March 2025 with little cloud.",
    "Find radar scenes over Naples for the first week of September 2025.",
]
SYSTEM = (
    "You find satellite scenes with the tools. Resolve the place with resolve_place first, then call "
    "search_scenes with the box it returns. Never invent coordinates or scene ids. Today is 2025-12-15."
)
DECIDE = (
    "Route the user's request. 'delegate' for a satellite scene search, 'decline' for anything the scene "
    "catalogue cannot answer, 'clarify' when no place or no period is given."
)
OUT_OF_SCOPE = "What will the weather be like in Naples this weekend?"
MAX_MODEL_CALLS = 3


class NoScenes:
    """An offline catalogue: every search finds nothing."""

    def search(self, request: SearchRequest) -> list[SceneSummary]:
        return []

    def get_scene(self, product: Product, scene_id: str) -> SceneSummary | None:
        return None


class Decision(BaseModel):
    route: Literal["delegate", "decline", "clarify"]
    reason: Literal["search", "out_of_scope", "missing_place", "missing_period"]


def describe(reply: AIMessage) -> str:
    usage: dict[str, Any] = dict(reply.usage_metadata or {})
    load_ms = (reply.response_metadata.get("load_duration") or 0) / 1e6
    thinking = bool(reply.additional_kwargs.get("reasoning_content"))
    return (
        f"tokens in/out {usage.get('input_tokens')}/{usage.get('output_tokens')}, "
        f"model load {load_ms:.0f} ms, thinking leaked: {thinking}, "
        f"invalid tool calls: {len(reply.invalid_tool_calls)}"
    )


async def ask(model: BaseChatModel, tools: list[BaseTool], question: str) -> None:
    by_name = {tool.name: tool for tool in tools}
    messages: list[BaseMessage] = [SystemMessage(SYSTEM), HumanMessage(question)]
    bound = model.bind_tools(tools)
    print(f"\n  Q: {question}")
    for step in range(1, MAX_MODEL_CALLS + 1):
        started = time.perf_counter()
        reply = await bound.ainvoke(messages)
        if not isinstance(reply, AIMessage):
            raise TypeError(f"expected an AIMessage, got {type(reply).__name__}")
        print(f"    step {step} ({time.perf_counter() - started:.1f} s; {describe(reply)})")
        if not reply.tool_calls:
            print(f"      no tool call; text: {str(reply.content)[:160]!r}")
            return
        messages.append(reply)
        for call in reply.tool_calls:
            print(f"      calls {call['name']}({json.dumps(call['args'])})")
            tool = by_name.get(call["name"])
            result = (
                await tool.ainvoke(call)
                if tool
                else ToolMessage(f"unknown tool {call['name']}", tool_call_id=call["id"])
            )
            if not isinstance(result, ToolMessage):
                raise TypeError(f"expected a ToolMessage, got {type(result).__name__}")
            messages.append(result)
            print(f"        -> {'ERROR ' if result.status == 'error' else ''}{str(result.content)[:140]}")


async def decide(model: BaseChatModel) -> None:
    for method in ("json_schema", "function_calling"):
        structured = model.with_structured_output(Decision, method=method, include_raw=True)
        started = time.perf_counter()
        try:
            out = await structured.ainvoke([SystemMessage(DECIDE), HumanMessage(OUT_OF_SCOPE)])
        except Exception as err:  # the probe reports every failure and moves on
            print(f"  decision via {method}: FAILED ({type(err).__name__}: {str(err)[:120]})")
            continue
        if not isinstance(out, dict):
            raise TypeError(f"expected a dict, got {type(out).__name__}")
        error = out.get("parsing_error")
        print(
            f"  decision via {method} ({time.perf_counter() - started:.1f} s): {out.get('parsed')!r}"
            + (f"  PARSE ERROR {error}" if error else "")
        )


async def probe(name: str) -> None:
    print(f"\n=== {name}")
    model = build_chat_model(Settings(llm_model=name))
    server = build_server(NoScenes(), Gazetteer.load())
    async with create_connected_server_and_client_session(server) as session:
        tools = [tool for tool in await load_mcp_tools(session) if tool.name != "get_scene"]
        for question in QUESTIONS:
            await ask(model, tools, question)
        await decide(model)
    # Unload the model through Ollama's API (keep_alive 0) so the next model has the GPU to itself.
    settings = Settings()
    httpx.post(f"{settings.ollama_url}/api/generate", json={"model": name, "keep_alive": 0}, timeout=30)


def main() -> None:
    for name in sys.argv[1:] or MODELS:
        asyncio.run(probe(name))


if __name__ == "__main__":
    main()
