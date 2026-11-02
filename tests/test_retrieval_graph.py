"""Specification for the retrieval loop (build_graph), run against the real MCP server in-process.

The model is scripted (no LLM) and the catalogue is fake (no network); the tools are the server's own, loaded
through langchain-mcp-adapters exactly as the agent will load them, and every call reaching the server is
recorded in its tool-call log.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import ollama
import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.shared.memory import create_connected_server_and_client_session

from agentic_earth_observation.catalog.gazetteer import Gazetteer
from agentic_earth_observation.catalog_mcp.server import build_server
from agentic_earth_observation.catalog_mcp.tool_log import ToolCallLog
from agentic_earth_observation.retrieval_agent.agent import AGENT_TOOLS, RetrievalState, build_graph
from tests.fakes import (
    MID_STREAM,
    SCENE,
    FakeCatalog,
    FakeOllama,
    OllamaError,
    ScriptedModel,
    ollama_tool_call,
    tool_call,
)

pytestmark = pytest.mark.anyio

MILAN_MARCH = {
    "product": "sentinel-2-l2a",
    "bbox": [9.0409, 45.3867, 9.2781, 45.5358],
    "start_date": "2025-03-01",
    "end_date": "2025-03-31",
    "max_cloud_cover": 20,
}
START: list[AnyMessage] = [
    SystemMessage("You find satellite scenes."),
    HumanMessage("REQUEST: Milan, March 2025, little cloud"),
]


def first_state() -> RetrievalState:
    """The state a turn starts from: the system prompt and the request, and no model call yet."""
    return RetrievalState(messages=list(START), model_calls=0, failure=None, failure_detail=None)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@asynccontextmanager
async def agent_tools(catalog: FakeCatalog, log: Path) -> AsyncIterator[list[BaseTool]]:
    """The tools the agent binds, served by the real MCP server over an in-process session."""
    server = build_server(catalog, Gazetteer.load(), ToolCallLog(log))
    async with create_connected_server_and_client_session(server) as session:
        yield [t for t in await load_mcp_tools(session) if t.name in AGENT_TOOLS]


async def run(
    script: Sequence[AIMessage], log: Path, catalog: FakeCatalog | None = None, max_model_calls: int = 6
) -> tuple[dict[str, Any], ScriptedModel]:
    """Run one turn with the scripted replies; return the final state and the model (with what it saw)."""
    model = ScriptedModel(messages=iter(script))
    async with agent_tools(catalog or FakeCatalog(), log) as tools:
        graph = build_graph(model, tools, max_model_calls)
        state = await graph.ainvoke(first_state())
    return state, model


def tool_messages(state: dict[str, Any]) -> list[ToolMessage]:
    return [m for m in state["messages"] if isinstance(m, ToolMessage)]


def recorded_tools(log: Path) -> list[str]:
    import json

    return [json.loads(line)["tool"] for line in log.read_text(encoding="utf-8").splitlines()]


async def test_the_agent_is_given_three_tools_but_not_get_scene() -> None:
    # Decision of 2026-10-31: no task needs get_scene, and some forbid it.
    assert set(AGENT_TOOLS) == {"list_products", "resolve_place", "search_scenes"}


async def test_the_model_sees_the_system_prompt_and_the_request_first(tmp_path: Path) -> None:
    _, model = await run([AIMessage("I can only search scene metadata.")], tmp_path / "calls.jsonl")
    assert model.seen[0] == START


async def test_a_search_goes_place_then_search_then_answer(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    script = [
        tool_call("resolve_place", name="Milan"),
        tool_call("search_scenes", "call-2", **MILAN_MARCH),
        AIMessage("One scene was found."),
    ]
    state, model = await run(script, log)
    assert state["failure"] is None
    assert state["model_calls"] == 3
    assert [m.name for m in tool_messages(state)] == ["resolve_place", "search_scenes"]
    assert all(m.status == "success" for m in tool_messages(state))
    assert SCENE.id in str(model.seen[2][-1].content)  # the model read the search result before answering
    assert state["messages"][-1].content == "One scene was found."
    assert recorded_tools(log) == ["resolve_place", "search_scenes"]


async def test_a_refused_call_goes_back_to_the_model_which_can_correct_it(tmp_path: Path) -> None:
    reversed_box = {**MILAN_MARCH, "bbox": [9.2781, 45.3867, 9.0409, 45.5358]}
    script = [
        tool_call("search_scenes", **reversed_box),
        tool_call("search_scenes", "call-2", **MILAN_MARCH),
        AIMessage("One scene was found."),
    ]
    state, _ = await run(script, tmp_path / "calls.jsonl")
    refused, accepted = tool_messages(state)
    assert refused.status == "error"
    assert "west must be less than east" in str(refused.content)
    assert accepted.status == "success"
    assert state["failure"] is None
    assert state["model_calls"] == 3


async def test_an_answer_without_a_tool_call_ends_the_turn(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    state, _ = await run([AIMessage("I can only search scene metadata.")], log)
    assert (state["model_calls"], state["failure"]) == (1, None)
    assert tool_messages(state) == []
    assert recorded_tools(log) == []


async def test_two_calls_in_one_reply_both_run(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    both = AIMessage(
        content="",
        tool_calls=[
            {"name": "resolve_place", "args": {"name": "Milan"}, "id": "call-1", "type": "tool_call"},
            {"name": "resolve_place", "args": {"name": "Napoli"}, "id": "call-2", "type": "tool_call"},
        ],
    )
    state, _ = await run([both, AIMessage("Both places are known.")], log)
    assert len(tool_messages(state)) == 2
    assert recorded_tools(log) == ["resolve_place", "resolve_place"]
    assert state["failure"] is None


async def test_running_out_of_steps_stops_before_another_tool_call(tmp_path: Path) -> None:
    # Each reply calls a tool, so the model would never stop. At the limit its last calls are not run:
    # nobody would read their results.
    log = tmp_path / "calls.jsonl"
    script = [tool_call("resolve_place", f"call-{i}", name="Milan") for i in range(10)]
    state, _ = await run(script, log, max_model_calls=4)
    assert state["model_calls"] == 4
    assert state["failure"] == "step_limit"
    assert len(tool_messages(state)) == 3
    assert recorded_tools(log) == ["resolve_place"] * 3


@pytest.mark.parametrize(
    "calls",
    [
        [{"name": "get_scene", "args": {"product": "sentinel-2-l2a", "scene_id": SCENE.id}}],  # not given
        [{"name": "get_weather", "args": {"place": "Napoli"}}],  # does not exist
        [  # one bad call spoils the reply: none of its calls runs
            {"name": "resolve_place", "args": {"name": "Milan"}},
            {"name": "get_scene", "args": {"product": "sentinel-2-l2a", "scene_id": SCENE.id}},
        ],
    ],
)
async def test_a_call_to_a_tool_the_agent_was_not_given_is_malformed(
    tmp_path: Path, calls: list[dict[str, Any]]
) -> None:
    log = tmp_path / "calls.jsonl"
    catalog = FakeCatalog()
    reply = AIMessage(
        content="",
        tool_calls=[{**c, "id": f"call-{i}", "type": "tool_call"} for i, c in enumerate(calls)],
    )
    state, _ = await run([reply, AIMessage("never reached")], log, catalog)
    assert state["failure"] == "malformed_tool_call"
    assert state["model_calls"] == 1
    assert tool_messages(state) == []
    assert recorded_tools(log) == []  # nothing reached the server
    assert catalog.lookups == []


async def test_an_unparseable_tool_call_is_malformed(tmp_path: Path) -> None:
    garbled = AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "name": "search_scenes",
                "args": '{"product": "sentinel-2-l2a", "bbox": [9.04',
                "id": "call-1",
                "error": "unterminated JSON",
                "type": "invalid_tool_call",
            }
        ],
    )
    state, _ = await run([garbled, AIMessage("never reached")], tmp_path / "calls.jsonl")
    assert (state["failure"], state["model_calls"]) == ("malformed_tool_call", 1)


async def test_an_error_outside_the_tools_reaches_the_caller() -> None:
    # Not the model's mistake: the runner must see it and count it as infrastructure; the model never sees it.
    @tool
    def resolve_place(name: str) -> str:
        """Find a known place by name."""
        raise RuntimeError("the MCP session was closed")

    model = ScriptedModel(
        messages=iter([tool_call("resolve_place", name="Milan"), AIMessage("never reached")])
    )
    graph = build_graph(model, [resolve_place], max_model_calls=6)
    with pytest.raises(RuntimeError, match="session was closed"):
        await graph.ainvoke(first_state())


@pytest.mark.parametrize(
    "error",
    [
        OllamaError(500, "XML syntax error on line 1: unexpected EOF"),  # qwen3.5's parser, before any output
        OllamaError(MID_STREAM, "tool 'get_scene' not found"),  # ministral-3's parser, after some text
    ],
)
async def test_a_reply_the_model_server_cannot_read_is_a_malformed_call(
    tmp_path: Path, error: OllamaError
) -> None:
    # Ollama parses the model's tool calls itself and reports one it cannot read as an error: the model's
    # failure, with the server's words kept; what the tools returned before stays (decision of 2026-11-02).
    log = tmp_path / "calls.jsonl"
    server = FakeOllama(ollama_tool_call("resolve_place", name="Milan"), error)
    async with agent_tools(FakeCatalog(), log) as tools:
        state = await build_graph(server.model(), tools, max_model_calls=6).ainvoke(first_state())
    assert (state["failure"], state["failure_detail"]) == ("malformed_tool_call", error.message)
    assert state["model_calls"] == 2
    assert [m.name for m in tool_messages(state)] == ["resolve_place"]
    assert recorded_tools(log) == ["resolve_place"]


@pytest.mark.parametrize(
    "error",
    [
        OllamaError(503, "server busy, please try again.  maximum pending requests exceeded"),
        OllamaError(404, 'model "qwen3.5:9b" not found, try pulling it first'),
        OllamaError(400, "does not support tools"),
    ],
)
async def test_other_model_server_errors_reach_the_caller(tmp_path: Path, error: OllamaError) -> None:
    # Not the model's reply: the runner decides whether it is infrastructure or a configuration error.
    async with agent_tools(FakeCatalog(), tmp_path / "calls.jsonl") as tools:
        graph = build_graph(FakeOllama(error).model(), tools, max_model_calls=6)
        with pytest.raises(ollama.ResponseError) as raised:
            await graph.ainvoke(first_state())
    assert raised.value.status_code == error.status
