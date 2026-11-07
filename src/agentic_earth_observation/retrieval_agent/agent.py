"""Retrieval agent: a LangGraph loop in which the model calls the catalogue tools until it can answer.

    START -> model --(tool calls)--> tools -> model -> ... --(a reply without tool calls)--> END
                   \\--(a malformed call)--> malformed -> END
                   \\--(tool calls, but no model call left)--> out_of_steps -> END
                   \\--(the model server could not read the reply)--> END

`build_graph` builds the loop for one turn. `run_retrieval` runs one turn: it opens an MCP session that files
every call under the turn's trace id, passes the system prompt and the request as the first messages, and
turns the final state into a RetrievalResult with `result_from_turn`, from the tools' results only.

Tool errors (a refused argument, a catalogue error) come back from the MCP tools as error messages and go to
the model, which may correct its call. The model's own failures (a malformed tool call, a reply the model
server could not read, running out of steps) set `failure` and end the turn. Any other exception escapes the
graph: the runner reports a failure outside the model's control (a timeout, a server that cannot be reached or
is busy) as infrastructure, and raises anything else, a bug or a configuration error. See docs/decisions.md,
2026-11-02. a2a_server.py publishes the agent over A2A.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from datetime import timedelta
from typing import Any
from uuid import UUID

import httpx
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, ToolException
from langchain_mcp_adapters.sessions import StreamableHttpConnection, create_session
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from mcp.shared._httpx_utils import McpHttpClientFactory

from agentic_earth_observation.catalog.stac import SceneSummary
from agentic_earth_observation.catalog_mcp.server import PlaceResult, SearchResult
from agentic_earth_observation.catalog_mcp.tool_log import TRACE_HEADER, TRACE_ID, TRACE_ID_RULE
from agentic_earth_observation.config import Settings, is_local_url
from agentic_earth_observation.failures import describe, is_infrastructure, leaves
from agentic_earth_observation.llm import model_seconds, unreadable_reply
from agentic_earth_observation.retrieval_agent.models import (
    ModelFailure,
    PreviousSearch,
    RetrievalRequest,
    RetrievalResult,
    Usage,
)
from agentic_earth_observation.retrieval_agent.prompts import retrieval_system_prompt, retrieval_user_message

AGENT_TOOLS = ("list_products", "resolve_place", "search_scenes")  # not get_scene: decision of 2026-10-31
SAME_BOX = 0.001  # degrees: a box copied to 3 decimals is still the box a tool returned
# Calls the model may ask for in one reply; they run at once, and a search reaches a public service. A turn
# needs two or three; a reply over the limit fails the turn, as running out of steps does.
MAX_TOOL_CALLS_PER_REPLY = 8


class RetrievalState(MessagesState):
    """What the graph carries through one turn: the messages, and how the model is doing."""

    model_calls: int  # how many times the model has been called in this turn
    failure: ModelFailure | None  # set when the model fails the turn; the turn then ends
    failure_detail: str | None  # the model server's words, when it could not read the model's reply


def build_graph(
    model: BaseChatModel, tools: Sequence[BaseTool], max_model_calls: int
) -> CompiledStateGraph[RetrievalState, None, RetrievalState, RetrievalState]:
    """The retrieval loop for one turn. The caller passes the system prompt and the request as messages."""
    bound = model.bind_tools(tools)
    names = {t.name for t in tools}

    # A reply the model server could not read is the model's failure (decision of 2026-11-02): the turn ends
    # with the server's words kept. Any other error escapes the graph.
    async def call_model(state: RetrievalState) -> dict[str, object]:
        try:
            reply = await bound.ainvoke(state["messages"])
        except Exception as err:
            if (detail := unreadable_reply(err)) is None:
                raise
            return {
                "model_calls": state["model_calls"] + 1,
                "failure": "malformed_tool_call",
                "failure_detail": detail,
            }
        return {"messages": [reply], "model_calls": state["model_calls"] + 1}

    # After the model:
    # - END if the model node has already failed the turn (its reply could not be read);
    # - "malformed" if the reply has unparseable tool calls, or calls a tool not in `tools`;
    # - END if the reply calls no tool (it is the answer);
    # - "out_of_steps" if it calls tools but the model has already replied max_model_calls times, or asks for
    #   more calls at once than a turn may send;
    # - "tools" otherwise.
    def route(state: RetrievalState) -> str:
        if state["failure"] is not None:
            return END
        reply = state["messages"][-1]
        if not isinstance(reply, AIMessage):
            raise TypeError(f"expected the model's reply, got {type(reply).__name__}")
        if reply.invalid_tool_calls or any(call["name"] not in names for call in reply.tool_calls):
            return "malformed"
        if not reply.tool_calls:
            return END
        if state["model_calls"] >= max_model_calls or len(reply.tool_calls) > MAX_TOOL_CALLS_PER_REPLY:
            return "out_of_steps"
        return "tools"

    def malformed(state: RetrievalState) -> dict[str, object]:
        return {"failure": "malformed_tool_call"}

    def out_of_steps(state: RetrievalState) -> dict[str, object]:
        return {"failure": "step_limit"}

    graph = StateGraph(RetrievalState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=ToolException))
    graph.add_node("malformed", malformed)
    graph.add_node("out_of_steps", out_of_steps)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, ["tools", "malformed", "out_of_steps", END])
    graph.add_edge("tools", "model")
    graph.add_edge("malformed", END)
    graph.add_edge("out_of_steps", END)
    return graph.compile()


def local_mcp_client(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
    """The HTTP client of an MCP session with a server on this machine: it ignores proxy settings."""
    return httpx.AsyncClient(
        headers=headers, timeout=timeout or httpx.Timeout(30.0, read=300.0), auth=auth, trust_env=False
    )


async def run_retrieval(
    request: RetrievalRequest,
    trace_id: str,
    model: BaseChatModel,
    settings: Settings,
    *,
    http_client: McpHttpClientFactory | None = None,
) -> RetrievalResult:
    """Run one delegated turn and report what the tools returned in it.

    The MCP session sends `trace_id` in its X-EO-Trace header, so the server files every call under this turn.
    A failure outside the model's control (the turn's deadline, a server that cannot be reached, stops
    answering or is busy) gives a result with `infra_error`; any other exception is raised. `model` must be
    the one built from `settings`, whose model id is recorded. `http_client` lets tests reach an in-process
    server.
    """
    if not TRACE_ID.fullmatch(trace_id):
        raise ValueError(f"invalid trace id {trace_id!r}: use {TRACE_ID_RULE}")
    started = time.perf_counter()
    counter = _UsageCounter()
    deadline = asyncio.timeout(settings.turn_deadline_s)
    try:
        async with deadline:
            state = await _run_turn(request, trace_id, model, settings, http_client, counter)
    except Exception as err:
        if deadline.expired() and isinstance(err, TimeoutError):
            reason = f"the turn ran past its deadline ({settings.turn_deadline_s:g} s)"
        else:
            reason = _infrastructure_error(err)  # raises the error if it is not the infrastructure's
        usage = counter.usage(settings.llm_model, time.perf_counter() - started)
        return RetrievalResult(status="not_searched", infra_error=reason, usage=usage)
    result = result_from_turn(
        state["messages"], state["failure"], request.previous_search, failure_detail=state["failure_detail"]
    )
    usage = counter.usage(settings.llm_model, time.perf_counter() - started)
    return result.model_copy(update={"usage": usage})


async def _run_turn(
    request: RetrievalRequest,
    trace_id: str,
    model: BaseChatModel,
    settings: Settings,
    http_client: McpHttpClientFactory | None,
    counter: _UsageCounter,
) -> dict[str, Any]:
    """The graph's final state for one turn, with the tools of one MCP session."""
    connection: StreamableHttpConnection = {
        "transport": "streamable_http",
        "url": settings.mcp_url,
        "headers": {TRACE_HEADER: trace_id},
        "timeout": settings.mcp_timeout_s,
        "sse_read_timeout": settings.mcp_timeout_s,  # the wait for a tool's answer
        # And the wait for every MCP answer, so that one the client cannot read fails in time too.
        "session_kwargs": {"read_timeout_seconds": timedelta(seconds=settings.mcp_timeout_s)},
        "httpx_client_factory": http_client or (local_mcp_client if is_local_url(settings.mcp_url) else None),
    }
    async with create_session(connection) as session:
        await session.initialize()
        tools = [t for t in await load_mcp_tools(session) if t.name in AGENT_TOOLS]
        if missing := sorted(set(AGENT_TOOLS) - {t.name for t in tools}):
            raise RuntimeError(f"the MCP server at {settings.mcp_url} lacks the tool(s) {', '.join(missing)}")
        graph = build_graph(model, tools, settings.max_model_calls)
        first = RetrievalState(
            messages=[
                SystemMessage(retrieval_system_prompt(request.today)),
                HumanMessage(retrieval_user_message(request)),
            ],
            model_calls=0,
            failure=None,
            failure_detail=None,
        )
        # A backstop only: a turn takes at most 2N steps (N model replies, N - 1 rounds of tool calls and one
        # failure node) and LangGraph needs one more, so the graph's own step limit ends the turn first.
        config: RunnableConfig = {"recursion_limit": 2 * settings.max_model_calls + 1, "callbacks": [counter]}
        return await graph.ainvoke(first, config)


def _infrastructure_error(err: Exception) -> str:
    """Describe a failure of the infrastructure, or re-raise the error if it is not one.

    The MCP client runs in task groups, so errors may arrive wrapped in nested exception groups. A single bug
    inside them is re-raised as itself, with its own cause, so that its message is the one reported.
    """
    inside = leaves(err)
    bugs = [e for e in inside if not is_infrastructure(e)]
    if len(bugs) == 1 and isinstance(err, BaseExceptionGroup):
        bugs[0].__suppress_context__ = True  # the exception groups around it add nothing
        raise bugs[0]
    if bugs:
        raise err
    return describe(inside[0])


class _UsageCounter(AsyncCallbackHandler):
    """Counts the model's calls and tokens as they happen, so that a turn cut short still reports them."""

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.model_seconds = 0.0
        self.load_seconds = 0.0

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ANN401 - the signature of AsyncCallbackHandler.on_chat_model_start
    ) -> None:
        self.calls += 1

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,  # noqa: ANN401 - the signature of AsyncCallbackHandler.on_llm_end
    ) -> None:
        for generations in response.generations:
            for generation in generations:
                message = generation.message if isinstance(generation, ChatGeneration) else None
                if isinstance(message, AIMessage) and message.usage_metadata is not None:
                    self.input_tokens += message.usage_metadata["input_tokens"]
                    self.output_tokens += message.usage_metadata["output_tokens"]
                if isinstance(message, AIMessage):
                    working, loading = model_seconds(message.response_metadata)
                    self.model_seconds += working
                    self.load_seconds += loading

    def usage(self, model_id: str, seconds: float) -> Usage:
        return Usage(
            model=model_id,
            model_calls=self.calls,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            seconds=seconds,
            model_seconds=self.model_seconds,
            load_seconds=self.load_seconds,
        )


def result_from_turn(
    messages: Sequence[BaseMessage],
    failure: ModelFailure | None,
    previous: PreviousSearch | None,
    *,
    failure_detail: str | None = None,
) -> RetrievalResult:
    """What the turn found, read from the tools' results only, never from the model's text.

    - A search succeeded: the last successful one gives `ok` with its scenes, or `empty`, and its arguments.
    - Otherwise, if resolve_place did not know a place: `unknown_place`, with the latest such call's
      suggestions, even if a later call found a place.
    - Otherwise, if any tool call failed: `error`.
    - Otherwise: `not_searched`.

    The model's `failure`, with the model server's words about it if any, is kept alongside. `previous` is
    the search a follow-up continues: a box reused from it keeps its place name. A result that cannot be
    matched to its call, or that breaks the tool's result type, is refused with an error.
    """
    # The boxes tools returned, with their place names: the previous search's, then each place found in turn.
    places: list[tuple[str | None, list[float]]] = []
    if previous is not None:
        places.append((previous.place, previous.bbox))
    calls: dict[str | None, ToolCall] = {}  # the tool calls of the latest reply, by id
    scenes: list[SceneSummary] = []
    search: PreviousSearch | None = None  # the arguments of the last successful search
    suggestions: list[str] | None = None  # set when resolve_place did not know a place
    failed = False
    for message in messages:
        if isinstance(message, AIMessage):
            calls = {call["id"]: call for call in message.tool_calls}
            if len(calls) < len(message.tool_calls):
                raise ValueError(
                    "a reply repeats a tool call id, so its results cannot be matched to its calls"
                )
        elif isinstance(message, ToolMessage) and message.status == "error":
            failed = True
        elif isinstance(message, ToolMessage) and message.name == "resolve_place":
            answer = PlaceResult.model_validate(_structured(message))
            if not answer.found:
                suggestions = answer.suggestions
            elif answer.place is None:
                raise ValueError("resolve_place said it found the place but returned none")
            else:
                places.append((answer.place.name, answer.place.bbox))
        elif isinstance(message, ToolMessage) and message.name == "search_scenes":
            scenes = SearchResult.model_validate(_structured(message)).scenes
            arguments = calls[message.tool_call_id]["args"]
            named = {**arguments, "place": _place_of(arguments["bbox"], places)}
            search = PreviousSearch.model_validate_json(json.dumps(named))  # as A2A will carry it

    if search is not None:
        return RetrievalResult(
            status="ok" if scenes else "empty",
            scenes=scenes,
            search=search,
            failure=failure,
            failure_detail=failure_detail,
        )
    if suggestions is not None:
        return RetrievalResult(
            status="unknown_place", suggestions=suggestions, failure=failure, failure_detail=failure_detail
        )
    return RetrievalResult(
        status="error" if failed else "not_searched", failure=failure, failure_detail=failure_detail
    )


def _structured(message: ToolMessage) -> dict[str, Any]:
    """The structured result of a successful MCP tool call."""
    artifact = message.artifact
    if not isinstance(artifact, dict) or not isinstance(artifact.get("structured_content"), dict):
        raise ValueError(f"{message.name} succeeded but returned no structured result")
    content: dict[str, Any] = artifact["structured_content"]
    return content


def _place_of(bbox: list[float], places: list[tuple[str | None, list[float]]]) -> str | None:
    """The name of the latest place with this box; None when no tool returned the box."""
    for name, box in reversed(places):
        if all(abs(a - b) <= SAME_BOX for a, b in zip(bbox, box, strict=True)):
            return name
    return None
