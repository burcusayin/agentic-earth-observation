"""Retrieval agent: a LangGraph loop in which the model calls the catalogue tools until it can answer.

    START -> model --(tool calls)--> tools -> model -> ... --(a reply without tool calls)--> END
                   \\--(a malformed call)--> malformed -> END
                   \\--(tool calls, but no model call left)--> out_of_steps -> END

`build_graph` builds the loop for one turn. `run_retrieval` (next step) opens the MCP session, passes the
system prompt and the request as the first messages, and turns the final state into a RetrievalResult.

Tool errors (a refused argument, a catalogue error) come back from the MCP tools as error messages and go to
the model, which may correct its call. Any other exception escapes the graph, so the runner counts it as
infrastructure. The model's own failures (a malformed tool call, running out of steps) set `failure` and end
the turn; see docs/decisions.md, 2026-11-02. a2a_server.py publishes the agent over A2A.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, ToolException
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from agentic_earth_observation.retrieval_agent.models import ModelFailure

AGENT_TOOLS = ("list_products", "resolve_place", "search_scenes")  # not get_scene: decision of 2026-10-31


class RetrievalState(MessagesState):
    """What the graph carries through one turn: the messages, and how the model is doing."""

    model_calls: int  # how many replies the model has given in this turn
    failure: ModelFailure | None  # set when the model fails the turn; the turn then ends


def build_graph(
    model: BaseChatModel, tools: Sequence[BaseTool], max_model_calls: int
) -> CompiledStateGraph[RetrievalState, None, RetrievalState, RetrievalState]:
    """The retrieval loop for one turn. The caller passes the system prompt and the request as messages."""
    # Bind the tools to the model, so that it can call them.
    bound = model.bind_tools(tools)
    names = {t.name for t in tools}

    # The model node: an async function of the state. It sends all the messages so far to the bound
    #    model and returns an update with the reply and one more model call.
    async def call_model(state: RetrievalState) -> dict[str, object]:
        reply = await bound.ainvoke(state["messages"])
        return {"messages": [reply], "model_calls": state["model_calls"] + 1}

    # The route after the model: a plain function of the state that names the next node.
    #    - "malformed" if the reply has unparseable tool calls, or calls a tool not in `tools`;
    #    - END if the reply calls no tool (it is the answer);
    #    - "out_of_steps" if it calls tools but the model has already replied max_model_calls times;
    #    - "tools" otherwise.
    def route(state: RetrievalState) -> str:
        reply = state["messages"][-1]
        if not isinstance(reply, AIMessage):
            raise TypeError(f"expected the model's reply, got {type(reply).__name__}")
        if reply.invalid_tool_calls or any(call["name"] not in names for call in reply.tool_calls):
            return "malformed"
        if not reply.tool_calls:
            return END
        if state["model_calls"] >= max_model_calls:
            return "out_of_steps"
        return "tools"

    # Two small nodes that only set `failure`: "malformed" sets "malformed_tool_call" and
    #    "out_of_steps" sets "step_limit".
    def malformed(state: RetrievalState) -> dict[str, object]:
        return {"failure": "malformed_tool_call"}

    def out_of_steps(state: RetrievalState) -> dict[str, object]:
        return {"failure": "step_limit"}

    # Wire and compile the graph: a StateGraph of RetrievalState with the four nodes ("model",
    #    "tools" as a ToolNode, "malformed", "out_of_steps"), the edges START -> model, tools -> model,
    #    malformed -> END and out_of_steps -> END, and the route as the conditional edge out of "model".
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
