"""The orchestrator: decides how each turn is handled, delegates searches to the retrieval agent, and replies.

    START -> decide --(route_after_decide)--> delegate -> reply -> END
                    \\-------------------------------> reply -> END

- decide: the model chooses a route and a reason (a Decision, through tool calling). A reply without a valid
  Decision, or one Ollama cannot read, is the model's failure (parse_error), and the turn falls back to a
  declining reply without searching. A model server that cannot be reached or is busy is infrastructure.
- delegate: sends the user's words, today's date and the previous search to the retrieval agent over A2A.
- reply: a template (replies.py), never the model's text.

Memory: a checkpointer keeps each conversation's previous search, so a follow-up can continue it. The state
holds only plain values and JSON text: the checkpointer brings back our own classes only if each one is
registered, and an unregistered one comes back silently changed (checked with langgraph 1.2.14).
"""

from __future__ import annotations

import time
from datetime import date
from typing import Literal, NotRequired, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from agentic_earth_observation.catalog_mcp.tool_log import TRACE_ID
from agentic_earth_observation.evals.golden import Route
from agentic_earth_observation.failures import describe, is_infrastructure
from agentic_earth_observation.llm import unreadable_reply
from agentic_earth_observation.orchestrator.client import RetrievalClient
from agentic_earth_observation.orchestrator.models import (
    Decision,
    DecisionRoute,
    Reason,
    TurnOutput,
    scored_status,
)
from agentic_earth_observation.orchestrator.prompts import decide_system_prompt
from agentic_earth_observation.orchestrator.replies import compose_reply
from agentic_earth_observation.retrieval_agent.models import (
    ModelFailure,
    PreviousSearch,
    RetrievalRequest,
    RetrievalResult,
    Usage,
)
from agentic_earth_observation.retrieval_agent.prompts import retrieval_user_message


class OrchestratorState(TypedDict):
    """One conversation's state. The previous search is kept from turn to turn; run_turn resets the rest."""

    previous_search: NotRequired[str | None]  # the last successful search, as JSON (a PreviousSearch)
    text: str  # the user's words
    today: date
    trace_id: str
    route: DecisionRoute | None  # the decision; None when the model gave none
    reason: Reason | None
    failure: ModelFailure | None  # the model's failure, in the decision or in the retrieval
    failure_detail: str | None
    infra_error: str | None
    decision_usage: str  # the decision's model call, as JSON (a Usage)
    retrieval: str | None  # the retrieval agent's result, as JSON (a RetrievalResult)
    reply: str
    cited_ids: list[str]


def route_after_decide(state: OrchestratorState) -> Literal["delegate", "reply"]:
    """The node after `decide`: "delegate" to run the search, or "reply" straight away.

    - "reply" when the decision step failed: `infra_error` or `failure` is set;
    - "delegate" when the decision's `route` is "delegate";
    - "reply" when it is "decline" or "clarify";
    - a state with no route, no failure and no infra_error cannot happen: raise ValueError.
    """
    if state["infra_error"] is not None or state["failure"] is not None:
        return "reply"
    if state["route"] == "delegate":
        return "delegate"
    if state["route"] in ("decline", "clarify"):
        return "reply"
    raise ValueError("decide set neither a route nor a failure")


def build_graph(
    model: BaseChatModel, client: RetrievalClient, model_id: str
) -> StateGraph[OrchestratorState, None, OrchestratorState, OrchestratorState]:
    """The orchestrator's graph for one turn, to compile with a checkpointer."""
    decider = model.with_structured_output(Decision, method="function_calling", include_raw=True)

    async def decide(state: OrchestratorState) -> dict[str, object]:
        request = RetrievalRequest(text=state["text"], today=state["today"], previous_search=_previous(state))
        messages = [
            SystemMessage(decide_system_prompt(state["today"])),
            HumanMessage(retrieval_user_message(request)),
        ]
        started = time.perf_counter()
        usage = Usage(model=model_id, model_calls=1)
        try:
            out = await decider.ainvoke(messages)
        except Exception as err:
            spent = usage.model_copy(update={"seconds": time.perf_counter() - started}).model_dump_json()
            if (detail := unreadable_reply(err)) is not None:
                return {"failure": "parse_error", "failure_detail": detail, "decision_usage": spent}
            if is_infrastructure(err):
                return {"infra_error": describe(err), "decision_usage": spent}
            raise
        if not isinstance(out, dict):  # with include_raw=True it is always a dict
            raise TypeError(f"expected a dict from the structured output, got {type(out).__name__}")
        raw = out["raw"]
        if isinstance(raw, AIMessage) and raw.usage_metadata is not None:
            tokens = {
                "input_tokens": raw.usage_metadata["input_tokens"],
                "output_tokens": raw.usage_metadata["output_tokens"],
            }
            usage = usage.model_copy(update=tokens)
        spent = usage.model_copy(update={"seconds": time.perf_counter() - started}).model_dump_json()
        decision = out["parsed"]
        if not isinstance(decision, Decision):
            # No Decision in the reply (Ollama ignores tool_choice, so the model may answer in text), or an
            # invalid one: the model's failure, never repaired.
            error = out["parsing_error"]
            detail = "the model answered without a Decision" if error is None else describe(error)
            return {"failure": "parse_error", "failure_detail": detail, "decision_usage": spent}
        return {"route": decision.route, "reason": decision.reason, "decision_usage": spent}

    async def delegate(state: OrchestratorState) -> dict[str, object]:
        request = RetrievalRequest(text=state["text"], today=state["today"], previous_search=_previous(state))
        result = await client.retrieve(request, state["trace_id"])  # RetrievalAgentError: a bug, raised
        update: dict[str, object] = {
            "retrieval": result.model_dump_json(),
            "failure": result.failure,
            "failure_detail": result.failure_detail,
            "infra_error": result.infra_error,
        }
        if result.search is not None:  # a successful search is the one a follow-up continues
            update["previous_search"] = result.search.model_dump_json()
        return update

    def reply(state: OrchestratorState) -> dict[str, object]:
        text, cited_ids = compose_reply(
            state["route"],
            state["reason"],
            _retrieval(state["retrieval"]),
            state["failure"],
            state["infra_error"],
        )
        return {"reply": text, "cited_ids": cited_ids}

    graph = StateGraph(OrchestratorState)
    graph.add_node("decide", decide)
    graph.add_node("delegate", delegate)
    graph.add_node("reply", reply)
    graph.add_edge(START, "decide")
    graph.add_conditional_edges("decide", route_after_decide, ["delegate", "reply"])
    graph.add_edge("delegate", "reply")
    graph.add_edge("reply", END)
    return graph


def _previous(state: OrchestratorState) -> PreviousSearch | None:
    stored = state.get("previous_search")
    return None if stored is None else PreviousSearch.model_validate_json(stored)


def _retrieval(stored: str | None) -> RetrievalResult | None:
    return None if stored is None else RetrievalResult.model_validate_json(stored)


class Orchestrator:
    """The orchestrator for any number of conversations, each remembered in memory under its own id."""

    def __init__(self, model: BaseChatModel, client: RetrievalClient, model_id: str) -> None:
        self.graph: CompiledStateGraph[OrchestratorState, None, OrchestratorState, OrchestratorState] = (
            build_graph(model, client, model_id).compile(checkpointer=InMemorySaver())
        )

    async def run_turn(self, conversation_id: str, text: str, trace_id: str, today: date) -> TurnOutput:
        """Handle one user turn of a conversation and report what it did.

        `trace_id` files the retrieval agent's tool calls under this turn; `today` is the date the agents take
        as today. A bug, or an error of the retrieval agent (RetrievalAgentError), is raised.
        """
        if not TRACE_ID.fullmatch(trace_id) or not TRACE_ID.fullmatch(conversation_id):
            raise ValueError("ids are 1 to 64 letters, digits, '.', '_' or '-'")
        try:
            RetrievalRequest(text=text, today=today)  # the text must be one the retrieval agent accepts
        except ValidationError as err:
            raise ValueError(f"the text cannot be sent: {err.errors()[0]['msg']}") from None
        start: OrchestratorState = {
            "text": text,
            "today": today,
            "trace_id": trace_id,
            "route": None,
            "reason": None,
            "failure": None,
            "failure_detail": None,
            "infra_error": None,
            "decision_usage": Usage().model_dump_json(),
            "retrieval": None,
            "reply": "",
            "cited_ids": [],
        }
        state = await self.graph.ainvoke(start, {"configurable": {"thread_id": conversation_id}})
        retrieval = _retrieval(state["retrieval"])
        if retrieval is not None:
            route = Route.DELEGATE
        else:
            route = Route.CLARIFY if state["route"] == "clarify" else Route.DECLINE
        return TurnOutput(
            trace_id=trace_id,
            route=route,
            status=scored_status(state["route"], retrieval),
            answer=state["reply"],
            cited_ids=state["cited_ids"],
            failure=state["failure"],
            failure_detail=state["failure_detail"],
            infra_error=state["infra_error"],
            decision_usage=Usage.model_validate_json(state["decision_usage"]),
            retrieval=retrieval,
        )
