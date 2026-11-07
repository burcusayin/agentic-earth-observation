"""The orchestrator: decides how each turn is handled, delegates searches to the retrieval agent, and replies.

    START -> decide --(route_after_decide)--> delegate -> reply -> END
                    \\-------------------------------> reply -> END

- decide: the model chooses a route and a reason (a Decision, through tool calling). A reply without a valid
  Decision, or one Ollama cannot read, is the model's failure (parse_error), and the turn falls back to a
  declining reply without searching. A model server that cannot be reached or is busy is infrastructure.
- delegate: sends the user's words, today's date and the previous search to the retrieval agent over A2A.
- reply: a template (replies.py), never the model's text.

Memory: a checkpointer keeps each conversation's previous search, so a follow-up can continue it, and the
question the last turn asked, so the next turn can answer it (next_pending). The state holds only plain values
and JSON text: the checkpointer brings back our own classes only if each one is
registered, and an unregistered one comes back silently changed (checked with langgraph 1.2.14).
"""

from __future__ import annotations

import asyncio
import time
from datetime import date
from typing import Literal, NotRequired, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from agentic_earth_observation.catalog_mcp.tool_log import TRACE_ID, TRACE_ID_RULE
from agentic_earth_observation.evals.golden import Route
from agentic_earth_observation.failures import describe, is_infrastructure
from agentic_earth_observation.llm import model_seconds, unreadable_reply
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
    Clarification,
    ModelFailure,
    PreviousSearch,
    RetrievalRequest,
    RetrievalResult,
    Usage,
)
from agentic_earth_observation.retrieval_agent.prompts import retrieval_user_message

REQUEST_LIMIT = 4000  # the longest request so far that an open question carries (Clarification.request)


class TurnRefused(ValueError):
    """A turn that run_turn will not take, before anything runs: invalid ids, or text the retrieval agent
    would refuse. Any other error from run_turn is a bug, even a ValueError (pydantic's ValidationError is
    one)."""


class OrchestratorState(TypedDict):
    """One conversation's state. The previous search and the open question are kept from turn to turn;
    run_turn resets the rest."""

    previous_search: NotRequired[str | None]  # the last successful search, as JSON (a PreviousSearch)
    pending: NotRequired[str | None]  # the question still open, as JSON (a Clarification)
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
    model: BaseChatModel, client: RetrievalClient, model_id: str, *, decision_deadline_s: float
) -> StateGraph[OrchestratorState, None, OrchestratorState, OrchestratorState]:
    """The orchestrator's graph for one turn, to compile with a checkpointer.

    A decision that runs past `decision_deadline_s` is the infrastructure's failure, like a retrieval turn
    past its deadline: the model server streams, so its own timeout bounds only a silence, not a reply.
    """
    decider = model.with_structured_output(Decision, method="function_calling", include_raw=True)

    async def decide(state: OrchestratorState) -> dict[str, object]:
        messages = [
            SystemMessage(decide_system_prompt(state["today"])),
            HumanMessage(retrieval_user_message(_request(state))),
        ]
        started = time.perf_counter()
        usage = Usage(model=model_id, model_calls=1)
        deadline = asyncio.timeout(decision_deadline_s)
        try:
            async with deadline:
                out = await decider.ainvoke(messages)
        except Exception as err:
            spent = usage.model_copy(update={"seconds": time.perf_counter() - started}).model_dump_json()
            if deadline.expired() and isinstance(err, TimeoutError):
                reason = f"the decision ran past its deadline ({decision_deadline_s:g} s)"
                return {"infra_error": reason, "decision_usage": spent}
            if (detail := unreadable_reply(err)) is not None:
                return {"failure": "parse_error", "failure_detail": detail, "decision_usage": spent}
            if is_infrastructure(err):
                return {"infra_error": describe(err), "decision_usage": spent}
            raise
        if not isinstance(out, dict):  # with include_raw=True it is always a dict
            raise TypeError(f"expected a dict from the structured output, got {type(out).__name__}")
        raw = out["raw"]
        if isinstance(raw, AIMessage) and raw.usage_metadata is not None:
            working, loading = model_seconds(raw.response_metadata)
            spent_so_far = {
                "input_tokens": raw.usage_metadata["input_tokens"],
                "output_tokens": raw.usage_metadata["output_tokens"],
                "model_seconds": working,
                "load_seconds": loading,
            }
            usage = usage.model_copy(update=spent_so_far)
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
        # RetrievalAgentError: a bug, raised
        result = await client.retrieve(_request(state), state["trace_id"])
        update: dict[str, object] = {
            "retrieval": result.model_dump_json(),
            "failure": result.failure,
            "failure_detail": result.failure_detail,
            "infra_error": result.infra_error,
        }
        # A successful search in a turn that did not fail is the one a follow-up continues: the user has not
        # been shown the search of a failed turn.
        if result.search is not None and result.failure is None:
            update["previous_search"] = result.search.model_dump_json()
        return update

    def reply(state: OrchestratorState) -> dict[str, object]:
        retrieval = _retrieval(state["retrieval"])
        text, cited_ids = compose_reply(
            state["route"], state["reason"], retrieval, state["failure"], state["infra_error"]
        )
        pending = next_pending(
            _pending(state),
            state["text"],
            text,
            asked=scored_status(state["route"], retrieval) == "clarified",
            failed=state["failure"] is not None or state["infra_error"] is not None,
        )
        return {
            "reply": text,
            "cited_ids": cited_ids,
            "pending": None if pending is None else pending.model_dump_json(),
        }

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


def _pending(state: OrchestratorState) -> Clarification | None:
    stored = state.get("pending")
    return None if stored is None else Clarification.model_validate_json(stored)


def _request(state: OrchestratorState) -> RetrievalRequest:
    """The turn as both models read it: the one that decides, and the retrieval agent it is sent to."""
    return RetrievalRequest(
        text=state["text"], today=state["today"], previous_search=_previous(state), clarifies=_pending(state)
    )


def next_pending(
    pending: Clarification | None, text: str, reply: str, *, asked: bool, failed: bool
) -> Clarification | None:
    """The question still open after a turn, which the next turn may answer.

    - The turn failed (the model, or the infrastructure): nothing was settled, so the open question stays.
    - The turn asked the user something (a clarification, or a place it does not know): its reply is the
      question, about the request so far. When the turn answered an earlier question, the request so far is
      the earlier one with this turn's words added; if that grows too long, it starts again from this turn.
    - Otherwise the question is settled: none is open.
    """
    if failed:
        return pending
    if not asked:
        return None
    request = text if pending is None else f"{pending.request} / {text}"
    if len(request) > REQUEST_LIMIT:
        request = text
    return Clarification(request=request, asked=reply)


def _retrieval(stored: str | None) -> RetrievalResult | None:
    return None if stored is None else RetrievalResult.model_validate_json(stored)


class Orchestrator:
    """The orchestrator for any number of conversations, each remembered in memory under its own id."""

    def __init__(
        self, model: BaseChatModel, client: RetrievalClient, model_id: str, *, decision_deadline_s: float
    ) -> None:
        graph = build_graph(model, client, model_id, decision_deadline_s=decision_deadline_s)
        self.graph: CompiledStateGraph[OrchestratorState, None, OrchestratorState, OrchestratorState] = (
            graph.compile(checkpointer=InMemorySaver())
        )

    async def run_turn(self, conversation_id: str, text: str, trace_id: str, today: date) -> TurnOutput:
        """Handle one user turn of a conversation and report what it did.

        `trace_id` files the retrieval agent's tool calls under this turn; `today` is the date the agents take
        as today. A turn it will not take raises TurnRefused before anything runs. A bug, or an error of the
        retrieval agent (RetrievalAgentError), is raised.
        """
        if not TRACE_ID.fullmatch(trace_id) or not TRACE_ID.fullmatch(conversation_id):
            raise TurnRefused(f"ids are {TRACE_ID_RULE}")
        try:
            RetrievalRequest(text=text, today=today)  # the text must be one the retrieval agent accepts
        except ValidationError as err:
            raise TurnRefused(f"the text cannot be sent: {err.errors()[0]['msg']}") from None
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
