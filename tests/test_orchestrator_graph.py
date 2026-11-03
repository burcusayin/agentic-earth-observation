"""Specification for the orchestrator: route_after_decide, then whole turns through run_turn.

The decision is made by the real ChatOllama client talking to a scripted Ollama server, so its replies and
errors are the ones that client produces. The retrieval agent is a fake client.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any, cast

import ollama
import pytest

from agentic_earth_observation.evals.golden import Route
from agentic_earth_observation.orchestrator.client import RetrievalAgentError
from agentic_earth_observation.orchestrator.graph import Orchestrator, OrchestratorState, route_after_decide
from agentic_earth_observation.orchestrator.prompts import decide_system_prompt
from agentic_earth_observation.retrieval_agent.models import PreviousSearch, RetrievalRequest, RetrievalResult
from agentic_earth_observation.retrieval_agent.prompts import retrieval_user_message
from tests.fakes import (
    CONNECTION_REFUSED,
    MID_STREAM,
    SCENE,
    FakeClient,
    FakeOllama,
    OllamaError,
    OllamaStep,
    ollama_text,
    ollama_tool_call,
)

pytestmark = pytest.mark.anyio

TODAY = date(2025, 12, 15)
MODEL_ID = "qwen3.5:9b"


def search(
    place: str, bbox: list[float], start: str, end: str, product: str = "sentinel-1-grd"
) -> PreviousSearch:
    fields = {"place": place, "product": product, "bbox": bbox, "start_date": start, "end_date": end}
    return PreviousSearch.model_validate_json(json.dumps(fields))


NAPLES_WEEK = search("Naples", [14.1321, 40.7912, 14.3538, 40.9161], "2025-09-01", "2025-09-07")
MILAN_MARCH = search(
    "Milan", [9.0409, 45.3867, 9.2781, 45.5358], "2025-03-01", "2025-03-31", "sentinel-2-l2a"
)
FOUND_NAPLES = RetrievalResult(status="ok", scenes=[SCENE], search=NAPLES_WEEK)
FOUND_MILAN = RetrievalResult(status="ok", scenes=[SCENE], search=MILAN_MARCH)
UNKNOWN = RetrievalResult(status="unknown_place", suggestions=["Naples"])

DELEGATE = ollama_tool_call("Decision", route="delegate", reason="search")
DECLINE = ollama_tool_call("Decision", route="decline", reason="out_of_scope")
NO_PLACE = ollama_tool_call("Decision", route="clarify", reason="missing_place")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def orchestrator(decisions: list[OllamaStep], client: FakeClient) -> tuple[Orchestrator, FakeOllama]:
    ollama_server = FakeOllama(*decisions)
    return Orchestrator(ollama_server.model(), client, MODEL_ID), ollama_server


def state(**fields: Any) -> OrchestratorState:  # noqa: ANN401 - the state's fields
    base: dict[str, Any] = {
        "text": "x",
        "today": TODAY,
        "trace_id": "t",
        "route": None,
        "reason": None,
        "failure": None,
        "failure_detail": None,
        "infra_error": None,
        "decision_usage": "{}",
        "retrieval": None,
        "reply": "",
        "cited_ids": [],
    }
    return cast(OrchestratorState, base | fields)


class TestRouteAfterDecide:
    @pytest.mark.parametrize(
        ("fields", "next_node"),
        [
            ({"route": "delegate", "reason": "search"}, "delegate"),
            ({"route": "decline", "reason": "out_of_scope"}, "reply"),
            ({"route": "clarify", "reason": "missing_place"}, "reply"),
            ({"route": "clarify", "reason": "missing_period"}, "reply"),
            ({"failure": "parse_error", "failure_detail": "the model answered without a Decision"}, "reply"),
            ({"infra_error": "ConnectError: All connection attempts failed"}, "reply"),
        ],
    )
    def test_the_next_node_follows_the_decision(self, fields: dict[str, Any], next_node: str) -> None:
        assert route_after_decide(state(**fields)) == next_node

    def test_a_state_without_a_decision_or_a_failure_cannot_happen(self) -> None:
        with pytest.raises(ValueError):  # noqa: PT011 - any message: the point is that it is refused
            route_after_decide(state())


class TestTurns:
    async def test_a_search_turn_delegates_and_replies_with_the_scenes(self) -> None:
        client = FakeClient(FOUND_NAPLES)
        orch, _ = orchestrator([DELEGATE], client)
        out = await orch.run_turn(
            "conv-1", "Radar over Naples, first week of September 2025", "turn-1", TODAY
        )
        assert (out.route, out.status, out.retrieval) == (Route.DELEGATE, "ok", FOUND_NAPLES)
        assert out.cited_ids == [SCENE.id]
        assert SCENE.id in out.answer
        assert (out.failure, out.infra_error, out.trace_id) == (None, None, "turn-1")
        usage = out.decision_usage
        assert (usage.model, usage.model_calls, usage.input_tokens, usage.output_tokens) == (
            MODEL_ID,
            1,
            100,
            10,
        )
        ((request, trace_id),) = client.calls
        assert request == RetrievalRequest(
            text="Radar over Naples, first week of September 2025", today=TODAY
        )
        assert trace_id == "turn-1"

    async def test_the_decision_reads_the_dated_prompt_and_the_request_and_is_offered_one_tool(self) -> None:
        orch, ollama_server = orchestrator([DECLINE], FakeClient())
        await orch.run_turn("conv-1", "What will the weather be like?", "turn-1", TODAY)
        ((sent,),) = [ollama_server.requests]
        system, user = sent["messages"]
        assert (system["role"], system["content"]) == ("system", decide_system_prompt(TODAY))
        request = RetrievalRequest(text="What will the weather be like?", today=TODAY)
        assert (user["role"], user["content"]) == ("user", retrieval_user_message(request))
        assert [tool["function"]["name"] for tool in sent["tools"]] == ["Decision"]

    async def test_a_follow_up_continues_the_previous_search(self) -> None:
        client = FakeClient(FOUND_NAPLES, FOUND_NAPLES)
        orch, ollama_server = orchestrator([DELEGATE, DELEGATE], client)
        await orch.run_turn("conv-1", "Radar over Naples, first week of September 2025", "turn-1", TODAY)
        await orch.run_turn("conv-1", "And the week after that?", "turn-2", TODAY)
        assert client.calls[1][0].previous_search == NAPLES_WEEK
        assert "PREVIOUS SEARCH: place=Naples" in ollama_server.requests[1]["messages"][1]["content"]

    async def test_the_previous_search_outlives_turns_without_a_search_until_a_new_one(self) -> None:
        client = FakeClient(FOUND_NAPLES, UNKNOWN, FOUND_MILAN, FOUND_MILAN)
        orch, _ = orchestrator([DELEGATE, DECLINE, DELEGATE, DELEGATE, DELEGATE], client)
        for i, text in enumerate(
            ["Naples radar", "The weather?", "Somewhere unknown", "Milan in March", "And April?"]
        ):
            await orch.run_turn("conv-1", text, f"turn-{i}", TODAY)
        previous = [request.previous_search for request, _ in client.calls]
        assert previous == [None, NAPLES_WEEK, NAPLES_WEEK, MILAN_MARCH]

    async def test_conversations_are_kept_apart(self) -> None:
        client = FakeClient(FOUND_NAPLES, FOUND_MILAN)
        orch, _ = orchestrator([DELEGATE, DELEGATE], client)
        await orch.run_turn("conv-a", "Naples radar", "turn-1", TODAY)
        await orch.run_turn("conv-b", "Milan in March", "turn-2", TODAY)
        assert client.calls[1][0].previous_search is None

    async def test_a_declined_request_never_reaches_the_agent(self) -> None:
        client = FakeClient()
        orch, _ = orchestrator([DECLINE], client)
        out = await orch.run_turn("conv-1", "What will the weather be like?", "turn-1", TODAY)
        assert (out.route, out.status, out.retrieval, out.cited_ids) == (Route.DECLINE, "declined", None, [])
        assert client.calls == []

    async def test_a_request_without_a_place_is_clarified_without_searching(self) -> None:
        client = FakeClient()
        orch, _ = orchestrator([NO_PLACE], client)
        out = await orch.run_turn("conv-1", "Show me some radar scenes", "turn-1", TODAY)
        assert (out.route, out.status, out.retrieval) == (Route.CLARIFY, "clarified", None)
        assert client.calls == []

    async def test_an_unknown_place_is_clarified_after_delegating(self) -> None:
        orch, _ = orchestrator([DELEGATE], FakeClient(UNKNOWN))
        out = await orch.run_turn("conv-1", "Scenes of Atlantis", "turn-1", TODAY)
        assert (out.route, out.status, out.cited_ids) == (Route.DELEGATE, "clarified", [])
        assert "Naples" in out.answer  # the suggestion

    async def test_each_turn_starts_afresh(self) -> None:
        orch, _ = orchestrator([ollama_text("Hmm."), DECLINE], FakeClient())
        first = await orch.run_turn("conv-1", "Something", "turn-1", TODAY)
        second = await orch.run_turn("conv-1", "The weather?", "turn-2", TODAY)
        assert first.failure == "parse_error"
        assert (second.failure, second.failure_detail, second.route) == (None, None, Route.DECLINE)

    async def test_the_conversation_memory_holds_only_plain_values(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # LangGraph warns when it restores a class of ours from a checkpoint, and silently changes one that is
        # not registered; the state keeps JSON text instead.
        caplog.set_level(logging.DEBUG)
        orch, _ = orchestrator([DELEGATE, DELEGATE], FakeClient(FOUND_NAPLES, FOUND_NAPLES))
        await orch.run_turn("conv-1", "Naples radar", "turn-1", TODAY)
        await orch.run_turn("conv-1", "And the week after?", "turn-2", TODAY)
        assert not [r for r in caplog.records if "eserializ" in r.getMessage() or "msgpack" in r.getMessage()]


class TestDecisionFailures:
    """A reply without a valid Decision is the model's failure: the turn declines without searching."""

    @pytest.mark.parametrize(
        ("step", "detail"),
        [
            (ollama_text("I would decline this."), "the model answered without a Decision"),
            (ollama_tool_call("Decision", route="delegate", reason="out_of_scope"), "ValidationError: "),
            (ollama_tool_call("Decision", route="delegate"), "ValidationError: "),
            (
                OllamaError(500, "XML syntax error on line 1: unexpected EOF"),
                "XML syntax error on line 1: unexpected EOF",
            ),
            (OllamaError(MID_STREAM, "tool 'Decide' not found"), "tool 'Decide' not found"),
        ],
    )
    async def test_the_model_fails_the_turn(self, step: OllamaStep, detail: str) -> None:
        client = FakeClient()
        orch, _ = orchestrator([step], client)
        out = await orch.run_turn("conv-1", "Something", "turn-1", TODAY)
        assert (out.failure, out.route, out.status, out.cited_ids) == (
            "parse_error",
            Route.DECLINE,
            "declined",
            [],
        )
        assert out.failure_detail is not None
        assert out.failure_detail.startswith(detail)
        assert (out.infra_error, out.decision_usage.model_calls) == (None, 1)
        assert client.calls == []

    @pytest.mark.parametrize(
        ("step", "reported"),
        [
            (OllamaError(503, "server busy"), "ResponseError: server busy (status code: 503)"),
            (CONNECTION_REFUSED, "ConnectError: All connection attempts failed"),
        ],
    )
    async def test_a_model_server_out_of_reach_is_infrastructure(
        self, step: OllamaStep, reported: str
    ) -> None:
        client = FakeClient()
        orch, _ = orchestrator([step], client)
        out = await orch.run_turn("conv-1", "Something", "turn-1", TODAY)
        assert (out.infra_error, out.failure, out.route, out.status) == (
            reported,
            None,
            Route.DECLINE,
            "declined",
        )
        assert client.calls == []

    async def test_a_model_the_server_does_not_have_is_raised(self) -> None:
        orch, _ = orchestrator([OllamaError(404, 'model "qwen3.5:9b" not found')], FakeClient())
        with pytest.raises(ollama.ResponseError, match="not found"):
            await orch.run_turn("conv-1", "Something", "turn-1", TODAY)


class TestRetrievalOutcomes:
    async def test_the_agents_infrastructure_failure_is_the_turns(self) -> None:
        down = RetrievalResult(status="not_searched", infra_error="A2A ReadTimeout")
        orch, _ = orchestrator([DELEGATE], FakeClient(down))
        out = await orch.run_turn("conv-1", "Naples radar", "turn-1", TODAY)
        assert (out.infra_error, out.route, out.status, out.cited_ids) == (
            "A2A ReadTimeout",
            Route.DELEGATE,
            "declined",
            [],
        )

    async def test_the_agents_model_failure_is_the_turns(self) -> None:
        failed = RetrievalResult(
            status="not_searched", failure="malformed_tool_call", failure_detail="XML syntax error"
        )
        orch, _ = orchestrator([DELEGATE], FakeClient(failed))
        out = await orch.run_turn("conv-1", "Naples radar", "turn-1", TODAY)
        assert (out.failure, out.failure_detail, out.infra_error) == (
            "malformed_tool_call",
            "XML syntax error",
            None,
        )

    async def test_an_error_of_the_agent_is_raised(self) -> None:
        bug = RetrievalAgentError("the retrieval agent ended turn turn-1 as TASK_STATE_FAILED: ...")
        orch, _ = orchestrator([DELEGATE], FakeClient(bug))
        with pytest.raises(RetrievalAgentError, match="TASK_STATE_FAILED"):
            await orch.run_turn("conv-1", "Naples radar", "turn-1", TODAY)


class TestBoundary:
    @pytest.mark.parametrize(
        ("conversation", "text", "trace", "message"),
        [
            ("conv-1", "Naples radar", "turn 1", "ids are 1 to 64 letters"),
            ("conv 1", "Naples radar", "turn-1", "ids are 1 to 64 letters"),
            ("conv-1", "   ", "turn-1", "the text cannot be sent: String should have at least 1 character"),
            (
                "conv-1",
                "x" * 2001,
                "turn-1",
                "the text cannot be sent: String should have at most 2000 characters",
            ),
        ],
    )
    async def test_a_turn_that_cannot_be_handled_is_refused_before_anything_runs(
        self, conversation: str, text: str, trace: str, message: str
    ) -> None:
        client = FakeClient()
        orch, ollama_server = orchestrator([DELEGATE], client)
        with pytest.raises(ValueError, match=message):
            await orch.run_turn(conversation, text, trace, TODAY)
        assert (ollama_server.requests, client.calls) == ([], [])
