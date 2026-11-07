"""Specification for the orchestrator: route_after_decide, then whole turns through run_turn.

The decision is made by the real ChatOllama client talking to a scripted Ollama server, so its replies and
errors are the ones that client produces. The retrieval agent is a fake client.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import date
from typing import Any, cast

import ollama
import pytest
from langgraph.checkpoint.serde import jsonplus

from agentic_earth_observation.evals.golden import Route
from agentic_earth_observation.orchestrator.client import RetrievalAgentError
from agentic_earth_observation.orchestrator.graph import (
    REQUEST_LIMIT,
    Orchestrator,
    OrchestratorState,
    TurnRefused,
    next_pending,
    route_after_decide,
)
from agentic_earth_observation.orchestrator.prompts import decide_system_prompt
from agentic_earth_observation.retrieval_agent.models import (
    Clarification,
    PreviousSearch,
    RetrievalRequest,
    RetrievalResult,
)
from agentic_earth_observation.retrieval_agent.prompts import retrieval_user_message
from tests.fakes import (
    CONNECTION_REFUSED,
    MID_STREAM,
    NEVER_FINISHES,
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
NO_PERIOD = ollama_tool_call("Decision", route="clarify", reason="missing_period")
ASKED = Clarification(request="Sentinel-2 scenes of Milan", asked="For which dates should I search?")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def langgraph_warns_again() -> None:
    """LangGraph warns once per process about a class of ours in a checkpoint: start with none warned."""
    jsonplus._warned_unregistered_types.clear()


def orchestrator(
    decisions: list[OllamaStep], client: FakeClient, *, decision_deadline_s: float = 150.0
) -> tuple[Orchestrator, FakeOllama]:
    ollama_server = FakeOllama(*decisions)
    model = ollama_server.model()
    return Orchestrator(model, client, MODEL_ID, decision_deadline_s=decision_deadline_s), ollama_server


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
        assert usage.model_seconds == pytest.approx(0.5)  # Ollama's own clock, as the fake reports it
        assert usage.load_seconds == pytest.approx(0.1)
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

    async def test_the_search_of_a_failed_turn_is_not_what_a_follow_up_continues(self) -> None:
        # The user was told the turn failed, so "and the week after?" cannot be about its search.
        failed = FOUND_MILAN.model_copy(update={"failure": "step_limit"})
        client = FakeClient(failed, FOUND_NAPLES)
        orch, _ = orchestrator([DELEGATE, DELEGATE], client)
        out = await orch.run_turn("conv-1", "Milan, March 2025", "turn-1", TODAY)
        assert out.failure == "step_limit"
        await orch.run_turn("conv-1", "And the week after?", "turn-2", TODAY)
        assert client.calls[1][0].previous_search is None

    @pytest.mark.usefixtures("langgraph_warns_again")
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

    async def test_a_decision_past_its_deadline_is_infrastructure(self) -> None:
        # The reply keeps streaming, so the model server's own timeout (a silence) never fires.
        client = FakeClient()
        orch, _ = orchestrator([NEVER_FINISHES], client, decision_deadline_s=0.3)
        started = time.perf_counter()
        async with asyncio.timeout(5):  # without the deadline the turn would never end: fail, do not hang
            out = await orch.run_turn("conv-1", "Something", "turn-1", TODAY)
        elapsed = time.perf_counter() - started
        assert (out.infra_error, out.failure, out.route, out.status) == (
            "the decision ran past its deadline (0.3 s)",
            None,
            Route.DECLINE,
            "declined",
        )
        assert 0.3 <= elapsed < 0.3 + 0.5  # enforced, and on time
        assert (out.decision_usage.model_calls, client.calls) == (1, [])
        assert 0.3 <= out.decision_usage.seconds <= elapsed

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
        with pytest.raises(TurnRefused, match=message):
            await orch.run_turn(conversation, text, trace, TODAY)
        assert (ollama_server.requests, client.calls) == ([], [])


class TestNextPending:
    """Which question stays open after a turn, for the next turn to answer."""

    def test_a_turn_that_asks_opens_a_question_about_its_request(self) -> None:
        opened = next_pending(
            None, "Sentinel-2 scenes of Milan", "For which dates?", asked=True, failed=False
        )
        assert opened == Clarification(request="Sentinel-2 scenes of Milan", asked="For which dates?")

    def test_a_second_question_builds_on_the_first(self) -> None:
        again = next_pending(ASKED, "with little cloud", "For which dates, again?", asked=True, failed=False)
        assert again == Clarification(
            request="Sentinel-2 scenes of Milan / with little cloud", asked="For which dates, again?"
        )

    @pytest.mark.parametrize("pending", [ASKED, None])
    def test_a_turn_that_does_not_ask_settles_it(self, pending: Clarification | None) -> None:
        assert next_pending(pending, "March 2025", "Found 4 scenes.", asked=False, failed=False) is None

    @pytest.mark.parametrize("pending", [ASKED, None])
    def test_a_failed_turn_leaves_it_as_it_was(self, pending: Clarification | None) -> None:
        assert next_pending(pending, "March 2025", "Sorry.", asked=True, failed=True) is pending
        assert next_pending(pending, "March 2025", "Sorry.", asked=False, failed=True) is pending

    def test_a_request_too_long_to_carry_starts_again_from_this_turn(self) -> None:
        fits = Clarification(request="x" * (REQUEST_LIMIT - 4), asked="?")
        kept = next_pending(fits, "y", "?", asked=True, failed=False)
        assert kept is not None
        assert kept.request == "x" * (REQUEST_LIMIT - 4) + " / y"  # exactly at the limit: kept
        too_long = Clarification(request="x" * (REQUEST_LIMIT - 3), asked="?")
        assert next_pending(too_long, "y", "?", asked=True, failed=False) == Clarification(
            request="y", asked="?"
        )

    def test_the_limit_is_the_one_the_request_accepts(self) -> None:
        limits = [
            m.max_length for m in Clarification.model_fields["request"].metadata if hasattr(m, "max_length")
        ]
        assert limits == [REQUEST_LIMIT]


class TestAnsweringAQuestion:
    async def test_an_answer_reaches_both_models_with_the_question_it_answers(self) -> None:
        client = FakeClient(FOUND_MILAN)
        orch, ollama_server = orchestrator([NO_PERIOD, DELEGATE], client)
        first = await orch.run_turn("conv-1", "Sentinel-2 scenes of Milan", "turn-1", TODAY)
        second = await orch.run_turn("conv-1", "March 2025", "turn-2", TODAY)
        question = Clarification(request="Sentinel-2 scenes of Milan", asked=first.answer)
        ((request, _),) = client.calls
        assert request == RetrievalRequest(text="March 2025", today=TODAY, clarifies=question)
        assert ollama_server.requests[1]["messages"][1]["content"] == retrieval_user_message(request)
        assert (first.status, second.route, second.status) == ("clarified", Route.DELEGATE, "ok")

    async def test_a_suggested_place_can_be_accepted(self) -> None:
        client = FakeClient(UNKNOWN, FOUND_NAPLES)
        orch, _ = orchestrator([DELEGATE, DELEGATE], client)
        first = await orch.run_turn(
            "conv-1", "Radar scenes of Napels, first week of September 2025", "turn-1", TODAY
        )
        await orch.run_turn("conv-1", "Yes", "turn-2", TODAY)
        assert "Naples" in first.answer  # the suggestion
        assert client.calls[0][0].clarifies is None
        assert client.calls[1][0].clarifies == Clarification(
            request="Radar scenes of Napels, first week of September 2025", asked=first.answer
        )

    async def test_questions_build_on_each_other(self) -> None:
        client = FakeClient(FOUND_MILAN)
        orch, _ = orchestrator([NO_PERIOD, NO_PERIOD, DELEGATE], client)
        await orch.run_turn("conv-1", "Sentinel-2 scenes of Milan", "turn-1", TODAY)
        second = await orch.run_turn("conv-1", "with little cloud", "turn-2", TODAY)
        await orch.run_turn("conv-1", "March 2025", "turn-3", TODAY)
        assert client.calls[0][0].clarifies == Clarification(
            request="Sentinel-2 scenes of Milan / with little cloud", asked=second.answer
        )

    @pytest.mark.parametrize("settling", [DELEGATE, DECLINE])
    async def test_a_turn_that_does_not_ask_settles_the_question(self, settling: OllamaStep) -> None:
        client = FakeClient(FOUND_MILAN, FOUND_MILAN)
        orch, ollama_server = orchestrator([NO_PERIOD, settling, DELEGATE], client)
        for i, text in enumerate(["Sentinel-2 scenes of Milan", "March 2025", "And April?"]):
            await orch.run_turn("conv-1", text, f"turn-{i}", TODAY)
        assert client.calls[-1][0].clarifies is None
        assert "THE SERVICE ASKED" not in ollama_server.requests[2]["messages"][1]["content"]

    async def test_a_failed_turn_leaves_the_question_open(self) -> None:
        client = FakeClient(FOUND_MILAN)
        orch, _ = orchestrator([NO_PERIOD, ollama_text("Hmm."), DELEGATE], client)
        first = await orch.run_turn("conv-1", "Sentinel-2 scenes of Milan", "turn-1", TODAY)
        failed = await orch.run_turn("conv-1", "Erm", "turn-2", TODAY)
        await orch.run_turn("conv-1", "March 2025", "turn-3", TODAY)
        assert failed.failure == "parse_error"
        assert client.calls[0][0].clarifies == Clarification(
            request="Sentinel-2 scenes of Milan", asked=first.answer
        )

    async def test_infrastructure_leaves_the_question_open(self) -> None:
        client = FakeClient(FOUND_MILAN)
        orch, _ = orchestrator([NO_PERIOD, OllamaError(503, "server busy"), DELEGATE], client)
        first = await orch.run_turn("conv-1", "Sentinel-2 scenes of Milan", "turn-1", TODAY)
        busy = await orch.run_turn("conv-1", "March 2025", "turn-2", TODAY)
        await orch.run_turn("conv-1", "March 2025", "turn-3", TODAY)
        assert busy.infra_error is not None
        assert client.calls[0][0].clarifies == Clarification(
            request="Sentinel-2 scenes of Milan", asked=first.answer
        )

    async def test_a_question_belongs_to_its_conversation(self) -> None:
        client = FakeClient(FOUND_MILAN)
        orch, _ = orchestrator([NO_PERIOD, DELEGATE], client)
        await orch.run_turn("conv-a", "Sentinel-2 scenes of Milan", "turn-1", TODAY)
        await orch.run_turn("conv-b", "Sentinel-2 scenes of Milan, March 2025", "turn-2", TODAY)
        assert client.calls[0][0].clarifies is None

    @pytest.mark.usefixtures("langgraph_warns_again")
    async def test_the_memory_stays_plain_with_a_question_in_it(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG)
        orch, _ = orchestrator([NO_PERIOD, DELEGATE], FakeClient(FOUND_MILAN))
        await orch.run_turn("conv-1", "Sentinel-2 scenes of Milan", "turn-1", TODAY)
        await orch.run_turn("conv-1", "March 2025", "turn-2", TODAY)
        assert not [r for r in caplog.records if "eserializ" in r.getMessage() or "msgpack" in r.getMessage()]
