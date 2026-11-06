"""Specification for what one turn of an evaluation run counts as (evals/outcome.py).

The first classes give each rule its cases, on outputs and tool-call records written by hand. TestRealTurns
then runs whole turns: a scripted model, the real retrieval loop, the real MCP server in-process with its real
tool-call log, and the real orchestrator around them. No ordinary turn may come out as a disagreement between
the agents and the log, or every task of a run would be counted as a bug. No network, and no model.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from mcp.shared._httpx_utils import McpHttpClientFactory

from agentic_earth_observation.catalog.gazetteer import Gazetteer
from agentic_earth_observation.catalog.stac import SceneSummary
from agentic_earth_observation.catalog_mcp.server import build_server
from agentic_earth_observation.catalog_mcp.tool_log import (
    ErrorKind,
    ToolCallLog,
    ToolCallRecord,
    read_tool_calls,
)
from agentic_earth_observation.config import Settings
from agentic_earth_observation.domain import SearchRequest
from agentic_earth_observation.evals.golden import GOLDEN_DIR, Outcome, Route, load_tasks
from agentic_earth_observation.evals.outcome import task_passed, turn_outcome, turn_trace
from agentic_earth_observation.evals.scoring import ToolCall, TurnScore, score_turn
from agentic_earth_observation.orchestrator.graph import Orchestrator
from agentic_earth_observation.orchestrator.models import TurnOutput, scored_status
from agentic_earth_observation.retrieval_agent.agent import run_retrieval
from agentic_earth_observation.retrieval_agent.models import (
    PreviousSearch,
    RetrievalRequest,
    RetrievalResult,
)
from tests.fakes import (
    SCENE,
    FakeCatalog,
    FakeOllama,
    OllamaStep,
    ScriptedModel,
    ollama_tool_call,
    tool_call,
)

pytestmark = pytest.mark.anyio

TODAY = date(2025, 12, 15)
MODEL_ID = "qwen3.5:9b"
TRACE = "0a1b2c3d-t01-a1-u1"
GAZETTEER = Gazetteer.load()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def box(place: str) -> list[float]:
    found = GAZETTEER.resolve(place)
    assert found is not None
    return found.bbox.as_list()


MILAN_MARCH: dict[str, Any] = {
    "product": "sentinel-2-l2a",
    "bbox": box("Milan"),
    "start_date": "2025-03-01",
    "end_date": "2025-03-31",
    "max_cloud_cover": 20,
}
OTHER_SCENE = SCENE.model_copy(update={"id": "S2B_T32TNR_20250312T101729_L2A"})


def the_search(arguments: dict[str, Any], place: str | None = "Milan") -> PreviousSearch:
    """The agent's own account of a search with these arguments, built as result_from_turn builds it."""
    return PreviousSearch.model_validate_json(json.dumps({**arguments, "place": place}))


# The retrieval agent's results, one per status.
FOUND = RetrievalResult(status="ok", scenes=[SCENE], search=the_search(MILAN_MARCH))
FOUND_TWO = RetrievalResult(status="ok", scenes=[SCENE, OTHER_SCENE], search=the_search(MILAN_MARCH))
NONE_FOUND = RetrievalResult(status="empty", search=the_search(MILAN_MARCH))
UNKNOWN_PLACE = RetrievalResult(status="unknown_place", suggestions=["Milan"])
TOOL_ERROR = RetrievalResult(status="error")
NOT_SEARCHED = RetrievalResult(status="not_searched")


def delegated(result: RetrievalResult) -> TurnOutput:
    """A delegated turn that ended with this result, as run_turn reports it."""
    return TurnOutput(
        trace_id=TRACE,
        route=Route.DELEGATE,
        status=scored_status("delegate", result),
        answer="(the reply)",
        cited_ids=[scene.id for scene in result.scenes],
        failure=result.failure,
        failure_detail=result.failure_detail,
        infra_error=result.infra_error,
        retrieval=result,
    )


def direct(route: Route = Route.DECLINE, **fields: Any) -> TurnOutput:  # noqa: ANN401 - fields of the output
    """A turn the orchestrator answered itself, without the retrieval agent."""
    status: Outcome = "clarified" if route is Route.CLARIFY else "declined"
    return TurnOutput(trace_id=TRACE, route=route, status=status, answer="(the reply)", **fields)


def call(
    tool: str,
    request_id: int = 1,
    /,
    *,
    ok: bool = True,
    kind: ErrorKind | None = None,
    ids: Sequence[str] = (),
    trace_id: str | None = TRACE,
    **arguments: Any,  # noqa: ANN401 - the tool's arguments
) -> ToolCallRecord:
    """One call as the MCP server records it. Positional-only, so a tool argument may be called `name`."""
    return ToolCallRecord(
        trace_id=trace_id,
        request_id=request_id,
        tool=tool,
        arguments=arguments,
        ok=ok,
        error_kind=kind,
        returned_ids=list(ids),
    )


def failed(
    tool: str,
    request_id: int,
    kind: ErrorKind,
    /,
    **arguments: Any,  # noqa: ANN401 - the tool's arguments
) -> ToolCallRecord:
    return call(tool, request_id, ok=False, kind=kind, **arguments)


def later(record: ToolCallRecord, request_id: int) -> ToolCallRecord:
    return record.model_copy(update={"request_id": request_id})


RESOLVED = call("resolve_place", 1, name="Milan")
SEARCHED = call("search_scenes", 2, ids=[SCENE.id], **MILAN_MARCH)
SEARCHED_NOTHING = call("search_scenes", 2, **MILAN_MARCH)
REFUSED = failed("search_scenes", 2, "invalid_arguments", **{**MILAN_MARCH, "start_date": "March 2025"})
CATALOGUE_FAILED = failed("search_scenes", 2, "catalogue", **MILAN_MARCH)
CANCELLED = failed("search_scenes", 2, "cancelled", **MILAN_MARCH)
UNEXPLAINED = failed("search_scenes", 2, "other", **MILAN_MARCH)


class TestTrace:
    """What the scoring reads: the orchestrator's route, status and cited ids, and the server's calls."""

    def test_the_route_the_status_and_the_cited_ids_come_from_the_output(self) -> None:
        trace = turn_trace(delegated(FOUND), [RESOLVED, SEARCHED])
        assert (trace.route, trace.status, trace.cited_ids) == (Route.DELEGATE, "ok", [SCENE.id])
        trace = turn_trace(direct(Route.CLARIFY), [])
        assert (trace.route, trace.status, trace.cited_ids, trace.calls) == (
            Route.CLARIFY,
            "clarified",
            [],
            [],
        )

    def test_the_calls_are_the_logs_in_the_order_given_refused_ones_included(self) -> None:
        trace = turn_trace(delegated(FOUND), [RESOLVED, REFUSED, later(SEARCHED, 3)])
        assert trace.calls == [
            ToolCall(name="resolve_place", arguments={"name": "Milan"}),
            ToolCall(name="search_scenes", arguments=REFUSED.arguments, ok=False),
            ToolCall(name="search_scenes", arguments=MILAN_MARCH, returned_ids=[SCENE.id]),
        ]

    def test_no_call_is_taken_from_the_agents_own_result(self) -> None:
        # The agent says it searched; the trace holds what the server recorded, which here is nothing.
        assert turn_trace(delegated(FOUND), []).calls == []


class TestInfrastructure:
    """Rules 2 and 3: a failure outside the model's control. The task is tried again and left out."""

    def test_the_decision_could_not_reach_the_model_server(self) -> None:
        output = direct(infra_error="ConnectError: All connection attempts failed")
        assert turn_outcome(output, []) == ("infrastructure", "ConnectError: All connection attempts failed")

    def test_the_retrieval_agent_reports_an_infrastructure_failure(self) -> None:
        lost = RetrievalResult(status="not_searched", infra_error="the turn ran past its deadline (240 s)")
        assert turn_outcome(delegated(lost), [RESOLVED]) == (
            "infrastructure",
            "the turn ran past its deadline (240 s)",
        )

    def test_a_failure_reported_without_words_still_gets_a_reason(self) -> None:
        outcome, reason = turn_outcome(direct(infra_error=""), [])
        assert outcome == "infrastructure"
        assert reason

    def test_the_catalogue_failed_a_call_though_the_agent_reports_only_an_error(self) -> None:
        outcome, reason = turn_outcome(delegated(TOOL_ERROR), [RESOLVED, CATALOGUE_FAILED])
        assert outcome == "infrastructure"
        assert reason is not None
        assert "catalogue" in reason
        assert "search_scenes" in reason

    def test_a_catalogue_failure_counts_even_when_a_later_search_succeeded(self) -> None:
        records = [RESOLVED, CATALOGUE_FAILED, later(SEARCHED, 3)]
        assert turn_outcome(delegated(FOUND), records)[0] == "infrastructure"

    @pytest.mark.parametrize(
        "records",
        [
            pytest.param([RESOLVED, SEARCHED], id="a search the placeholder result does not report"),
            pytest.param([RESOLVED, CANCELLED], id="a call cut off when the client gave up"),
            pytest.param([RESOLVED, UNEXPLAINED], id="a call that failed for an unknown reason"),
        ],
    )
    def test_infrastructure_is_decided_before_the_log_is_compared(
        self, records: list[ToolCallRecord]
    ) -> None:
        # After an infrastructure failure the agent's result is a placeholder: it reports none of its calls.
        lost = RetrievalResult(status="not_searched", infra_error="A2A ReadTimeout")
        assert turn_outcome(delegated(lost), records) == ("infrastructure", "A2A ReadTimeout")

    def test_a_catalogue_failure_is_decided_before_the_log_is_compared(self) -> None:
        records = [RESOLVED, CATALOGUE_FAILED, later(UNEXPLAINED, 3)]
        assert turn_outcome(delegated(TOOL_ERROR), records)[0] == "infrastructure"


def with_failure(result: RetrievalResult) -> RetrievalResult:
    return result.model_copy(update={"failure": "step_limit"})


AGREEING = [
    pytest.param(direct(), [], id="declined, and no call"),
    pytest.param(direct(Route.CLARIFY), [], id="clarified, and no call"),
    pytest.param(direct(failure="parse_error"), [], id="no decision, and no call"),
    pytest.param(delegated(FOUND), [RESOLVED, SEARCHED], id="scenes found"),
    pytest.param(delegated(FOUND), [SEARCHED], id="scenes found without a place lookup"),
    pytest.param(
        delegated(FOUND), [RESOLVED, REFUSED, later(SEARCHED, 3)], id="found after a refused search"
    ),
    pytest.param(
        delegated(FOUND),
        [RESOLVED, call("search_scenes", 2, ids=[OTHER_SCENE.id], **MILAN_MARCH), later(SEARCHED, 3)],
        id="the last of two searches",
    ),
    pytest.param(
        delegated(FOUND),
        [RESOLVED, SEARCHED, failed("search_scenes", 3, "invalid_arguments", product="landsat")],
        id="found, then a refused search",
    ),
    pytest.param(
        delegated(FOUND_TWO),
        [call("search_scenes", ids=[SCENE.id, OTHER_SCENE.id], **MILAN_MARCH)],
        id="two scenes, in the order returned",
    ),
    pytest.param(delegated(NONE_FOUND), [RESOLVED, SEARCHED_NOTHING], id="nothing found"),
    pytest.param(
        delegated(RetrievalResult(status="ok", scenes=[SCENE], search=the_search(MILAN_MARCH, None))),
        [SEARCHED],
        id="the place name is the agent's own and is not compared",
    ),
    pytest.param(delegated(with_failure(FOUND)), [RESOLVED, SEARCHED], id="found, then the model failed"),
    pytest.param(delegated(UNKNOWN_PLACE), [call("resolve_place", name="Mil")], id="an unknown place"),
    pytest.param(
        delegated(UNKNOWN_PLACE),
        [call("resolve_place", name="Mil"), failed("resolve_place", 2, "invalid_arguments", place="Milan")],
        id="an unknown place, then a refused lookup",
    ),
    pytest.param(delegated(TOOL_ERROR), [RESOLVED, REFUSED], id="a refused search the model gave up on"),
    pytest.param(
        delegated(TOOL_ERROR),
        [failed("resolve_place", 1, "invalid_arguments", place="Milan")],
        id="a refused place lookup",
    ),
    pytest.param(delegated(NOT_SEARCHED), [], id="delegated, and the model called no tool"),
    pytest.param(delegated(NOT_SEARCHED), [call("list_products")], id="only the product list"),
    pytest.param(delegated(NOT_SEARCHED), [RESOLVED], id="a place found and never searched"),
    pytest.param(delegated(with_failure(NOT_SEARCHED)), [RESOLVED], id="the model failed before a search"),
]

DISAGREEING = [
    pytest.param(direct(), [RESOLVED], "not delegated", id="declined, yet a call"),
    pytest.param(direct(Route.CLARIFY), [RESOLVED, SEARCHED], "not delegated", id="clarified, yet calls"),
    pytest.param(delegated(FOUND), [], "no successful search_scenes", id="found, yet no call"),
    pytest.param(
        delegated(FOUND), [RESOLVED, REFUSED], "no successful search_scenes", id="found, yet refused"
    ),
    pytest.param(delegated(NONE_FOUND), [RESOLVED], "no successful search_scenes", id="empty, yet no search"),
    pytest.param(
        delegated(FOUND),
        [RESOLVED, call("search_scenes", 2, ids=[OTHER_SCENE.id], **MILAN_MARCH)],
        "scenes",
        id="found, yet the search returned another scene",
    ),
    pytest.param(
        delegated(FOUND_TWO),
        [call("search_scenes", ids=[OTHER_SCENE.id, SCENE.id], **MILAN_MARCH)],
        "scenes",
        id="found, yet the search returned them in another order",
    ),
    pytest.param(delegated(FOUND), [RESOLVED, SEARCHED_NOTHING], "scenes", id="found, yet nothing returned"),
    pytest.param(delegated(NONE_FOUND), [RESOLVED, SEARCHED], "scenes", id="empty, yet a scene returned"),
    pytest.param(
        delegated(FOUND),
        [RESOLVED, SEARCHED, call("search_scenes", 3, ids=[OTHER_SCENE.id], **MILAN_MARCH)],
        "scenes",
        id="the agent reports a search that is not the last",
    ),
    pytest.param(
        delegated(FOUND),
        [call("search_scenes", ids=[SCENE.id], **{**MILAN_MARCH, "end_date": "2025-03-30"})],
        "arguments",
        id="found, yet with another end date",
    ),
    pytest.param(
        delegated(FOUND),
        [call("search_scenes", ids=[SCENE.id], **{**MILAN_MARCH, "limit": 5})],
        "arguments",
        id="found, yet with a limit the agent does not report",
    ),
    pytest.param(
        delegated(FOUND),
        [call("search_scenes", ids=[SCENE.id], **{**MILAN_MARCH, "bbox": box("Naples")})],
        "arguments",
        id="found, yet over another box",
    ),
    pytest.param(
        delegated(FOUND),
        [call("search_scenes", ids=[SCENE.id], product="sentinel-2-l2a")],
        "arguments",
        id="found, yet the logged arguments are not a search",
    ),
    pytest.param(
        delegated(UNKNOWN_PLACE),
        [RESOLVED, SEARCHED],
        "a successful search_scenes",
        id="unknown, yet searched",
    ),
    pytest.param(
        delegated(TOOL_ERROR),
        [REFUSED, later(SEARCHED, 3)],
        "a successful search_scenes",
        id="error, yet found",
    ),
    pytest.param(
        delegated(NOT_SEARCHED),
        [RESOLVED, SEARCHED_NOTHING],
        "a successful search_scenes",
        id="none, yet searched",
    ),
    pytest.param(delegated(UNKNOWN_PLACE), [], "resolve_place", id="unknown, yet no call"),
    pytest.param(
        delegated(UNKNOWN_PLACE), [call("list_products")], "resolve_place", id="unknown, yet no lookup"
    ),
    pytest.param(
        delegated(UNKNOWN_PLACE),
        [failed("resolve_place", 1, "invalid_arguments", place="Mil")],
        "resolve_place",
        id="unknown, yet the lookup was refused",
    ),
    pytest.param(delegated(TOOL_ERROR), [], "no failed call", id="error, yet no call"),
    pytest.param(delegated(TOOL_ERROR), [RESOLVED], "no failed call", id="error, yet every call succeeded"),
    pytest.param(
        delegated(FOUND), [RESOLVED, UNEXPLAINED, later(SEARCHED, 3)], "other", id="an unexplained failure"
    ),
    pytest.param(
        delegated(TOOL_ERROR), [UNEXPLAINED], "other", id="an unexplained failure, reported as error"
    ),
    pytest.param(
        delegated(FOUND), [RESOLVED, CANCELLED, later(SEARCHED, 3)], "cancelled", id="a cancelled call"
    ),
    pytest.param(delegated(TOOL_ERROR), [CANCELLED], "cancelled", id="a cancelled call, reported as error"),
]


class TestTheAgentsAndTheLog:
    """Rules 4 and 5: the orchestrator's output against the server's own records of the turn."""

    @pytest.mark.parametrize(("output", "records"), AGREEING)
    def test_a_turn_whose_two_accounts_agree_is_scored(
        self, output: TurnOutput, records: list[ToolCallRecord]
    ) -> None:
        assert turn_outcome(output, records) == ("scored", None)

    @pytest.mark.parametrize(("output", "records", "says"), DISAGREEING)
    def test_a_turn_whose_two_accounts_disagree_is_a_bug(
        self, output: TurnOutput, records: list[ToolCallRecord], says: str
    ) -> None:
        outcome, reason = turn_outcome(output, records)
        assert outcome == "errored"
        assert reason is not None
        assert says in reason

    def test_numbers_are_compared_as_numbers(self) -> None:
        # The log keeps 20 as the model sent it; the agent's search holds 20.0.
        assert SEARCHED.arguments["max_cloud_cover"] == 20
        assert isinstance(SEARCHED.arguments["max_cloud_cover"], int)
        assert turn_outcome(delegated(FOUND), [SEARCHED]) == ("scored", None)


class TestRecordsOfTheTurn:
    @pytest.mark.parametrize("trace_id", ["0a1b2c3d-t01-a1-u2", None])
    def test_records_filed_under_another_trace_id_are_refused(self, trace_id: str | None) -> None:
        # The caller read the wrong calls: a bug in the runner, not a fact about the turn.
        stray = call("resolve_place", trace_id=trace_id, name="Milan")
        with pytest.raises(ValueError, match="trace id"):
            turn_outcome(delegated(NOT_SEARCHED), [RESOLVED, stray])
        with pytest.raises(ValueError, match="trace id"):
            turn_trace(delegated(NOT_SEARCHED), [RESOLVED, stray])


PASS = TurnScore(route=True, tools=True, arguments=None, first_arguments=None, grounded=True, outcome=True)
FAIL = PASS.model_copy(update={"outcome": False})


class TestTaskPassed:
    """A task passes when every turn's score passes and no turn has a model failure."""

    def test_every_turn_passed(self) -> None:
        assert task_passed([PASS], [delegated(FOUND)]) is True
        assert task_passed([PASS, PASS], [delegated(FOUND), direct()]) is True

    @pytest.mark.parametrize("scores", [[FAIL], [PASS, FAIL], [FAIL, PASS], [PASS, PASS, FAIL]])
    def test_one_failed_turn_fails_the_task(self, scores: list[TurnScore]) -> None:
        assert task_passed(scores, [delegated(FOUND)] * len(scores)) is False

    @pytest.mark.parametrize(
        "outputs",
        [
            pytest.param([delegated(with_failure(FOUND))], id="in the retrieval"),
            pytest.param([direct(failure="parse_error")], id="in the decision"),
            pytest.param([delegated(FOUND), delegated(with_failure(FOUND))], id="in the last turn"),
            pytest.param([delegated(with_failure(FOUND)), delegated(FOUND)], id="in the first turn"),
        ],
    )
    def test_a_model_failure_fails_the_task_whatever_the_scores(self, outputs: list[TurnOutput]) -> None:
        assert task_passed([PASS] * len(outputs), outputs) is False

    def test_a_task_without_turns_is_refused(self) -> None:
        with pytest.raises(ValueError, match="one score and one output per turn"):
            task_passed([], [])

    @pytest.mark.parametrize(
        ("scores", "outputs"), [([PASS], [delegated(FOUND)] * 2), ([PASS, PASS], [delegated(FOUND)])]
    )
    def test_scores_and_outputs_that_do_not_pair_up_are_refused(
        self, scores: list[TurnScore], outputs: list[TurnOutput]
    ) -> None:
        with pytest.raises(ValueError, match="one score and one output per turn"):
            task_passed(scores, outputs)

    def test_a_turn_lost_to_the_infrastructure_is_refused(self) -> None:
        # Such an attempt is tried again and left out of the scores; it never passes or fails.
        lost = delegated(RetrievalResult(status="not_searched", infra_error="A2A ReadTimeout"))
        with pytest.raises(ValueError, match="infrastructure"):
            task_passed([PASS, PASS], [delegated(FOUND), lost])


# ── Whole turns through the real system ──────────────────────────────────────────────────────────────────

DELEGATE = ollama_tool_call("Decision", route="delegate", reason="search")
DECLINE = ollama_tool_call("Decision", route="decline", reason="out_of_scope")
NO_PERIOD = ollama_tool_call("Decision", route="clarify", reason="missing_period")
QUESTION = "Sentinel-2 scenes of Milan from March 2025 with little cloud"
NAPLES_WEEK_1 = {
    "product": "sentinel-1-grd",
    "bbox": box("Naples"),
    "start_date": "2025-09-01",
    "end_date": "2025-09-07",
}
NAPLES_WEEK_2 = NAPLES_WEEK_1 | {"start_date": "2025-09-08", "end_date": "2025-09-14"}
DEV = {
    task.id: task for task in load_tasks(GOLDEN_DIR / "dev.jsonl")
}  # the dev split only, never the test one

Turn = tuple[TurnOutput, list[ToolCallRecord]]


def settings(**changes: Any) -> Settings:  # noqa: ANN401 - any setting, validated by Settings itself
    return Settings(_env_file=None, **changes)  # type: ignore[call-arg]  # a pydantic-settings init option


def says(text: str = "Here is what I found.") -> AIMessage:
    return AIMessage(text)


@asynccontextmanager
async def mcp_server(log: Path, catalog: FakeCatalog) -> AsyncIterator[McpHttpClientFactory]:
    """The real MCP server in-process, writing its real tool-call log; yields what reaches it over HTTP."""
    server = build_server(catalog, GAZETTEER, ToolCallLog(log))
    app = server.streamable_http_app()  # before its session manager, which it creates

    def client(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        transport = httpx.ASGITransport(app=app)
        return httpx.AsyncClient(transport=transport, headers=headers, timeout=timeout, auth=auth)

    async with server.session_manager.run():
        yield client


class InProcessAgent:
    """Stands in for the A2A client: the real retrieval loop, its result carried as JSON, as over A2A."""

    def __init__(self, model: BaseChatModel, http_client: McpHttpClientFactory, config: Settings) -> None:
        self.model = model
        self.http_client = http_client
        self.config = config

    async def retrieve(self, request: RetrievalRequest, trace_id: str) -> RetrievalResult:
        result = await run_retrieval(request, trace_id, self.model, self.config, http_client=self.http_client)
        return RetrievalResult.model_validate_json(result.model_dump_json())


async def run_turns(
    log: Path,
    script: Sequence[AIMessage],
    *,
    decisions: Sequence[OllamaStep] = (DELEGATE,),
    texts: Sequence[str] = (QUESTION,),
    catalog: FakeCatalog | None = None,
    **changes: Any,  # noqa: ANN401 - settings of the retrieval agent
) -> list[Turn]:
    """Run one conversation through the real system: each turn's output, and its calls as the log has them.

    `decisions` are the orchestrator's model replies, one per turn; `script` holds the retrieval agent's
    model replies for all the turns, in order.
    """
    agent_model = ScriptedModel(messages=iter(script))
    turns: list[Turn] = []
    async with mcp_server(log, catalog or FakeCatalog()) as http_client:
        agent = InProcessAgent(agent_model, http_client, settings(**changes))
        orchestrator = Orchestrator(
            FakeOllama(*decisions).model(), agent, MODEL_ID, decision_deadline_s=150.0
        )
        for number, text in enumerate(texts, start=1):
            trace_id = f"turn-{number}"
            output = await orchestrator.run_turn("conversation-1", text, trace_id, TODAY)
            turns.append((output, read_tool_calls(log, trace_id)))
    return turns


class FirstSearchFinishesLast(FakeCatalog):
    """Holds a search for one scene until another search is in the log, so the call sent first ends last."""

    def __init__(self, log: Path) -> None:
        super().__init__([SCENE, OTHER_SCENE])
        self.log = log

    def search(self, request: SearchRequest) -> list[SceneSummary]:
        give_up = time.monotonic() + 5
        while request.limit == 1 and '"search_scenes"' not in self.log.read_text(encoding="utf-8"):
            if time.monotonic() > give_up:
                raise AssertionError("the other search was never recorded")
            time.sleep(0.01)  # in the server's worker thread, not the event loop
        return super().search(request)


RESOLVE_MILAN = tool_call("resolve_place", "call-1", name="Milan")
SEARCH_MILAN = tool_call("search_scenes", "call-2", **MILAN_MARCH)
BAD_SEARCH = tool_call("search_scenes", "call-2", **{**MILAN_MARCH, "cloud_cover": 20})

ORDINARY_TURNS = [
    pytest.param(
        [RESOLVE_MILAN, SEARCH_MILAN, says()], "ok", ["resolve_place", "search_scenes"], id="scenes found"
    ),
    pytest.param(
        [RESOLVE_MILAN, BAD_SEARCH, tool_call("search_scenes", "call-3", **MILAN_MARCH), says()],
        "ok",
        ["resolve_place", "search_scenes", "search_scenes"],
        id="a refused search, then a corrected one",
    ),
    pytest.param(
        [RESOLVE_MILAN, BAD_SEARCH, says("I could not search.")],
        "error",
        ["resolve_place", "search_scenes"],
        id="a refused search the model gives up on",
    ),
    pytest.param(
        [
            RESOLVE_MILAN,
            tool_call("search_scenes", "call-2", **{**MILAN_MARCH, "bbox": [9.041, 45.387, 9.278, 45.536]}),
            says(),
        ],
        "ok",
        ["resolve_place", "search_scenes"],
        id="a box copied to three decimals",
    ),
    pytest.param(
        [
            tool_call("search_scenes", "call-1", **{**NAPLES_WEEK_1, "max_cloud_cover": None, "limit": 3}),
            says(),
        ],
        "ok",
        ["search_scenes"],
        id="a typed box, an explicit null and a limit",
    ),
    pytest.param(
        [tool_call("resolve_place", "call-1", name="Nowhereville"), says("Which place do you mean?")],
        "clarified",
        ["resolve_place"],
        id="a place the gazetteer does not know",
    ),
    pytest.param(
        [
            tool_call("resolve_place", "call-1", name="Nowhereville"),
            tool_call("resolve_place", "call-2", name="Milan"),
            says("Which place do you mean?"),
        ],
        "clarified",
        ["resolve_place", "resolve_place"],
        id="an unknown place, then a known one, and no search",
    ),
    pytest.param(
        [tool_call("resolve_place", "call-1", name="Milan", country="IT"), says("I could not look it up.")],
        "error",
        ["resolve_place"],
        id="a refused place lookup",
    ),
    pytest.param([says("I can only search scene metadata.")], "declined", [], id="no tool call at all"),
    pytest.param(
        [tool_call("list_products", "call-1"), says("These are the products.")],
        "declined",
        ["list_products"],
        id="only the product list",
    ),
    pytest.param(
        [RESOLVE_MILAN, says("I searched and found scenes.")],
        "declined",
        ["resolve_place"],
        id="a place found and never searched",
    ),
    pytest.param(
        [tool_call("get_scene", "call-1", product="sentinel-2-l2a", scene_id=SCENE.id), says("never read")],
        "declined",
        [],
        id="a tool the agent was not given",
    ),
]


class TestRealTurns:
    """The rules against turns as the system produces them: an ordinary turn is never taken for a bug."""

    @pytest.mark.parametrize(("script", "status", "tools"), ORDINARY_TURNS)
    async def test_an_ordinary_turn_is_scored(
        self, tmp_path: Path, script: list[AIMessage], status: str, tools: list[str]
    ) -> None:
        [(output, records)] = await run_turns(tmp_path / "calls.jsonl", script)
        assert (output.route, output.status) == (Route.DELEGATE, status)
        assert [record.tool for record in records] == tools
        assert turn_outcome(output, records) == ("scored", None)
        assert [called.name for called in turn_trace(output, records).calls] == tools

    async def test_a_search_that_finds_nothing_is_scored(self, tmp_path: Path) -> None:
        script = [RESOLVE_MILAN, SEARCH_MILAN, says("No scenes match.")]
        [(output, records)] = await run_turns(tmp_path / "calls.jsonl", script, catalog=FakeCatalog([]))
        assert output.status == "empty"
        assert turn_outcome(output, records) == ("scored", None)

    @pytest.mark.parametrize(("decision", "route"), [(DECLINE, Route.DECLINE), (NO_PERIOD, Route.CLARIFY)])
    async def test_a_turn_the_orchestrator_answers_itself_is_scored(
        self, tmp_path: Path, decision: OllamaStep, route: Route
    ) -> None:
        [(output, records)] = await run_turns(tmp_path / "calls.jsonl", [], decisions=[decision])
        assert (output.route, records) == (route, [])
        assert turn_outcome(output, records) == ("scored", None)

    async def test_the_last_search_is_the_last_one_sent_not_the_last_one_to_finish(
        self, tmp_path: Path
    ) -> None:
        # The graph runs both calls of one reply at once. The agent reports the second; the log's file has the
        # second first, and read by request id it gives the second last, so the two accounts agree.
        log = tmp_path / "calls.jsonl"
        both = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_scenes",
                    "args": MILAN_MARCH | {"limit": 1},
                    "id": "call-1",
                    "type": "tool_call",
                },
                {
                    "name": "search_scenes",
                    "args": MILAN_MARCH | {"limit": 2},
                    "id": "call-2",
                    "type": "tool_call",
                },
            ],
        )
        [(output, records)] = await run_turns(log, [both, says()], catalog=FirstSearchFinishesLast(log))
        written = [json.loads(line)["arguments"]["limit"] for line in log.read_text("utf-8").splitlines()]
        assert written == [2, 1]  # the order they finished
        assert [record.arguments["limit"] for record in records] == [1, 2]  # the order they were sent
        assert output.cited_ids == [SCENE.id, OTHER_SCENE.id]
        assert turn_outcome(output, records) == ("scored", None)

    async def test_a_follow_up_that_reuses_the_previous_box_is_scored(self, tmp_path: Path) -> None:
        script = [
            tool_call("resolve_place", "call-1", name="Naples"),
            tool_call("search_scenes", "call-2", **NAPLES_WEEK_1),
            says(),
            tool_call("search_scenes", "call-3", **NAPLES_WEEK_2),
            says(),
        ]
        first, second = await run_turns(
            tmp_path / "calls.jsonl",
            script,
            decisions=[DELEGATE, DELEGATE],
            texts=["Radar over Naples, first week of September 2025", "And the week after that?"],
        )
        assert [record.tool for record in second[1]] == ["search_scenes"]
        assert second[0].retrieval is not None
        assert second[0].retrieval.search is not None
        assert second[0].retrieval.search.place == "Naples"  # kept from the first turn; the log has no place
        assert turn_outcome(*first) == ("scored", None)
        assert turn_outcome(*second) == ("scored", None)

    async def test_a_model_that_runs_out_of_steps_after_a_search_is_scored_and_fails_the_task(
        self, tmp_path: Path
    ) -> None:
        script = [
            RESOLVE_MILAN,
            SEARCH_MILAN,
            tool_call("resolve_place", "call-3", name="Milan"),
            tool_call("resolve_place", "call-4", name="Milan"),  # the fourth reply still calls a tool
        ]
        [(output, records)] = await run_turns(tmp_path / "calls.jsonl", script, max_model_calls=4)
        assert (output.status, output.failure) == ("ok", "step_limit")
        assert [record.tool for record in records] == ["resolve_place", "search_scenes", "resolve_place"]
        assert turn_outcome(output, records) == ("scored", None)
        assert task_passed([PASS], [output]) is False

    async def test_a_catalogue_failure_is_infrastructure(self, tmp_path: Path) -> None:
        script = [RESOLVE_MILAN, SEARCH_MILAN, says("The catalogue did not answer.")]
        [(output, records)] = await run_turns(
            tmp_path / "calls.jsonl", script, catalog=FakeCatalog(fail=True)
        )
        assert (output.status, output.infra_error) == ("error", None)  # the agent cannot tell
        assert turn_outcome(output, records)[0] == "infrastructure"

    async def test_a_real_turn_with_a_call_missing_from_the_log_is_a_bug(self, tmp_path: Path) -> None:
        # The control for this class: the comparison does tell a real turn from one whose log lost a call.
        [(output, records)] = await run_turns(tmp_path / "calls.jsonl", [RESOLVE_MILAN, SEARCH_MILAN, says()])
        assert turn_outcome(output, records) == ("scored", None)
        assert turn_outcome(output, records[:-1])[0] == "errored"


class TestDevTasks:
    """From a dev question to the task's result: the trace built here is one the frozen scoring accepts."""

    async def test_a_search_task_passes(self, tmp_path: Path) -> None:
        turn = DEV["dev-s2-milan-march"].turns[0]
        script = [RESOLVE_MILAN, SEARCH_MILAN, says()]
        [(output, records)] = await run_turns(tmp_path / "calls.jsonl", script, texts=[turn.user])
        assert turn_outcome(output, records) == ("scored", None)
        score = score_turn(turn.expect, turn_trace(output, records), GAZETTEER)
        assert score.passed
        assert task_passed([score], [output]) is True

    async def test_a_search_with_the_wrong_dates_fails_on_the_logged_arguments(self, tmp_path: Path) -> None:
        turn = DEV["dev-s2-milan-march"].turns[0]
        wrong = tool_call("search_scenes", "call-2", **{**MILAN_MARCH, "end_date": "2025-03-30"})
        [(output, records)] = await run_turns(
            tmp_path / "calls.jsonl", [RESOLVE_MILAN, wrong, says()], texts=[turn.user]
        )
        assert turn_outcome(output, records) == ("scored", None)
        score = score_turn(turn.expect, turn_trace(output, records), GAZETTEER)
        assert score.arguments is not None
        assert (score.arguments.dates, score.arguments.bbox, score.passed) == (False, True, False)
        assert task_passed([score], [output]) is False

    async def test_a_follow_up_task_passes_with_the_ids_of_its_first_turn(self, tmp_path: Path) -> None:
        task = DEV["dev-follow-naples-weeks"]
        script = [
            tool_call("resolve_place", "call-1", name="Naples"),
            tool_call("search_scenes", "call-2", **NAPLES_WEEK_1),
            says(),
            tool_call("search_scenes", "call-3", **NAPLES_WEEK_2),
            says(),
        ]
        turns = await run_turns(
            tmp_path / "calls.jsonl",
            script,
            decisions=[DELEGATE, DELEGATE],
            texts=[turn.user for turn in task.turns],
        )
        scores: list[TurnScore] = []
        known: frozenset[str] = frozenset()
        for turn, (output, records) in zip(task.turns, turns, strict=True):
            assert turn_outcome(output, records) == ("scored", None)
            trace = turn_trace(output, records)
            scores.append(score_turn(turn.expect, trace, GAZETTEER, known))
            known |= {scene_id for called in trace.calls for scene_id in called.returned_ids}
        assert [score.passed for score in scores] == [True, True]
        assert task_passed(scores, [output for output, _ in turns]) is True

    async def test_a_declined_task_passes(self, tmp_path: Path) -> None:
        turn = DEV["dev-oos-weather-naples"].turns[0]
        [(output, records)] = await run_turns(
            tmp_path / "calls.jsonl", [], decisions=[DECLINE], texts=[turn.user]
        )
        assert turn_outcome(output, records) == ("scored", None)
        score = score_turn(turn.expect, turn_trace(output, records), GAZETTEER)
        assert task_passed([score], [output]) is True
