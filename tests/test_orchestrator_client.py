"""Specification for the orchestrator's A2A client, against the real A2A server app in-process.

The app is reached through httpx.ASGITransport, so no socket is opened; a wrapper around that transport raises
the network failures that httpx raises on a real socket. The agent's turn is a stand-in, and so is the whole
agent where a test needs it to break the contract.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

import httpx
import pytest
from a2a.client import A2AClientError, AgentCardResolutionError
from a2a.helpers.proto_helpers import new_data_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types.a2a_pb2 import (
    AgentCard,
    Artifact,
    Message,
    Part,
    Role,
    StreamResponse,
    Task,
    TaskState,
    TaskStatus,
)
from a2a.utils.errors import InternalError
from langchain_core.language_models import BaseChatModel

from agentic_earth_observation.config import Settings
from agentic_earth_observation.orchestrator.client import (
    A2ARetrievalClient,
    RetrievalAgentError,
    RetrievalClient,
    _result,
)
from agentic_earth_observation.retrieval_agent import a2a_server
from agentic_earth_observation.retrieval_agent.a2a_server import RetrievalExecutor, build_app
from agentic_earth_observation.retrieval_agent.models import (
    RESULT_ARTIFACT,
    Clarification,
    PreviousSearch,
    RetrievalRequest,
    RetrievalResult,
)
from tests.fakes import SCENE, ScriptedModel

pytestmark = pytest.mark.anyio

JSON = "application/json"
NAPLES_WEEK = PreviousSearch.model_validate_json(
    json.dumps(
        {
            "place": "Naples",
            "product": "sentinel-1-grd",
            "bbox": [14.1321, 40.7912, 14.3538, 40.9161],
            "start_date": "2025-09-01",
            "end_date": "2025-09-07",
            "limit": 3,  # an integer, which must arrive as one
        }
    )
)
REQUEST = RetrievalRequest(
    text="And the week after that?", today=date(2025, 12, 15), previous_search=NAPLES_WEEK
)
FOUND = RetrievalResult(status="ok", scenes=[SCENE], search=NAPLES_WEEK)
Outcome = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def settings(**changes: Any) -> Settings:  # noqa: ANN401 - any setting, validated by Settings itself
    return Settings(_env_file=None, **changes)  # type: ignore[call-arg]  # a pydantic-settings init option


class StandIn:
    """Stands in for run_retrieval on the agent's side: records each call, then returns `result` or raises."""

    def __init__(self, result: RetrievalResult = FOUND, bug: Exception | None = None) -> None:
        self.result = result
        self.bug = bug
        self.calls: list[tuple[RetrievalRequest, str]] = []

    async def __call__(
        self, request: RetrievalRequest, trace_id: str, model: BaseChatModel, config: Settings
    ) -> RetrievalResult:
        self.calls.append((request, trace_id))
        if self.bug is not None:
            raise self.bug
        return self.result


class Ends(AgentExecutor):
    """An agent that ends every task as told: a stand-in for a broken agent, or one from another version."""

    def __init__(
        self, state: TaskState, artifacts: Sequence[tuple[str, list[Part]]] = (), reason: str = ""
    ) -> None:
        self.state = state
        self.artifacts = list(artifacts)
        self.reason = reason

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id, context_id = context.task_id or "", context.context_id or ""
        status = TaskStatus(state=TaskState.TASK_STATE_SUBMITTED)
        await event_queue.enqueue_event(Task(id=task_id, context_id=context_id, status=status))
        updater = TaskUpdater(event_queue, task_id, context_id)
        for name, parts in self.artifacts:
            await updater.add_artifact(parts, name=name)
        message = updater.new_agent_message([Part(text=self.reason)]) if self.reason else None
        if self.state == TaskState.TASK_STATE_COMPLETED:
            await updater.complete(message)
        elif self.state == TaskState.TASK_STATE_REJECTED:
            await updater.reject(message)
        elif self.state == TaskState.TASK_STATE_CANCELED:
            await updater.cancel(message)
        # any other state: the task is left as it is, which the SDK returns as submitted

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        return None


class AnswersWithMessage(AgentExecutor):
    """An agent that answers with a plain message instead of a task, which the contract does not allow."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        parts = [json_text(FOUND.model_dump_json())]
        await event_queue.enqueue_event(
            Message(role=Role.ROLE_AGENT, message_id="m-1", context_id=context.context_id or "", parts=parts)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        return None


class Rigged(httpx.AsyncBaseTransport):
    """Passes requests on to the app, except those with `method`, which get `outcome` (it may raise) while
    `times` lasts. Without an outcome, every request passes."""

    def __init__(
        self,
        inner: httpx.AsyncBaseTransport,
        method: str = "",
        outcome: Outcome | None = None,
        times: int = 1000,
    ) -> None:
        self.inner = inner
        self.method = method
        self.outcome = outcome
        self.times = times
        self.seen: list[str] = []
        self.bodies: list[dict[str, Any]] = []  # the JSON-RPC requests sent

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(f"{request.method} {request.url.path}")
        if request.method == "POST":
            self.bodies.append(json.loads(request.content))
        if self.outcome is not None and request.method == self.method and self.times > 0:
            self.times -= 1
            return self.outcome(request)
        return await self.inner.handle_async_request(request)


def raising(error: Exception) -> Outcome:
    def outcome(request: httpx.Request) -> httpx.Response:
        error.request = request  # type: ignore[attr-defined]  # httpx's errors carry the request
        raise error

    return outcome


def answering(status: int, **content: Any) -> Outcome:  # noqa: ANN401 - httpx.Response's content arguments
    return lambda request: httpx.Response(status, **content)


@asynccontextmanager
async def agent(
    executor: AgentExecutor | None = None,
    *,
    method: str = "",
    outcome: Outcome | None = None,
    times: int = 1000,
    config: Settings | None = None,
) -> AsyncIterator[tuple[A2ARetrievalClient, Rigged]]:
    """The real A2A app in-process (with a stand-in turn unless `executor` is given) and a client of it."""
    config = config or settings()
    model = ScriptedModel(messages=iter([]))
    app = build_app(executor or RetrievalExecutor(model, config, StandIn()), config)
    transport = Rigged(httpx.ASGITransport(app=app), method, outcome, times)
    http = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(config.a2a_timeout_s))
    async with app.router.lifespan_context(app), A2ARetrievalClient(config, http=http) as client:
        yield client, transport


def json_text(text: str) -> Part:
    return Part(text=text, media_type=JSON)


class TestAnswers:
    async def test_a_completed_turn_returns_the_agents_result(self) -> None:
        run = StandIn()
        config = settings()
        app = build_app(RetrievalExecutor(ScriptedModel(messages=iter([])), config, run), config)
        transport = Rigged(httpx.ASGITransport(app=app))
        client = A2ARetrievalClient(config, http=httpx.AsyncClient(transport=transport))
        async with app.router.lifespan_context(app), client:
            first = await client.retrieve(REQUEST, "run-1.turn-1")
            second = await client.retrieve(REQUEST, "run-1.turn-2")
        assert first == second == FOUND
        assert run.calls == [(REQUEST, "run-1.turn-1"), (REQUEST, "run-1.turn-2")]
        assert run.calls[0][0].previous_search is not None
        assert run.calls[0][0].previous_search.limit == 3
        # The card is read once, then each turn is one call to the endpoint it names.
        assert transport.seen == ["GET /.well-known/agent-card.json", "POST /a2a", "POST /a2a"]

    async def test_each_call_is_one_new_user_message_never_streamed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Even an agent that offers streaming gets a plain request, so that there is one answer to read.
        offered = a2a_server.agent_card

        def streaming_card(config: Settings) -> AgentCard:
            card = offered(config)
            card.capabilities.streaming = True
            return card

        monkeypatch.setattr(a2a_server, "agent_card", streaming_card)
        async with agent() as (client, transport):
            assert await client.retrieve(REQUEST, "turn-1") == FOUND
            assert await client.retrieve(REQUEST, "turn-2") == FOUND
        first, second = transport.bodies
        assert (first["method"], second["method"]) == ("SendMessage", "SendMessage")
        assert (first["params"]["message"]["role"], second["params"]["message"]["role"]) == (
            "ROLE_USER",
            "ROLE_USER",
        )
        assert first["params"]["message"]["messageId"] != second["params"]["message"]["messageId"]
        assert (first["params"]["metadata"], second["params"]["metadata"]) == (
            {"trace_id": "turn-1"},
            {"trace_id": "turn-2"},
        )
        ((part,),) = [first["params"]["message"]["parts"]]
        assert part == {"text": REQUEST.model_dump_json(), "mediaType": "application/json"}

    async def test_the_question_an_answer_answers_crosses_intact(self) -> None:
        run = StandIn()
        config = settings()
        asked = Clarification(
            request="Radar scenes of Napels in September 2025", asked="Did you mean Naples?"
        )
        answer = RetrievalRequest(
            text="Yes", today=REQUEST.today, previous_search=NAPLES_WEEK, clarifies=asked
        )
        async with agent(RetrievalExecutor(ScriptedModel(messages=iter([])), config, run), config=config) as (
            client,
            _,
        ):
            await client.retrieve(answer, "turn-1")
        assert run.calls == [(answer, "turn-1")]

    async def test_an_infrastructure_result_from_the_agent_is_returned_as_it_is(self) -> None:
        failed = RetrievalResult(
            status="not_searched", infra_error="ConnectError: All connection attempts failed"
        )
        config = settings()
        executor = RetrievalExecutor(ScriptedModel(messages=iter([])), config, StandIn(failed))
        async with agent(executor, config=config) as (client, _):
            assert await client.retrieve(REQUEST, "turn-1") == failed

    async def test_a_failed_turn_raises(self) -> None:
        config = settings()
        bug = StandIn(bug=ValueError("internal detail /secret/path"))
        async with agent(RetrievalExecutor(ScriptedModel(messages=iter([])), config, bug), config=config) as (
            client,
            _,
        ):
            with pytest.raises(RetrievalAgentError) as raised:
                await client.retrieve(REQUEST, "turn-7")
        assert str(raised.value) == (
            "the retrieval agent ended turn turn-7 as TASK_STATE_FAILED: "
            "the retrieval agent failed (ValueError); its log has the details"
        )

    async def test_a_rejected_request_raises_with_the_agents_reason(self) -> None:
        reason = "invalid RetrievalRequest: today: Field required"
        async with agent(Ends(TaskState.TASK_STATE_REJECTED, reason=reason)) as (client, _):
            with pytest.raises(
                RetrievalAgentError, match=f"ended turn turn-1 as TASK_STATE_REJECTED: {reason}$"
            ):
                await client.retrieve(REQUEST, "turn-1")

    @pytest.mark.parametrize(
        ("executor", "problem"),
        [
            (Ends(TaskState.TASK_STATE_COMPLETED), "is not one JSON result"),
            (
                Ends(
                    TaskState.TASK_STATE_COMPLETED,
                    [("retrieval_result", [json_text(FOUND.model_dump_json())])] * 2,
                ),
                "is not one JSON result",
            ),
            (
                Ends(TaskState.TASK_STATE_COMPLETED, [("result", [json_text(FOUND.model_dump_json())])]),
                "is not one JSON result",
            ),
            (
                Ends(
                    TaskState.TASK_STATE_COMPLETED,
                    [("retrieval_result", [json_text(FOUND.model_dump_json())] * 2)],
                ),
                "is not one JSON result",
            ),
            (
                Ends(
                    TaskState.TASK_STATE_COMPLETED,
                    [("retrieval_result", [Part(text=FOUND.model_dump_json(), media_type="text/plain")])],
                ),
                "is not one JSON result",
            ),
            (
                Ends(
                    TaskState.TASK_STATE_COMPLETED,
                    [
                        (
                            "retrieval_result",
                            [new_data_part(json.loads(FOUND.model_dump_json()), media_type=JSON)],
                        )
                    ],
                ),
                "is not one JSON result",
            ),
            (
                Ends(
                    TaskState.TASK_STATE_COMPLETED, [("retrieval_result", [json_text('{"status": "lost"}')])]
                ),
                "is not a valid RetrievalResult",
            ),
            (
                Ends(TaskState.TASK_STATE_CANCELED),
                "ended turn turn-1 as TASK_STATE_CANCELED: no reason given",
            ),
            (
                Ends(TaskState.TASK_STATE_WORKING),
                "ended turn turn-1 as TASK_STATE_SUBMITTED: no reason given",
            ),
        ],
    )
    async def test_an_answer_outside_the_contract_raises(self, executor: AgentExecutor, problem: str) -> None:
        async with agent(executor) as (client, _):
            with pytest.raises(RetrievalAgentError, match=problem):
                await client.retrieve(REQUEST, "turn-1")

    async def test_an_answer_that_is_not_a_task_raises(self) -> None:
        async with agent(AnswersWithMessage()) as (client, _):
            with pytest.raises(RetrievalAgentError, match="answered turn turn-1 without a task"):
                await client.retrieve(REQUEST, "turn-1")

    def test_two_answers_to_one_turn_are_outside_the_contract(self) -> None:
        # The client never streams, so the SDK hands it exactly one answer; the check on their number is the
        # contract's own and is reached directly, with two completed tasks that would each be valid alone.
        completed = Task(
            id="task-1",
            context_id="conv-1",
            status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
            artifacts=[
                Artifact(artifact_id="a-1", name=RESULT_ARTIFACT, parts=[json_text(FOUND.model_dump_json())])
            ],
        )
        assert _result([StreamResponse(task=completed)], "turn-1") == FOUND
        with pytest.raises(RetrievalAgentError, match="answered turn turn-1 2 times, not once"):
            _result([StreamResponse(task=completed), StreamResponse(task=completed)], "turn-1")

    @pytest.mark.parametrize("trace_id", ["", "turn 1", "turn-1\n", "a" * 65, "run-3/turn-1"])
    async def test_a_malformed_trace_id_is_refused_before_anything_is_sent(self, trace_id: str) -> None:
        async with agent() as (client, transport):
            with pytest.raises(ValueError, match="invalid trace id"):
                await client.retrieve(REQUEST, trace_id)
        assert transport.seen == []


class TestInfrastructure:
    """The agent out of reach is infrastructure: the turn is retried and excluded, never scored."""

    @pytest.mark.parametrize(
        ("method", "error", "reported"),
        [
            (
                "GET",
                httpx.ConnectError("All connection attempts failed"),
                "A2A ConnectError: All connection attempts failed",
            ),
            (
                "POST",
                httpx.ConnectError("All connection attempts failed"),
                "A2A ConnectError: All connection attempts failed",
            ),
            ("POST", httpx.ReadTimeout(""), "A2A ReadTimeout"),
            (
                "POST",
                httpx.RemoteProtocolError("Server disconnected without sending a response."),
                "A2A RemoteProtocolError: Server disconnected without sending a response.",
            ),
        ],
    )
    async def test_an_agent_that_cannot_be_reached_or_does_not_answer(
        self, method: str, error: Exception, reported: str
    ) -> None:
        async with agent(method=method, outcome=raising(error)) as (client, _):
            result = await client.retrieve(REQUEST, "turn-1")
        assert result == RetrievalResult(status="not_searched", infra_error=reported)

    async def test_the_card_is_fetched_again_after_a_failed_fetch(self) -> None:
        down = raising(httpx.ConnectError("All connection attempts failed"))
        async with agent(method="GET", outcome=down, times=1) as (client, transport):
            assert (await client.retrieve(REQUEST, "turn-1")).infra_error is not None
            assert await client.retrieve(REQUEST, "turn-2") == FOUND
        assert transport.seen == ["GET /.well-known/agent-card.json"] * 2 + ["POST /a2a"]


class TestRaised:
    """Configuration errors and bugs are raised, never counted as infrastructure."""

    async def test_a_refused_request(self) -> None:
        refused = answering(403, text="requests from web pages are not accepted")
        async with agent(method="POST", outcome=refused) as (client, _):
            with pytest.raises(A2AClientError, match="HTTP Error 403"):
                await client.retrieve(REQUEST, "turn-1")

    async def test_a_card_that_is_not_there(self) -> None:
        async with agent(method="GET", outcome=answering(404, text="Not Found")) as (client, _):
            with pytest.raises(AgentCardResolutionError, match="HTTP 404"):
                await client.retrieve(REQUEST, "turn-1")

    async def test_a_json_rpc_error(self) -> None:
        error = {"jsonrpc": "2.0", "id": "1", "error": {"code": -32603, "message": "Internal error: boom"}}
        async with agent(method="POST", outcome=answering(200, json=error)) as (client, _):
            with pytest.raises(InternalError, match="boom"):
                await client.retrieve(REQUEST, "turn-1")

    async def test_an_answer_that_is_not_json(self) -> None:
        async with agent(method="POST", outcome=answering(200, text="<html>oops</html>")) as (client, _):
            with pytest.raises(A2AClientError, match="JSON Decode Error"):
                await client.retrieve(REQUEST, "turn-1")


class TestLifecycle:
    async def test_by_default_a_call_may_take_as_long_as_the_a2a_timeout(self) -> None:
        client = A2ARetrievalClient(settings(a2a_timeout_s=280))
        assert client.http.timeout == httpx.Timeout(280)
        await client.aclose()
        assert client.http.is_closed

    async def test_closing_after_use_closes_the_http_client(self) -> None:
        async with agent() as (client, _):
            await client.retrieve(REQUEST, "turn-1")
        assert client.http.is_closed

    async def test_it_is_what_the_orchestrator_depends_on(self) -> None:
        client: RetrievalClient = A2ARetrievalClient(settings())  # checked by mypy
        assert isinstance(client, A2ARetrievalClient)
        await client.aclose()

    async def test_calls_can_run_side_by_side(self) -> None:
        async with agent() as (client, transport):
            results = await asyncio.gather(*(client.retrieve(REQUEST, f"turn-{i}") for i in range(3)))
        assert results == [FOUND] * 3
        assert transport.seen.count("POST /a2a") == 3


async def test_a_client_of_this_machine_ignores_proxy_settings() -> None:
    # A request to a local server cannot get there through a proxy, and would hand the user's words to it.
    async with A2ARetrievalClient(Settings(_env_file=None)) as client:  # type: ignore[call-arg]  # a pydantic-settings init option
        assert client.http.trust_env is False
