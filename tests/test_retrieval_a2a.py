"""Specification for the retrieval agent's A2A server, through the real A2A client.

The app runs in-process and is reached through httpx.ASGITransport, so no socket is opened. The retrieval turn
is a stand-in, except in the last test, which runs the real one through the in-process MCP server.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from a2a.client import Client, ClientConfig, create_client
from a2a.helpers.proto_helpers import new_data_part
from a2a.types.a2a_pb2 import (
    CancelTaskRequest,
    GetTaskRequest,
    Message,
    Part,
    Role,
    SendMessageConfiguration,
    SendMessageRequest,
    Task,
    TaskState,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from mcp.shared._httpx_utils import McpHttpClientFactory
from starlette.applications import Starlette

from agentic_earth_observation.catalog.gazetteer import Gazetteer
from agentic_earth_observation.catalog_mcp.server import build_server
from agentic_earth_observation.catalog_mcp.tool_log import ToolCallLog, read_tool_calls
from agentic_earth_observation.config import Settings
from agentic_earth_observation.retrieval_agent import a2a_server
from agentic_earth_observation.retrieval_agent.a2a_server import (
    RetrievalExecutor,
    build_app,
    local_port,
)
from agentic_earth_observation.retrieval_agent.agent import run_retrieval
from agentic_earth_observation.retrieval_agent.models import PreviousSearch, RetrievalRequest, RetrievalResult
from tests.fakes import SCENE, FakeCatalog, ScriptedModel, tool_call

pytestmark = pytest.mark.anyio

JSON = "application/json"
CARD_PATH = "/.well-known/agent-card.json"
TODAY = date(2025, 12, 15)
NAPLES_WEEK = PreviousSearch.model_validate_json(
    json.dumps(
        {
            "place": "Naples",
            "product": "sentinel-1-grd",
            "bbox": [14.1321, 40.7912, 14.3538, 40.9161],
            "start_date": "2025-09-01",
            "end_date": "2025-09-07",
            "limit": 3,  # an integer, which a data part would turn into 3.0
        }
    )
)
REQUEST = RetrievalRequest(text="And the week after that?", today=TODAY, previous_search=NAPLES_WEEK)
FOUND = RetrievalResult(status="ok", scenes=[SCENE], search=NAPLES_WEEK)
MILAN_MARCH = {
    "product": "sentinel-2-l2a",
    "bbox": [9.0409, 45.3867, 9.2781, 45.5358],
    "start_date": "2025-03-01",
    "end_date": "2025-03-31",
    "max_cloud_cover": 20,
}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def settings(**changes: Any) -> Settings:  # noqa: ANN401 - any setting, validated by Settings itself
    return Settings(_env_file=None, **changes)  # type: ignore[call-arg]  # a pydantic-settings init option


class FakeRun:
    """Stands in for run_retrieval: records each call, then returns `result`, raises `bug` or waits."""

    def __init__(
        self, result: RetrievalResult = FOUND, *, bug: Exception | None = None, wait: bool = False
    ) -> None:
        self.result = result
        self.bug = bug
        self.wait = wait
        self.calls: list[tuple[RetrievalRequest, str, BaseChatModel, Settings]] = []
        self.cancelled = False

    async def __call__(
        self, request: RetrievalRequest, trace_id: str, model: BaseChatModel, settings: Settings
    ) -> RetrievalResult:
        self.calls.append((request, trace_id, model, settings))
        if self.bug is not None:
            raise self.bug
        if self.wait:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return self.result


MODEL = ScriptedModel(messages=iter([]))


@asynccontextmanager
async def a2a(
    run: Any,  # noqa: ANN401 - a FakeRun or a wrapper of run_retrieval
    config: Settings | None = None,
    model: BaseChatModel = MODEL,
) -> AsyncIterator[tuple[Client, httpx.AsyncClient]]:
    """The A2A app in-process, with its lifespan, and the SDK client that reaches it through its card."""
    config = config or settings()
    app = build_app(RetrievalExecutor(model, config, run), config)
    async with app.router.lifespan_context(app):
        hx = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=config.a2a_url)
        client = await create_client(
            config.a2a_url, client_config=ClientConfig(httpx_client=hx, streaming=False)
        )
        try:
            yield client, hx
        finally:
            await client.close()


def message(*parts: Part) -> Message:
    return Message(role=Role.ROLE_USER, message_id=f"m-{time.monotonic_ns()}", parts=list(parts))


def json_text(text: str) -> Part:
    return Part(text=text, media_type=JSON)


async def send(client: Client, sent: Message, metadata: dict[str, Any] | None, **configuration: Any) -> Task:  # noqa: ANN401 - SendMessageConfiguration's fields
    request = SendMessageRequest(
        message=sent, metadata=metadata or {}, configuration=SendMessageConfiguration(**configuration)
    )
    responses = [r async for r in client.send_message(request)]
    assert len(responses) == 1
    assert responses[0].HasField("task")
    return responses[0].task


async def ask(client: Client, request: RetrievalRequest = REQUEST, trace_id: str = "turn-1") -> Task:
    return await send(client, message(json_text(request.model_dump_json())), {"trace_id": trace_id})


def state(task: Task) -> str:
    return str(TaskState.Name(task.status.state))  # typed as Any by the protobuf stubs


def status_text(task: Task) -> str:
    return " ".join(part.text for part in task.status.message.parts)


def result_of(task: Task) -> RetrievalResult:
    """The result a completed task carries: one artifact with one JSON text part."""
    assert state(task) == "TASK_STATE_COMPLETED"
    ((artifact,),) = [task.artifacts]
    ((part,),) = [artifact.parts]
    assert (artifact.name, part.media_type) == ("retrieval_result", JSON)
    return RetrievalResult.model_validate_json(part.text)


def sdk_tasks() -> int:
    """The SDK's own asyncio tasks still running, which must not pile up across requests."""
    return sum(
        1 for t in asyncio.all_tasks() if t.get_name().startswith(("producer:", "consumer:")) and not t.done()
    )


async def settled() -> int:
    """The SDK's tasks left once the last answer has been sent; they finish within moments."""
    give_up = time.monotonic() + 2
    while sdk_tasks() and time.monotonic() < give_up:
        await asyncio.sleep(0.01)
    return sdk_tasks()


class TestCard:
    async def test_the_card_offers_one_json_skill_at_the_a2a_endpoint(self) -> None:
        async with a2a(FakeRun()) as (_, hx):
            response = await hx.get(CARD_PATH)
        assert response.status_code == 200
        card = response.json()
        assert card["supportedInterfaces"] == [
            {"url": "http://127.0.0.1:8002/a2a", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}
        ]
        assert card["version"] == version("agentic-earth-observation")
        assert card["capabilities"] == {"streaming": False, "pushNotifications": False}
        assert card["defaultInputModes"] == card["defaultOutputModes"] == [JSON]
        ((skill,),) = [card["skills"]]
        assert (skill["id"], skill["inputModes"], skill["outputModes"]) == ("find_scenes", [JSON], [JSON])
        assert "'trace_id'" in skill["description"]

    @pytest.mark.parametrize("url", ["http://127.0.0.1:9100", "http://127.0.0.1:9100/"])
    async def test_the_endpoint_follows_the_configured_url(self, url: str) -> None:
        config = settings(a2a_url=url)
        async with a2a(FakeRun(), config) as (_, hx):
            card = (await hx.get(CARD_PATH)).json()
        assert card["supportedInterfaces"][0]["url"] == "http://127.0.0.1:9100/a2a"


class TestTasks:
    async def test_a_request_completes_with_the_result_as_json_text(self) -> None:
        run = FakeRun()
        config = settings(max_model_calls=5)  # not the default, so that the settings passed on are these
        async with a2a(run, config) as (client, _):
            task = await ask(client, trace_id="run-1.task-2.turn-1")
        assert result_of(task) == FOUND
        ((request, trace_id, model, used),) = run.calls
        assert request == REQUEST
        assert request.previous_search is not None
        assert request.previous_search.limit == 3  # exact, both ways
        assert (trace_id, model, used) == ("run-1.task-2.turn-1", MODEL, config)

    async def test_a_trace_id_of_64_characters_is_accepted(self) -> None:
        run = FakeRun()
        async with a2a(run) as (client, _):
            task = await ask(client, trace_id="a" * 64)
        assert state(task) == "TASK_STATE_COMPLETED"
        assert run.calls[0][1] == "a" * 64

    async def test_an_infrastructure_failure_is_a_completed_task_with_infra_error(self) -> None:
        # Infrastructure is an outcome of the turn, which the runner retries and excludes; not a broken agent.
        failed = RetrievalResult(
            status="not_searched", infra_error="ConnectError: All connection attempts failed"
        )
        async with a2a(FakeRun(failed)) as (client, _):
            task = await ask(client)
        assert result_of(task) == failed

    async def test_a_bug_fails_the_task_and_its_details_stay_in_the_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        run = FakeRun(bug=ValueError("internal detail /secret/path"))
        async with a2a(run) as (client, _):
            task = await ask(client, trace_id="turn-7")
        assert state(task) == "TASK_STATE_FAILED"
        assert status_text(task) == "the retrieval agent failed (ValueError); its log has the details"
        assert list(task.artifacts) == []
        assert "the retrieval turn turn-7 failed" in caplog.text
        assert "internal detail /secret/path" in caplog.text

    @pytest.mark.parametrize(
        ("parts", "metadata", "reason"),
        [
            ([json_text(REQUEST.model_dump_json())], None, "the request metadata needs a 'trace_id'"),
            (
                [json_text(REQUEST.model_dump_json())],
                {"trace_id": 3},
                "the request metadata needs a 'trace_id'",
            ),
            (
                [json_text(REQUEST.model_dump_json())],
                {"trace_id": "turn 1"},
                "the request metadata needs a 'trace_id': 1 to 64 letters",
            ),
            (
                [json_text(REQUEST.model_dump_json())],
                {"trace_id": "a" * 65},
                "the request metadata needs a 'trace_id': 1 to 64 letters",
            ),
            ([json_text("{not json")], {"trace_id": "t"}, "invalid RetrievalRequest: request: Invalid JSON"),
            (
                [json_text('{"text": "Milan"}')],
                {"trace_id": "t"},
                "invalid RetrievalRequest: today: Field required",
            ),
            (
                [json_text(REQUEST.model_copy(update={"text": "  "}).model_dump_json())],
                {"trace_id": "t"},
                "invalid RetrievalRequest: text: String should have at least 1 character",
            ),
            (
                [Part(text=REQUEST.model_dump_json(), media_type="text/plain")],
                {"trace_id": "t"},
                "send the RetrievalRequest as a text part with media type application/json",
            ),
            (
                [new_data_part(json.loads(REQUEST.model_dump_json()), media_type=JSON)],
                {"trace_id": "t"},
                "send the RetrievalRequest as a text part with media type application/json",
            ),
            (
                [json_text(REQUEST.model_dump_json())] * 2,
                {"trace_id": "t"},
                "send exactly one part: the RetrievalRequest as JSON text",
            ),
        ],
    )
    async def test_a_request_that_cannot_be_read_is_rejected_with_the_reason(
        self, parts: list[Part], metadata: dict[str, Any] | None, reason: str
    ) -> None:
        run = FakeRun()
        async with a2a(run) as (client, _):
            task = await send(client, message(*parts), metadata)
        assert state(task) == "TASK_STATE_REJECTED"
        assert status_text(task).startswith(reason)
        assert run.calls == []

    async def test_a_cancelled_task_stops_its_turn(self) -> None:
        # The orchestrator never cancels; the protocol allows it, so it must work.
        run = FakeRun(wait=True)
        async with a2a(run) as (client, _):
            task = await send(
                client,
                message(json_text(REQUEST.model_dump_json())),
                {"trace_id": "t"},
                return_immediately=True,
            )
            assert state(task) == "TASK_STATE_SUBMITTED"
            give_up = time.monotonic() + 2
            while not run.calls and time.monotonic() < give_up:
                await asyncio.sleep(0.01)
            cancelled = await client.cancel_task(CancelTaskRequest(id=task.id))
            assert state(cancelled) == "TASK_STATE_CANCELED"
            stored = await client.get_task(GetTaskRequest(id=task.id))
        assert state(stored) == "TASK_STATE_CANCELED"
        assert run.cancelled

    async def test_requests_leave_no_sdk_tasks_behind(self) -> None:
        # The SDK's message-style answers leak four asyncio tasks per request; task-style answers do not.
        async with a2a(FakeRun()) as (client, _):
            for i in range(10):
                assert state(await ask(client, trace_id=f"turn-{i}")) == "TASK_STATE_COMPLETED"
            assert await settled() == 0

    async def test_stopping_the_server_ends_a_turn_still_running(self) -> None:
        run = FakeRun(wait=True)
        async with a2a(run) as (client, _):
            await send(
                client,
                message(json_text(REQUEST.model_dump_json())),
                {"trace_id": "t"},
                return_immediately=True,
            )
            give_up = time.monotonic() + 2
            while not run.calls and time.monotonic() < give_up:
                await asyncio.sleep(0.01)
            assert sdk_tasks() > 0
        assert sdk_tasks() == 0
        assert run.cancelled


class TestGuards:
    """Only programs on this machine may start a turn: not other hosts, and not web pages."""

    @pytest.mark.parametrize(
        "host", ["evil.example", "evil.example:8002", "127.0.0.1.evil.example", "[::1]:8002"]
    )
    async def test_a_foreign_host_is_refused(self, host: str) -> None:
        async with a2a(FakeRun()) as (_, hx):
            card = await hx.get(CARD_PATH, headers={"host": host})
            rpc = await hx.post(
                "/a2a", headers={"host": host}, json={"jsonrpc": "2.0", "id": 1, "method": "SendMessage"}
            )
        assert (card.status_code, rpc.status_code) == (400, 400)

    @pytest.mark.parametrize("host", ["127.0.0.1:8002", "localhost:8002", "127.0.0.1"])
    async def test_this_machine_is_accepted(self, host: str) -> None:
        async with a2a(FakeRun()) as (_, hx):
            assert (await hx.get(CARD_PATH, headers={"host": host})).status_code == 200

    @pytest.mark.parametrize("origin", ["http://evil.example", "null", "http://127.0.0.1:8002"])
    async def test_a_request_from_a_web_page_is_refused(self, origin: str) -> None:
        run = FakeRun()
        async with a2a(run) as (_, hx):
            card = await hx.get(CARD_PATH, headers={"origin": origin})
            rpc = await hx.post(
                "/a2a", headers={"origin": origin}, json={"jsonrpc": "2.0", "id": 1, "method": "SendMessage"}
            )
        assert (card.status_code, rpc.status_code) == (403, 403)
        assert run.calls == []

    async def test_the_sdk_client_sends_no_origin(self) -> None:
        # Otherwise the orchestrator itself would be refused.
        sent: list[httpx.Headers] = []

        async def record(request: httpx.Request) -> None:
            sent.append(request.headers)

        async with a2a(FakeRun()) as (client, hx):
            hx.event_hooks["request"].append(record)
            await ask(client)
        assert sent
        assert all("origin" not in headers for headers in sent)


@pytest.mark.parametrize(("url", "port"), [("http://127.0.0.1:8002", 8002), ("http://localhost:9100/", 9100)])
def test_the_server_port_comes_from_a_local_url(url: str, port: int) -> None:
    assert local_port(url) == port


@pytest.mark.parametrize(
    "url", ["https://127.0.0.1:8002", "http://0.0.0.0:8002", "http://192.168.1.5:8002", "http://127.0.0.1"]
)
def test_a_url_off_this_machine_is_refused(url: str) -> None:
    with pytest.raises(ValueError, match=re.escape("EO_A2A_URL must be http://127.0.0.1:<port>")):
        local_port(url)


@asynccontextmanager
async def catalogue(log: Path) -> AsyncIterator[McpHttpClientFactory]:
    """The real MCP server in-process; yields the HTTP client factory that reaches it."""
    server = build_server(FakeCatalog(), Gazetteer.load(), ToolCallLog(log))
    app = server.streamable_http_app()

    def client(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), headers=headers, timeout=timeout, auth=auth
        )

    async with server.session_manager.run():
        yield client


async def test_a_turn_runs_from_a2a_through_mcp_under_the_trace_id_it_was_sent(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    model = ScriptedModel(
        messages=iter(
            [
                tool_call("resolve_place", name="Milan"),
                tool_call("search_scenes", "call-2", **MILAN_MARCH),
                AIMessage("One scene was found."),
            ]
        )
    )
    milan = RetrievalRequest(text="Milan, March 2025, little cloud", today=TODAY)
    async with catalogue(log) as mcp_client:

        async def run(
            request: RetrievalRequest, trace_id: str, model: BaseChatModel, config: Settings
        ) -> RetrievalResult:
            return await run_retrieval(request, trace_id, model, config, http_client=mcp_client)

        async with a2a(run, model=model) as (client, _):
            task = await ask(client, milan, trace_id="a2a-turn-1")
    result = result_of(task)
    assert (result.status, result.scenes) == ("ok", [SCENE])
    assert result.search is not None
    assert result.search.place == "Milan"
    assert [c.tool for c in read_tool_calls(log, "a2a-turn-1")] == ["resolve_place", "search_scenes"]


def test_the_server_binds_to_this_machine_at_the_configured_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served: list[tuple[object, str, int]] = []

    def run(app: object, *, host: str, port: int) -> None:
        served.append((app, host, port))

    monkeypatch.chdir(tmp_path)  # Settings reads `.env` from the current folder: here there is none
    monkeypatch.setenv("EO_A2A_URL", "http://localhost:9100")
    monkeypatch.setattr(uvicorn, "run", run)
    a2a_server.main()
    ((app, host, port),) = served
    assert (host, port) == ("127.0.0.1", 9100)
    assert isinstance(app, Starlette)
