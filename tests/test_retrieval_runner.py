"""Specification for run_retrieval: one delegated turn, from the request to the result.

The real MCP server runs in-process and is reached over HTTP, as the agent reaches it in production, so the
trace header and the tool-call log are those of a real run. The model is scripted, or is the real ChatOllama
client talking to a scripted Ollama server, so its errors are the ones that client raises. The catalogue is
fake.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import ollama
import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED, ErrorData
from starlette.applications import Starlette

from agentic_earth_observation.catalog.gazetteer import Gazetteer
from agentic_earth_observation.catalog.stac import SceneSummary
from agentic_earth_observation.catalog_mcp.server import build_server
from agentic_earth_observation.catalog_mcp.tool_log import TRACE_HEADER, ToolCallLog, read_tool_calls
from agentic_earth_observation.config import Settings
from agentic_earth_observation.domain import SearchRequest
from agentic_earth_observation.retrieval_agent.agent import AGENT_TOOLS, _infrastructure_error, run_retrieval
from agentic_earth_observation.retrieval_agent.models import PreviousSearch, RetrievalRequest, RetrievalResult
from agentic_earth_observation.retrieval_agent.prompts import retrieval_system_prompt, retrieval_user_message
from tests.fakes import (
    CONNECTION_REFUSED,
    SCENE,
    FakeCatalog,
    FakeOllama,
    OllamaError,
    ScriptedModel,
    ollama_text,
    ollama_tool_call,
    tool_call,
)

pytestmark = pytest.mark.anyio

TODAY = date(2025, 12, 15)
REQUEST = RetrievalRequest(text="Milan, March 2025, little cloud", today=TODAY)
MILAN_BOX = [9.0409, 45.3867, 9.2781, 45.5358]
NAPLES_BOX = [14.1321, 40.7912, 14.3538, 40.9161]
MILAN_MARCH = {
    "product": "sentinel-2-l2a",
    "bbox": MILAN_BOX,
    "start_date": "2025-03-01",
    "end_date": "2025-03-31",
    "max_cloud_cover": 20,
}
TOKENS = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
Fault = Callable[[httpx.Request], httpx.Response | None]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def settings(**changes: Any) -> Settings:  # noqa: ANN401 - any setting, validated by Settings itself
    return Settings(_env_file=None, **changes)  # type: ignore[call-arg]  # a pydantic-settings init option


def quick() -> Settings:
    """Settings with short timeouts, still nested as Settings requires."""
    return settings(
        stac_timeout_s=0.1, stac_deadline_s=0.2, mcp_timeout_s=0.3, model_timeout_s=0.3, turn_deadline_s=0.6
    )


def on_method(method: str, status: int, body: dict[str, Any]) -> Fault:
    """A fault that answers one JSON-RPC method with `status` and `body` instead of the server."""

    def fault(request: httpx.Request) -> httpx.Response | None:
        sent = json.loads(request.content) if request.content else None
        return (
            httpx.Response(status, json=body)
            if isinstance(sent, dict) and sent.get("method") == method
            else None
        )

    return fault


class _Faulty(httpx.AsyncBaseTransport):
    def __init__(
        self, inner: httpx.AsyncBaseTransport, fault: Fault | None, close_bug: Exception | None = None
    ) -> None:
        self.inner = inner
        self.fault = fault
        self.close_bug = close_bug

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.fault is not None and (response := self.fault(request)) is not None:
            return response
        return await self.inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self.inner.aclose()
        if self.close_bug is not None:
            raise self.close_bug


class InProcessClient:
    """Builds HTTP clients that reach the in-process MCP server, and records how each one was built."""

    def __init__(self, app: Starlette, fault: Fault | None, close_bug: Exception | None) -> None:
        self.app = app
        self.fault = fault
        self.close_bug = close_bug
        self.built: list[tuple[dict[str, str] | None, httpx.Timeout | None]] = []

    def __call__(
        self,
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        self.built.append((headers, timeout))
        transport = _Faulty(httpx.ASGITransport(app=self.app), self.fault, self.close_bug)
        return httpx.AsyncClient(transport=transport, headers=headers, timeout=timeout, auth=auth)


@asynccontextmanager
async def mcp_server(
    log: Path,
    catalog: FakeCatalog | None = None,
    *,
    without: Sequence[str] = (),
    fault: Fault | None = None,
    close_bug: Exception | None = None,
) -> AsyncIterator[InProcessClient]:
    """The real MCP server in-process; yields the HTTP client factory that reaches it.

    `without` removes tools, `fault` replaces some of the server's answers, and `close_bug` is raised when an
    HTTP connection to it is closed.
    """
    server = build_server(catalog or FakeCatalog(), Gazetteer.load(), ToolCallLog(log))
    for name in without:
        server.remove_tool(name)
    app = server.streamable_http_app()  # before its session manager, which it creates
    async with server.session_manager.run():
        yield InProcessClient(app, fault, close_bug)


async def run_model(
    model: BaseChatModel,
    log: Path,
    *,
    request: RetrievalRequest = REQUEST,
    catalog: FakeCatalog | None = None,
    config: Settings | None = None,
    trace_id: str = "turn-1",
) -> RetrievalResult:
    async with mcp_server(log, catalog) as client:
        return await run_retrieval(request, trace_id, model, config or settings(), http_client=client)


async def run(
    script: Sequence[AIMessage],
    log: Path,
    **options: Any,  # noqa: ANN401 - the options of run_model
) -> tuple[RetrievalResult, ScriptedModel]:
    model = ScriptedModel(messages=iter(script))
    return await run_model(model, log, **options), model


def with_tokens(message: AIMessage) -> AIMessage:
    return message.model_copy(update={"usage_metadata": TOKENS})


def raw_tools(log: Path) -> list[str]:
    """The tools in the order their calls finished, as the log file has them."""
    return [json.loads(line)["tool"] for line in log.read_text(encoding="utf-8").splitlines()]


class BrokenModel(ScriptedModel):
    """A model wrapper with a bug in it: every call raises `bug`."""

    bug: Exception = ValueError("a bug in the model wrapper")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,  # noqa: ANN401 - the signature of BaseChatModel._agenerate
    ) -> ChatResult:
        raise self.bug


class HangingModel(ScriptedModel):
    """Gives its scripted replies, then never answers again."""

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,  # noqa: ANN401 - the signature of BaseChatModel._agenerate
    ) -> ChatResult:
        try:
            return self._generate(messages, stop=stop)
        except StopIteration:
            await asyncio.Event().wait()
            raise AssertionError("unreachable") from None


class WaitsForResolve(FakeCatalog):
    """Searches only once resolve_place is in the log, so that a search sent first finishes last."""

    def __init__(self, log: Path) -> None:
        super().__init__()
        self.log = log

    def search(self, request: SearchRequest) -> list[SceneSummary]:
        give_up = time.monotonic() + 5
        while '"resolve_place"' not in self.log.read_text(encoding="utf-8"):
            if time.monotonic() > give_up:
                raise AssertionError("resolve_place was never recorded")
            time.sleep(0.01)  # in the server's worker thread, not the event loop
        return super().search(request)


class TestTurns:
    async def test_a_search_turn_reports_what_the_tools_returned(self, tmp_path: Path) -> None:
        script = [
            with_tokens(tool_call("resolve_place", name="Milan")),
            with_tokens(tool_call("search_scenes", "call-2", **MILAN_MARCH)),
            with_tokens(AIMessage("One scene was found.")),
        ]
        started = time.perf_counter()
        result, _ = await run(script, tmp_path / "calls.jsonl")
        elapsed = time.perf_counter() - started
        assert (result.status, result.scenes, result.failure, result.infra_error) == (
            "ok",
            [SCENE],
            None,
            None,
        )
        assert result.search is not None
        assert (result.search.place, result.search.bbox, result.search.max_cloud_cover) == (
            "Milan",
            MILAN_BOX,
            20,
        )
        assert (result.usage.model, result.usage.model_calls) == ("qwen3.5:9b", 3)
        assert (result.usage.input_tokens, result.usage.output_tokens) == (300, 30)
        assert 0 < result.usage.seconds <= elapsed

    async def test_the_real_ollama_client_runs_a_whole_turn_with_only_the_agents_tools(
        self, tmp_path: Path
    ) -> None:
        server = FakeOllama(
            ollama_tool_call("resolve_place", name="Milan"),
            ollama_tool_call("search_scenes", **MILAN_MARCH),
            ollama_text("One scene was found."),
        )
        result = await run_model(server.model(), tmp_path / "calls.jsonl")
        assert (result.status, result.scenes) == ("ok", [SCENE])
        assert (result.usage.model_calls, result.usage.input_tokens, result.usage.output_tokens) == (
            3,
            300,
            30,
        )
        # Ollama's own clock, summed over the three replies: the fake reports 0.1 s load, 0.5 s work each.
        assert result.usage.model_seconds == pytest.approx(1.5)
        assert result.usage.load_seconds == pytest.approx(0.3)
        offered = [sorted(t["function"]["name"] for t in sent["tools"]) for sent in server.requests]
        assert offered == [sorted(AGENT_TOOLS)] * 3

    async def test_the_model_reads_the_dated_prompt_then_the_request(self, tmp_path: Path) -> None:
        _, model = await run([AIMessage("I can only search scene metadata.")], tmp_path / "calls.jsonl")
        # Compared by type and text: the graph gives each message an id.
        assert [(type(m), m.content) for m in model.seen[0]] == [
            (SystemMessage, retrieval_system_prompt(TODAY)),
            (HumanMessage, retrieval_user_message(REQUEST)),
        ]

    @pytest.mark.parametrize("trace_id", ["run-7.task-3.turn-1", "a" * 64])
    async def test_every_call_is_filed_under_the_turns_trace_id(self, tmp_path: Path, trace_id: str) -> None:
        log = tmp_path / "calls.jsonl"
        script = [
            tool_call("resolve_place", name="Milan"),
            tool_call("search_scenes", "call-2", **MILAN_MARCH),
            AIMessage("One scene was found."),
        ]
        await run(script, log, trace_id=trace_id)
        calls = read_tool_calls(log, trace_id)
        assert [c.tool for c in calls] == ["resolve_place", "search_scenes"]
        assert calls[1].returned_ids == [SCENE.id]
        assert read_tool_calls(log, "turn-1") == []

    async def test_the_http_client_gets_the_trace_header_and_the_mcp_timeouts(self, tmp_path: Path) -> None:
        config = settings(mcp_timeout_s=100)
        model = ScriptedModel(messages=iter([AIMessage("I can only search scene metadata.")]))
        async with mcp_server(tmp_path / "calls.jsonl") as client:
            await run_retrieval(REQUEST, "turn-9", model, config, http_client=client)
        assert client.built == [({TRACE_HEADER: "turn-9"}, httpx.Timeout(100, read=100))]

    async def test_calls_sent_together_are_read_back_in_the_models_order(self, tmp_path: Path) -> None:
        # The graph runs both calls of one reply at once; the search, asked first, is made to finish last.
        log = tmp_path / "calls.jsonl"
        both = AIMessage(
            content="",
            tool_calls=[
                {"name": "search_scenes", "args": MILAN_MARCH, "id": "call-1", "type": "tool_call"},
                {"name": "resolve_place", "args": {"name": "Milan"}, "id": "call-2", "type": "tool_call"},
            ],
        )
        await run([both, AIMessage("One scene was found.")], log, catalog=WaitsForResolve(log))
        assert raw_tools(log) == ["resolve_place", "search_scenes"]  # the order they finished
        assert [c.tool for c in read_tool_calls(log, "turn-1")] == ["search_scenes", "resolve_place"]

    async def test_a_follow_up_sees_the_previous_search_and_keeps_its_place(self, tmp_path: Path) -> None:
        naples_week = PreviousSearch.model_validate_json(
            json.dumps(
                {
                    "place": "Naples",
                    "product": "sentinel-1-grd",
                    "bbox": NAPLES_BOX,
                    "start_date": "2025-09-01",
                    "end_date": "2025-09-07",
                }
            )
        )
        follow_up = RetrievalRequest(
            text="And the week after that?", today=TODAY, previous_search=naples_week
        )
        week_after = {
            "product": "sentinel-1-grd",
            "bbox": NAPLES_BOX,
            "start_date": "2025-09-08",
            "end_date": "2025-09-14",
        }
        script = [tool_call("search_scenes", **week_after), AIMessage("One scene was found.")]
        result, model = await run(script, tmp_path / "calls.jsonl", request=follow_up)
        assert "PREVIOUS SEARCH: place=Naples" in str(model.seen[0][1].content)
        assert result.search is not None
        assert (result.search.place, result.search.start_date) == ("Naples", date(2025, 9, 8))


class TestModelFailures:
    """The model's own failures fail the task; they are never infrastructure."""

    async def test_running_out_of_steps(self, tmp_path: Path) -> None:
        script = [tool_call("resolve_place", f"call-{i}", name="Milan") for i in range(10)]
        result, _ = await run(script, tmp_path / "calls.jsonl", config=settings(max_model_calls=4))
        assert (result.status, result.failure, result.infra_error) == ("not_searched", "step_limit", None)
        assert result.usage.model_calls == 4

    async def test_a_tool_the_agent_was_not_given_never_runs(self, tmp_path: Path) -> None:
        log = tmp_path / "calls.jsonl"
        catalog = FakeCatalog()
        script = [
            tool_call("get_scene", product="sentinel-2-l2a", scene_id=SCENE.id),
            AIMessage("never reached"),
        ]
        result, _ = await run(script, log, catalog=catalog)
        assert (result.status, result.failure, result.infra_error) == (
            "not_searched",
            "malformed_tool_call",
            None,
        )
        assert (catalog.lookups, read_tool_calls(log, "turn-1")) == ([], [])

    async def test_a_reply_the_model_server_cannot_read(self, tmp_path: Path) -> None:
        detail = "XML syntax error on line 1: unexpected EOF"
        server = FakeOllama(ollama_tool_call("resolve_place", name="Milan"), OllamaError(500, detail))
        result = await run_model(server.model(), tmp_path / "calls.jsonl")
        assert (result.status, result.failure, result.failure_detail) == (
            "not_searched",
            "malformed_tool_call",
            detail,
        )
        assert result.infra_error is None
        assert (result.usage.model_calls, result.usage.input_tokens, result.usage.output_tokens) == (
            2,
            100,
            10,
        )


class TestToolErrors:
    """A tool's error goes to the model, and the turn ends `error` unless a search later succeeds."""

    async def test_a_refused_place_lookup(self, tmp_path: Path) -> None:
        log = tmp_path / "calls.jsonl"
        script = [
            tool_call("resolve_place", name="Milan", country="IT"),
            AIMessage("I could not look it up."),
        ]
        result, _ = await run(script, log)
        assert (result.status, result.failure, result.infra_error) == ("error", None, None)
        assert [c.error_kind for c in read_tool_calls(log, "turn-1")] == ["invalid_arguments"]

    async def test_a_catalogue_failure(self, tmp_path: Path) -> None:
        # The agent reports `error`. The evaluation runner counts it as infrastructure from the log's error
        # kind, the only reliable source: the model sees just a fixed message.
        log = tmp_path / "calls.jsonl"
        script = [
            tool_call("resolve_place", name="Milan"),
            tool_call("search_scenes", "call-2", **MILAN_MARCH),
            AIMessage("The catalogue did not answer."),
        ]
        result, _ = await run(script, log, catalog=FakeCatalog(fail=True))
        assert (result.status, result.failure, result.infra_error) == ("error", None, None)
        assert [c.error_kind for c in read_tool_calls(log, "turn-1")] == [None, "catalogue"]


class TestInfrastructure:
    """Failures outside the model's control are reported, so that the task can be retried and excluded."""

    async def test_a_turn_past_its_deadline(self, tmp_path: Path) -> None:
        config = quick()
        model = HangingModel(messages=iter([with_tokens(tool_call("resolve_place", name="Milan"))]))
        async with mcp_server(tmp_path / "calls.jsonl") as client:
            started = time.perf_counter()
            result = await run_retrieval(REQUEST, "turn-1", model, config, http_client=client)
            elapsed = time.perf_counter() - started
        assert (result.status, result.failure) == ("not_searched", None)
        assert result.infra_error == "the turn ran past its deadline (0.6 s)"
        assert config.turn_deadline_s <= elapsed < config.turn_deadline_s + 0.5  # enforced, and on time
        # What the model had already spent is kept: the hanging call is the second.
        assert (result.usage.model, result.usage.model_calls) == ("qwen3.5:9b", 2)
        assert (result.usage.input_tokens, result.usage.output_tokens) == (100, 10)
        assert config.turn_deadline_s <= result.usage.seconds <= elapsed

    async def test_an_mcp_server_that_cannot_be_reached(self) -> None:
        def refusing(
            headers: dict[str, str] | None = None,
            timeout: httpx.Timeout | None = None,
            auth: httpx.Auth | None = None,
        ) -> httpx.AsyncClient:
            def refuse(request: httpx.Request) -> httpx.Response:
                raise httpx.ConnectError("connection refused", request=request)

            return httpx.AsyncClient(transport=httpx.MockTransport(refuse), headers=headers, timeout=timeout)

        model = ScriptedModel(messages=iter([AIMessage("never reached")]))
        result = await run_retrieval(REQUEST, "turn-1", model, settings(), http_client=refusing)
        assert result.infra_error == "ConnectError: connection refused"
        assert (result.status, result.failure, model.seen) == ("not_searched", None, [])

    async def test_an_mcp_answer_the_client_cannot_read_times_out_in_time(self, tmp_path: Path) -> None:
        # Without its own timeout the session would wait for the turn's deadline instead.
        config = quick()
        garbled = on_method("tools/list", 200, {"detail": "not JSON-RPC"})
        model = ScriptedModel(messages=iter([AIMessage("never reached")]))
        async with mcp_server(tmp_path / "calls.jsonl", fault=garbled) as client:
            started = time.perf_counter()
            result = await run_retrieval(REQUEST, "turn-1", model, config, http_client=client)
            elapsed = time.perf_counter() - started
        assert result.infra_error is not None
        assert result.infra_error.startswith("McpError: Timed out while waiting for response")
        assert config.mcp_timeout_s <= elapsed < config.turn_deadline_s

    async def test_a_busy_model_server(self, tmp_path: Path) -> None:
        busy = OllamaError(503, "server busy, please try again.  maximum pending requests exceeded")
        result = await run_model(FakeOllama(busy).model(), tmp_path / "calls.jsonl")
        assert result.infra_error == (
            "ResponseError: server busy, please try again. maximum pending requests exceeded"
            " (status code: 503)"
        )
        assert (result.status, result.failure, result.usage.model_calls) == ("not_searched", None, 1)

    async def test_a_model_server_that_cannot_be_reached(self, tmp_path: Path) -> None:
        result = await run_model(FakeOllama(CONNECTION_REFUSED).model(), tmp_path / "calls.jsonl")
        assert result.infra_error == "ConnectError: All connection attempts failed"
        assert (result.status, result.failure) == ("not_searched", None)


class TestRaised:
    """Bugs and configuration errors are raised, never counted as infrastructure: infrastructure is excluded
    from the scores, so a bug counted as infrastructure would shrink them unseen."""

    @pytest.mark.parametrize("bug", [ValueError("a bug in the model wrapper"), PermissionError("a file")])
    async def test_a_bug(self, tmp_path: Path, bug: Exception) -> None:
        async with mcp_server(tmp_path / "calls.jsonl") as client:
            model = BrokenModel(messages=iter([]), bug=bug)
            with pytest.raises(type(bug)) as raised:
                await run_retrieval(REQUEST, "turn-1", model, settings(), http_client=client)
        assert raised.value is bug

    async def test_a_bug_while_the_deadline_cancels_the_turn(self, tmp_path: Path) -> None:
        # Here in closing the MCP connection. (LangGraph itself drops an error raised inside a node while the
        # node is being cancelled, so a bug there cannot reach the runner.)
        model = HangingModel(messages=iter([]))
        async with mcp_server(tmp_path / "calls.jsonl", close_bug=KeyError("a bug while closing")) as client:
            with pytest.raises(KeyError, match="a bug while closing"):
                await run_retrieval(REQUEST, "turn-1", model, quick(), http_client=client)

    async def test_a_model_the_model_server_does_not_have(self, tmp_path: Path) -> None:
        model = FakeOllama(OllamaError(404, 'model "qwen3.5:9b" not found, try pulling it first')).model()
        async with mcp_server(tmp_path / "calls.jsonl") as client:
            with pytest.raises(ollama.ResponseError, match="not found"):
                await run_retrieval(REQUEST, "turn-1", model, settings(), http_client=client)

    async def test_an_http_error_from_our_mcp_server(self, tmp_path: Path) -> None:
        broken = on_method("tools/list", 500, {"error": "Internal Server Error"})
        model = ScriptedModel(messages=iter([AIMessage("never reached")]))
        async with mcp_server(tmp_path / "calls.jsonl", fault=broken) as client:
            with pytest.raises(httpx.HTTPStatusError, match="500"):
                await run_retrieval(REQUEST, "turn-1", model, quick(), http_client=client)

    async def test_a_wrong_mcp_url(self, tmp_path: Path) -> None:
        config = quick().model_copy(update={"mcp_url": "http://127.0.0.1:8001/wrong"})
        model = ScriptedModel(messages=iter([AIMessage("never reached")]))
        async with mcp_server(tmp_path / "calls.jsonl") as client:
            with pytest.raises(McpError, match="Session terminated"):
                await run_retrieval(REQUEST, "turn-1", model, config, http_client=client)

    async def test_a_server_without_the_agents_tools(self, tmp_path: Path) -> None:
        model = ScriptedModel(messages=iter([AIMessage("never reached")]))
        async with mcp_server(tmp_path / "calls.jsonl", without=["search_scenes"]) as client:
            with pytest.raises(RuntimeError, match=r"lacks the tool\(s\) search_scenes"):
                await run_retrieval(REQUEST, "turn-1", model, settings(), http_client=client)
        assert model.seen == []

    @pytest.mark.parametrize("trace_id", ["", "turn 1", "turn-1\n", "a" * 65, "run-3/turn-1", "türn-1"])
    async def test_a_malformed_trace_id_before_anything_is_sent(self, tmp_path: Path, trace_id: str) -> None:
        # The server would record such a header as no trace, and the turn's calls would be lost to scoring.
        log = tmp_path / "calls.jsonl"
        model = ScriptedModel(messages=iter([AIMessage("never reached")]))
        async with mcp_server(log) as client:
            with pytest.raises(ValueError, match="invalid trace id"):
                await run_retrieval(REQUEST, trace_id, model, settings(), http_client=client)
        assert (log.read_text(encoding="utf-8"), model.seen) == ("", [])


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://127.0.0.1:8001/mcp")
    return httpx.HTTPStatusError(
        f"HTTP {status}", request=request, response=httpx.Response(status, request=request)
    )


INFRASTRUCTURE = [
    TimeoutError(),
    ConnectionRefusedError("Failed to connect to Ollama"),
    httpx.ConnectError("All connection attempts failed"),
    httpx.ReadTimeout(""),
    httpx.RemoteProtocolError("peer closed connection without sending complete message body"),
    McpError(ErrorData(code=408, message="Timed out while waiting for response")),
    McpError(ErrorData(code=CONNECTION_CLOSED, message="Connection closed")),
    ollama.ResponseError("server busy", 503),
]
NOT_INFRASTRUCTURE = [
    ValueError("a bug"),
    PermissionError("a file we cannot read"),
    httpx.UnsupportedProtocol("Request URL is missing an 'http://' or 'https://' protocol."),
    _http_error(500),  # from our own MCP server: a bug in it
    _http_error(421),  # a Host header the server refuses: a configuration error
    McpError(ErrorData(code=32600, message="Session terminated")),  # a wrong MCP URL
    McpError(ErrorData(code=-32603, message="Internal error")),
    ollama.ResponseError("model not found", 404),
    ollama.ResponseError("XML syntax error", 500),  # the model's failure, which the graph handles first
]


class TestErrorSorting:
    @pytest.mark.parametrize("err", INFRASTRUCTURE, ids=repr)
    async def test_failures_outside_the_models_control_are_infrastructure(self, err: Exception) -> None:
        assert _infrastructure_error(err).startswith(type(err).__name__)
        assert _infrastructure_error(ExceptionGroup("session", [ExceptionGroup("transport", [err])]))

    @pytest.mark.parametrize("err", NOT_INFRASTRUCTURE, ids=repr)
    async def test_bugs_and_configuration_errors_are_raised_as_themselves(self, err: Exception) -> None:
        for wrapped in (err, ExceptionGroup("session", [ExceptionGroup("graph", [err])])):
            with pytest.raises(type(err)) as raised:
                _infrastructure_error(wrapped)
            assert raised.value is err

    async def test_one_bug_among_infrastructure_errors_is_raised_with_its_own_cause(self) -> None:
        bug = RuntimeError("the graph broke")
        bug.__cause__ = KeyError("state")
        mixed = ExceptionGroup("session", [httpx.ConnectError("refused"), ExceptionGroup("graph", [bug])])
        with pytest.raises(RuntimeError) as raised:
            _infrastructure_error(mixed)
        assert raised.value is bug
        assert isinstance(raised.value.__cause__, KeyError)

    async def test_several_bugs_are_raised_together(self) -> None:
        bugs = ExceptionGroup("session", [KeyError("x"), ValueError("y")])
        with pytest.raises(ExceptionGroup) as raised:
            _infrastructure_error(bugs)
        assert raised.value is bugs

    @pytest.mark.parametrize(
        ("err", "text"),
        [
            (httpx.ReadTimeout(""), "ReadTimeout"),  # no message of its own
            (
                httpx.ConnectError("All connection\n  attempts failed"),
                "ConnectError: All connection attempts failed",
            ),
            (httpx.ConnectError("x" * 500), "ConnectError: " + "x" * 286),  # 300 characters at most
        ],
    )
    async def test_an_infrastructure_error_is_described_on_one_short_line(
        self, err: Exception, text: str
    ) -> None:
        assert _infrastructure_error(err) == text
