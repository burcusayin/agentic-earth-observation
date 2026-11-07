"""Tests for the tool-call record kept by the MCP server: contents, error kinds and the trace header."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import TextContent
from pydantic import BaseModel, ValidationError

from agentic_earth_observation.catalog.gazetteer import Gazetteer
from agentic_earth_observation.catalog.stac import CatalogError, CatalogRefused, SceneSummary
from agentic_earth_observation.catalog_mcp.server import UNRECORDED, build_server
from agentic_earth_observation.catalog_mcp.tool_log import (
    TRACE_HEADER,
    ToolCallLog,
    ToolCallRecord,
    error_kind,
    read_log,
    read_tool_calls,
    returned_ids,
)
from agentic_earth_observation.domain import SearchRequest
from tests.fakes import ROME_JULY, SCENE, BlockingCatalog, FakeCatalog

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def server_with_log(path: Path, catalog: FakeCatalog | None = None) -> FastMCP:
    return build_server(catalog or FakeCatalog(), Gazetteer.load(), ToolCallLog(path))


async def call(mcp: FastMCP, tool: str, arguments: dict[str, Any]) -> None:
    async with create_connected_server_and_client_session(mcp) as session:
        await session.call_tool(tool, arguments)


def all_records(path: Path) -> list[dict[str, Any]]:
    import json

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def record(trace_id: str | None = "turn-1", request_id: int = 1, name: str = "Milan") -> ToolCallRecord:
    """A successful resolve_place call, written straight to a log."""
    return ToolCallRecord(
        trace_id=trace_id, request_id=request_id, tool="resolve_place", arguments={"name": name}, ok=True
    )


class FirstFinishesLastCatalog(FakeCatalog):
    """The first search waits until the second has finished, so two calls finish in reverse order."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._second_done = threading.Event()

    def search(self, request: SearchRequest) -> list[SceneSummary]:
        with self._lock:
            first = not self.requests
            scenes = super().search(request)
        if first:
            self._second_done.wait(timeout=5)
        else:
            self._second_done.set()
        return scenes


async def test_a_successful_search_is_recorded_with_its_scene_ids(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    await call(server_with_log(log), "search_scenes", ROME_JULY)
    (record,) = all_records(log)
    assert record["tool"] == "search_scenes"
    assert record["ok"] is True
    assert record["returned_ids"] == [SCENE.id]
    assert record["arguments"]["max_cloud_cover"] == 10
    assert record["trace_id"] is None  # no HTTP request in an in-process session
    assert isinstance(record["request_id"], int)


@pytest.mark.parametrize(
    ("arguments", "kind"),
    [
        ({**ROME_JULY, "bbox": [12.8, 41.6, 12.2, 42.1]}, "invalid_arguments"),  # rejected by the validators
        ({**ROME_JULY, "cloud_cover": 5}, "invalid_arguments"),  # unknown argument name
        ({**ROME_JULY, "max_cloud_cover": "20"}, "invalid_arguments"),  # wrong type
        ({**ROME_JULY, "bbox": "[12.2344, 41.6556, 12.8558, 42.141]"}, "invalid_arguments"),  # a list as text
    ],
)
async def test_the_models_mistakes_are_recorded_as_invalid_arguments(
    tmp_path: Path, arguments: dict[str, Any], kind: str
) -> None:
    log = tmp_path / "calls.jsonl"
    await call(server_with_log(log), "search_scenes", arguments)
    (record,) = all_records(log)
    assert record["ok"] is False
    assert record["error_kind"] == kind
    assert record["arguments"] == arguments  # exactly what the model sent


async def test_an_unknown_tool_is_recorded_as_such(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    await call(server_with_log(log), "get_weather", {"place": "Napoli"})
    (record,) = all_records(log)
    assert (record["tool"], record["ok"], record["error_kind"]) == ("get_weather", False, "unknown_tool")


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [("search_scenes", ROME_JULY), ("get_scene", {"product": "sentinel-2-l2a", "scene_id": SCENE.id})],
)
async def test_a_catalogue_failure_is_recorded_as_infrastructure(
    tmp_path: Path, tool: str, arguments: dict[str, Any]
) -> None:
    log = tmp_path / "calls.jsonl"
    await call(server_with_log(log, FakeCatalog(fail=True)), tool, arguments)
    (record,) = all_records(log)
    assert record["ok"] is False
    assert record["error_kind"] == "catalogue"


async def test_a_missed_catalogue_deadline_is_recorded_as_infrastructure(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    catalog = BlockingCatalog()
    mcp = build_server(catalog, Gazetteer.load(), ToolCallLog(log), catalog_deadline_s=0.05)
    try:
        await call(mcp, "search_scenes", ROME_JULY)
    finally:
        catalog.release.set()
    (record,) = all_records(log)
    assert (record["ok"], record["error_kind"]) == (False, "catalogue")


async def test_a_cancelled_call_is_recorded(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    catalog = BlockingCatalog()
    task = asyncio.create_task(server_with_log(log, catalog).call_tool("search_scenes", ROME_JULY))
    try:
        assert await asyncio.to_thread(catalog.started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        catalog.release.set()
    (record,) = all_records(log)
    assert (record["ok"], record["error_kind"]) == (False, "cancelled")


async def test_calls_that_finish_out_of_order_keep_the_order_they_were_sent_in(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    async with create_connected_server_and_client_session(
        server_with_log(log, FirstFinishesLastCatalog())
    ) as session:
        await asyncio.gather(
            session.call_tool("search_scenes", {**ROME_JULY, "limit": 1}),
            session.call_tool("search_scenes", {**ROME_JULY, "limit": 2}),
        )
    written = all_records(log)
    assert [r["arguments"]["limit"] for r in written] == [2, 1]  # written as they finished
    sent = sorted(written, key=lambda r: int(r["request_id"]))
    assert [r["arguments"]["limit"] for r in sent] == [1, 2]


async def test_every_tool_is_recorded_even_without_scenes(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    await call(server_with_log(log), "resolve_place", {"name": "Napoli"})
    (record,) = all_records(log)
    assert record["tool"] == "resolve_place"
    assert record["ok"] is True
    assert record["returned_ids"] == []


async def test_the_trace_header_files_calls_under_their_turn(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    mcp = server_with_log(log)
    app = mcp.streamable_http_app()

    async def call_with_trace(trace: str) -> None:
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), headers={TRACE_HEADER: trace})
        async with (
            client,
            streamable_http_client("http://127.0.0.1:8001/mcp", http_client=client) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            await session.call_tool("resolve_place", {"name": "Milan"})

    async with mcp.session_manager.run():
        await call_with_trace("turn-1")
        await call_with_trace("has spaces; not a valid id")  # untrusted header: refused, recorded as None

    assert [r.tool for r in read_tool_calls(log, "turn-1")] == ["resolve_place"]
    assert [r["trace_id"] for r in all_records(log)] == ["turn-1", None]


@pytest.mark.parametrize(
    ("header", "accepted"),
    [("a" * 64, True), ("a" * 65, False), ("turn-1\n", False), ("turn 1", False), ("run-3/turn-1", False)],
)
async def test_only_plain_trace_ids_of_up_to_64_characters_are_accepted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, header: str, accepted: bool
) -> None:
    log = tmp_path / "calls.jsonl"
    mcp = server_with_log(log)
    transport = httpx.ASGITransport(app=mcp.streamable_http_app())
    client = httpx.AsyncClient(transport=transport, headers={TRACE_HEADER: header})
    async with (
        mcp.session_manager.run(),
        client,
        streamable_http_client("http://127.0.0.1:8001/mcp", http_client=client) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        await session.call_tool("resolve_place", {"name": "Milan"})
    (written,) = all_records(log)
    assert written["trace_id"] == (header if accepted else None)
    assert ("refused an invalid X-EO-Trace header" in caplog.text) is not accepted


def test_reading_returns_a_trace_in_the_order_the_calls_were_sent(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    log = ToolCallLog(path)
    for request_id, trace in ((3, "turn-1"), (1, "turn-1"), (2, "turn-2"), (2, "turn-1")):
        log.write(record(trace, request_id))
    assert [r.request_id for r in read_tool_calls(path, "turn-1")] == [1, 2, 3]


def test_a_last_line_still_being_written_is_not_a_record_yet(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    ToolCallLog(path).write(record())
    with path.open("a", encoding="utf-8") as log:
        log.write('{"trace_id": "turn-1", "req')
    assert [r.trace_id for r in read_log(path)] == ["turn-1"]


def test_a_catalogue_that_refuses_our_request_is_not_counted_as_a_failed_catalogue() -> None:
    assert error_kind(CatalogError("the catalogue did not answer")) == "catalogue"
    assert error_kind(CatalogRefused("the catalogue did not answer correctly (HTTP 400)")) == "other"


async def test_arguments_that_json_cannot_hold_are_reported_and_do_not_escape(tmp_path: Path) -> None:
    mcp = build_server(FakeCatalog(), Gazetteer.load(), ToolCallLog(tmp_path / "calls.jsonl"))
    async with create_connected_server_and_client_session(mcp) as session:
        result = await session.call_tool("resolve_place", {"name": "\ud800"})
    assert result.isError
    assert UNRECORDED in "".join(c.text for c in result.content if isinstance(c, TextContent))


def test_reading_the_whole_log_gives_every_call_as_it_was_written(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    log = ToolCallLog(path)
    for request_id, trace in ((3, "turn-1"), (1, None), (2, "turn-2"), (2, "turn-1")):
        log.write(record(trace, request_id))
    assert [(r.trace_id, r.request_id) for r in read_log(path)] == [
        ("turn-1", 3),
        (None, 1),
        ("turn-2", 2),
        ("turn-1", 2),
    ]


SEPARATED = "Mil\u2028an\u0085o\u2029"  # str.splitlines() would split on each of these


def test_unusual_line_separators_in_arguments_do_not_split_a_record(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    log = ToolCallLog(path)
    log.write(record(name=SEPARATED))
    log.write(record(request_id=2, name="Napoli"))
    assert [r.arguments["name"] for r in read_tool_calls(path, "turn-1")] == [SEPARATED, "Napoli"]


def test_blank_lines_are_skipped_and_a_corrupt_line_is_named(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    path.write_text(record().model_dump_json() + "\n\n" + '{"oops"\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"calls\.jsonl:3: not a tool-call record"):
        read_tool_calls(path, "turn-1")


def test_reading_a_log_that_does_not_exist_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="EO_TOOL_LOG"):
        read_tool_calls(tmp_path / "missing.jsonl", "turn-1")


def test_the_log_file_and_its_folders_are_created_when_the_log_is(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "calls.jsonl"
    ToolCallLog(path)
    assert path.is_file()
    assert read_tool_calls(path, "turn-1") == []


def test_a_log_path_that_cannot_be_written_stops_the_server_at_start(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot write the tool-call log"):
        ToolCallLog(tmp_path)  # a directory


async def test_a_record_that_cannot_be_written_never_shows_the_path_or_hides_the_tools_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "calls.jsonl"
    mcp = server_with_log(path)
    path.unlink()
    path.mkdir()  # from now on every write fails
    async with create_connected_server_and_client_session(mcp) as session:
        succeeded = await session.call_tool("resolve_place", {"name": "Milan"})
        failed = await session.call_tool("search_scenes", {**ROME_JULY, "bbox": [12.8, 41.6, 12.2, 42.1]})
    texts = ["".join(c.text for c in r.content if isinstance(c, TextContent)) for r in (succeeded, failed)]
    assert succeeded.isError
    assert UNRECORDED in texts[0]
    assert failed.isError
    assert "west must be less than east" in texts[1]  # the tool's own error, not the recording failure
    assert all(str(tmp_path) not in text for text in texts)
    assert "could not write the tool-call log" in caplog.text


def test_error_kind_follows_the_cause_chain() -> None:
    class Model(BaseModel):
        x: int

    try:
        Model.model_validate({"x": "no"})
    except ValidationError as validation:
        wrapped = RuntimeError("Error executing tool")
        wrapped.__cause__ = validation
        assert error_kind(wrapped) == "invalid_arguments"
    catalogue = RuntimeError("Error executing tool")
    catalogue.__cause__ = CatalogError("scene search failed")
    assert error_kind(catalogue) == "catalogue"
    assert error_kind(RuntimeError("something else")) == "other"


def test_returned_ids_reads_searches_and_lookups() -> None:
    assert returned_ids(([], {"count": 1, "scenes": [{"id": "A"}, {"id": "B"}]})) == ["A", "B"]
    assert returned_ids(([], {"found": True, "scene": {"id": "A"}})) == ["A"]
    assert returned_ids(([], {"found": False, "scene": None})) == []
    assert returned_ids("not structured") == []


def test_a_log_without_a_path_writes_nothing_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    log = ToolCallLog(None)
    assert "tool-call log is off" in caplog.text
    log.write(record())  # no file, no error
