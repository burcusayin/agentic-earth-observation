"""Tests for the tool-call record kept by the MCP server: contents, error kinds and the trace header."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import BaseModel, ValidationError

from agentic_earth_observation.catalog.gazetteer import Gazetteer
from agentic_earth_observation.catalog.stac import CatalogError
from agentic_earth_observation.catalog_mcp.server import build_server
from agentic_earth_observation.catalog_mcp.tool_log import (
    TRACE_HEADER,
    ToolCallLog,
    error_kind,
    read_tool_calls,
    returned_ids,
)
from tests.fakes import ROME_JULY, SCENE, FakeCatalog

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


async def test_a_successful_search_is_recorded_with_its_scene_ids(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    await call(server_with_log(log), "search_scenes", ROME_JULY)
    (record,) = all_records(log)
    assert record["tool"] == "search_scenes"
    assert record["ok"] is True
    assert record["returned_ids"] == [SCENE.id]
    assert record["arguments"]["max_cloud_cover"] == 10
    assert record["trace_id"] is None  # no HTTP request in an in-process session


@pytest.mark.parametrize(
    ("arguments", "kind"),
    [
        ({**ROME_JULY, "bbox": [12.8, 41.6, 12.2, 42.1]}, "invalid_arguments"),  # rejected by the validators
        ({**ROME_JULY, "cloud_cover": 5}, "invalid_arguments"),  # unknown argument name
        ({**ROME_JULY, "max_cloud_cover": "20"}, "invalid_arguments"),  # wrong type
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


async def test_a_catalogue_failure_is_recorded_as_infrastructure(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    await call(server_with_log(log, FakeCatalog(fail=True)), "search_scenes", ROME_JULY)
    (record,) = all_records(log)
    assert record["ok"] is False
    assert record["error_kind"] == "catalogue"


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
    ToolCallLog(None)
    assert "tool-call log is off" in caplog.text
