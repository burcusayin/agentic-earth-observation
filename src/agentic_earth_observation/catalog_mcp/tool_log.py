"""The record of every tool call the MCP server answers, kept by the server itself for the evaluation.

docs/evaluation.md: "Tool calls are recorded at the MCP server, not reported by the agents." Each call is one
JSON line with the turn's trace id (sent by the agent in the X-EO-Trace header), the JSON-RPC request id (the
order in which the client sent its calls), the tool, its arguments as sent, whether it succeeded, the kind of
error, and the scene ids it returned.

The error kind says what made a call fail; it is not a score:
- invalid_arguments: the tool refused the arguments, for a wrong type or name or because the request breaks a
  rule (dates before launch, for example). This alone does not make it a malformed call.
- unknown_tool: the call named a tool the server does not have.
- catalogue: the catalogue failed or did not answer in time (infrastructure).
- cancelled: the client gave up before the call finished.
- other: anything else.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.lowlevel.server import request_ctx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentic_earth_observation.catalog.stac import CatalogError

logger = logging.getLogger(__name__)

TRACE_HEADER = "X-EO-Trace"
TRACE_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")  # the header is untrusted input; matched in full

ErrorKind = Literal["invalid_arguments", "unknown_tool", "catalogue", "cancelled", "other"]


class ArgumentError(ToolError):
    """The server refused a call's arguments. The message says how to fix them."""


class UnknownToolError(ToolError):
    """The call named a tool the server does not have."""


class ToolCallRecord(BaseModel):
    """One tool call, as the server saw it."""

    model_config = ConfigDict(frozen=True)

    trace_id: str | None
    request_id: int | str | None = None  # numbered by the client, in the order it sent its calls
    tool: str
    arguments: dict[str, Any]
    ok: bool
    error_kind: ErrorKind | None = None
    returned_ids: list[str] = []
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))  # when the call arrived


class ToolCallLog:
    """Appends records to a JSONL file, one line per call, safely from several threads. No path: no log.

    The file is created when the log is, so a path that cannot be written stops the server at start-up rather
    than failing every call.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()
        if path is None:
            logger.warning("the tool-call log is off: evaluation runs cannot be scored without it")
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.open("a", encoding="utf-8").close()
        except OSError as err:
            raise ValueError(
                f"cannot write the tool-call log at {path} ({err.strerror}); "
                "set EO_TOOL_LOG to a writable file"
            ) from err

    def write(self, record: ToolCallRecord) -> None:
        if self.path is None:
            return
        line = record.model_dump_json() + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as log:
            log.write(line)


def read_tool_calls(path: Path, trace_id: str) -> list[ToolCallRecord]:
    """The calls recorded for one trace id, in the order the client sent them.

    Calls that run at the same time are written in the order they finish, so they are sorted by request id.
    Request ids are numbered per MCP session, which is one turn: a trace id must not span two sessions.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"no tool-call log at {path}: is the MCP server writing to another file (EO_TOOL_LOG)?"
        )
    records = []
    # Split on "\n" only: str.splitlines() also splits on U+2028 and the like, which JSON leaves unescaped.
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        try:
            record = ToolCallRecord.model_validate_json(line)
        except ValidationError as err:
            raise ValueError(f"{path}:{number}: not a tool-call record: {err}") from err
        if record.trace_id == trace_id:
            records.append(record)
    return sorted(records, key=_sent_order)


def _sent_order(record: ToolCallRecord) -> tuple[int, int]:
    """Records with a numeric request id first, by that id; the others keep the order they were written in."""
    return (0, record.request_id) if isinstance(record.request_id, int) else (1, 0)


def current_trace_id() -> str | None:
    """The trace id of the request being answered; None outside HTTP or without a valid header."""
    try:
        request = request_ctx.get().request
    except LookupError:
        return None
    if request is None or not hasattr(request, "headers"):
        return None
    value = request.headers.get(TRACE_HEADER)
    if value is None:
        return None
    if isinstance(value, str) and TRACE_ID.fullmatch(value):
        return value
    logger.warning(
        "refused an invalid %s header %r; the call is recorded without a trace id",
        TRACE_HEADER,
        str(value)[:80],
    )
    return None


def current_request_id() -> int | str | None:
    """The JSON-RPC id of the request being answered; None outside a request."""
    try:
        return request_ctx.get().request_id
    except LookupError:
        return None


def error_kind(err: BaseException) -> ErrorKind:
    """What made a call fail, from the exception and its chain of causes."""
    seen: BaseException | None = err
    while seen is not None:
        if isinstance(seen, asyncio.CancelledError):
            return "cancelled"
        if isinstance(seen, CatalogError):
            return "catalogue"
        if isinstance(seen, UnknownToolError):
            return "unknown_tool"
        if isinstance(seen, ArgumentError | ValidationError):
            return "invalid_arguments"
        seen = seen.__cause__
    return "other"


def returned_ids(result: object) -> list[str]:
    """Scene ids in a tool's structured result: every scene of a search, or the scene of a lookup."""
    structured = result[1] if isinstance(result, tuple) and len(result) == 2 else result
    if not isinstance(structured, dict):
        return []
    scenes = structured.get("scenes")
    if isinstance(scenes, list):
        return [s["id"] for s in scenes if isinstance(s, dict) and isinstance(s.get("id"), str)]
    scene = structured.get("scene")
    if isinstance(scene, dict) and isinstance(scene.get("id"), str):
        return [scene["id"]]
    return []
