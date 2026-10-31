"""The record of every tool call the MCP server answers, kept by the server itself for the evaluation.

docs/evaluation.md: "Tool calls are recorded at the MCP server, not reported by the agents." Each call is one
JSON line with the turn's trace id (sent by the agent in the X-EO-Trace header), the tool, its arguments,
whether it succeeded, the kind of error (the model's mistake or the catalogue's failure), and the scene ids it
returned.
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from mcp.server.lowlevel.server import request_ctx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentic_earth_observation.catalog.stac import CatalogError

logger = logging.getLogger(__name__)

TRACE_HEADER = "X-EO-Trace"
_TRACE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")  # the header is untrusted input

ErrorKind = Literal["invalid_arguments", "catalogue", "other"]


class ToolCallRecord(BaseModel):
    """One tool call, as the server saw it."""

    model_config = ConfigDict(frozen=True)

    trace_id: str | None
    tool: str
    arguments: dict[str, Any]
    ok: bool
    error_kind: ErrorKind | None = None
    returned_ids: list[str] = []
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ToolCallLog:
    """Appends records to a JSONL file, one line per call, safely from several threads. No path: no log."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()
        if path is None:
            logger.warning("the tool-call log is off: evaluation runs cannot be scored without it")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: ToolCallRecord) -> None:
        if self.path is None:
            return
        line = record.model_dump_json() + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as log:
            log.write(line)


def read_tool_calls(path: Path, trace_id: str) -> list[ToolCallRecord]:
    """The calls recorded for one trace id, in the order the server answered them."""
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = ToolCallRecord.model_validate_json(line)
        except ValidationError as err:
            raise ValueError(f"{path}:{number}: not a tool-call record: {err}") from err
        if record.trace_id == trace_id:
            records.append(record)
    return records


def current_trace_id() -> str | None:
    """The trace id of the request being answered; None outside HTTP or without a valid header."""
    try:
        request = request_ctx.get().request
    except LookupError:
        return None
    if request is None or not hasattr(request, "headers"):
        return None
    value = request.headers.get(TRACE_HEADER)
    return value if isinstance(value, str) and _TRACE_ID.match(value) else None


def error_kind(err: BaseException) -> ErrorKind:
    """Whether a failed call was the model's mistake or the catalogue's failure, from the cause chain."""
    seen: BaseException | None = err
    while seen is not None:
        if isinstance(seen, CatalogError):
            return "catalogue"
        if isinstance(seen, ValidationError):
            return "invalid_arguments"
        seen = seen.__cause__
    text = str(err)
    if text.startswith(("invalid arguments", "unknown argument")) or "invalid arguments" in text:
        return "invalid_arguments"
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
