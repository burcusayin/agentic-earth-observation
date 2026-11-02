"""MCP server exposing the satellite catalogue as four tools.

No LLM runs here: each tool is a plain function over the validated domain types, the gazetteer and the
catalogue client, so it can be tested in-process and used by any MCP client. The tool docstrings and
parameter descriptions are what a model reads; they are part of what the evaluation measures, so change
them deliberately.

Run it with `make serve-mcp`; it listens on http://127.0.0.1:8001/mcp only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, date, datetime
from typing import Annotated, Any, NamedTuple, Protocol

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ContentBlock, Tool, ToolAnnotations
from pydantic import BaseModel, Field, ValidationError, WithJsonSchema

from agentic_earth_observation.catalog.gazetteer import Gazetteer
from agentic_earth_observation.catalog.stac import CatalogError, SceneSummary, StacCatalog
from agentic_earth_observation.catalog_mcp.tool_log import (
    ArgumentError,
    ErrorKind,
    ToolCallLog,
    ToolCallRecord,
    UnknownToolError,
    current_request_id,
    current_trace_id,
    error_kind,
    returned_ids,
)
from agentic_earth_observation.config import Settings
from agentic_earth_observation.domain import MISSION_START, Product, SearchRequest

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
PORT = 8001
CLOUD_SCHEMA = {
    "type": "number",
    "minimum": 0,
    "maximum": 100,
    "description": "Sentinel-2 only: maximum cloud cover in percent (0-100). Leave it out for no filter.",
}
ISO_DAY = r"^\d{4}-\d{2}-\d{2}$"  # dates as YYYY-MM-DD only, so a number is never read as a timestamp
SCENE_ID = r"^[A-Za-z0-9_.-]+$"
KINDS = {"array": "a list", "number": "a number", "integer": "a whole number"}  # types FastMCP would parse
UNRECORDED = "internal server error: this call could not be recorded. It is not a mistake in the call."


class Catalog(Protocol):
    """What the server needs from a catalogue. StacCatalog provides it; tests pass a fake."""

    def search(self, request: SearchRequest) -> list[SceneSummary]: ...

    def get_scene(self, product: Product, scene_id: str) -> SceneSummary | None: ...


class ProductInfo(BaseModel):
    id: Product
    description: str
    cloud_filter: bool
    first_date: date


class ProductList(BaseModel):
    products: list[ProductInfo]


class PlaceInfo(BaseModel):
    key: str
    name: str
    bbox: list[float] = Field(description="west, south, east, north: pass it to search_scenes as it is")


class PlaceResult(BaseModel):
    found: bool
    place: PlaceInfo | None = None
    suggestions: list[str] = []


class SearchResult(BaseModel):
    count: int
    scenes: list[SceneSummary]
    note: str | None = None


class SceneResult(BaseModel):
    found: bool
    scene: SceneSummary | None = None


def explain(err: ValidationError) -> str:
    """One readable line from a pydantic error: field and message, without pydantic internals.

    Wrong number types get a message that says exactly how to fix the call, because models often send "20"
    (text) for 20 and, told only "should be a valid number", repeat the same call.
    """
    parts = []
    for error in err.errors():
        message = _number_hint(error) or str(error["msg"]).removeprefix("Value error, ")
        where = ".".join(str(part) for part in error["loc"])
        parts.append(f"{where}: {message}" if where else message)
    return "; ".join(parts)


def _number_hint(error: Any) -> str | None:  # noqa: ANN401 - a pydantic ErrorDetails dict
    """An actionable message for a value of the wrong type where a number is expected, else None."""
    if error["type"] not in ("float_type", "int_type"):
        return None
    kind = "a whole number" if error["type"] == "int_type" else "a number"
    value = error.get("input")
    if isinstance(value, bool):
        return f"must be {kind}, not {str(value).lower()}"
    if isinstance(value, str):
        return _text_hint(kind, value, optional=False)
    if isinstance(value, float) and error["type"] == "int_type":
        return f"must be {kind} such as {round(value)}, not {value}"
    return f"must be {kind}"


def text_instead_of_values(arguments: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """One message for each argument sent as text where the input schema expects a list or a number.

    FastMCP parses such text as JSON before validating it, so "[1, 2, 3, 4]" would pass as a list and "null"
    as a missing value, while the record kept the text the model sent. These calls are refused instead.
    """
    properties: dict[str, Any] = schema.get("properties", {})
    required = set(schema.get("required", []))
    problems = []
    for name, value in arguments.items():
        kind = KINDS.get(properties.get(name, {}).get("type"))
        if isinstance(value, str) and kind is not None:
            problems.append(f"{name}: {_text_hint(kind, value, optional=name not in required)}")
    return problems


def _text_hint(kind: str, value: str, *, optional: bool) -> str:
    """What to tell a model that sent `value` as text where `kind` belongs."""
    if optional and value.strip() == "null":
        return 'leave it out instead of sending the text "null"'
    shown = value if len(value) <= 60 else value[:60] + "..."
    if _is_json_of_kind(kind, value):  # the quotes are the only mistake
        return f'must be {kind} such as {value.strip()}, not the text "{shown}"; send it without quotes'
    return f'must be {kind}, not the text "{shown}"'


def _is_json_of_kind(kind: str, value: str) -> bool:
    try:
        parsed = json.loads(value)
    except ValueError:
        return False
    if isinstance(parsed, bool):  # JSON true is not a number
        return False
    if kind == "a list":
        return isinstance(parsed, list)
    if kind == "a whole number":
        return isinstance(parsed, int)
    if kind == "a number":
        return isinstance(parsed, int | float) and math.isfinite(parsed)
    return False


class _Arrival(NamedTuple):
    """Who sent a call and when, taken before the tool runs."""

    trace_id: str | None
    request_id: int | str | None
    at: datetime


class CatalogMCP(FastMCP):
    """FastMCP that checks every call before the tool runs, explains each refusal and records every call.

    By default FastMCP drops arguments it does not know (a misspelt `cloud_cover` would search with no
    cloud limit, and nobody would notice), quietly parses text such as "[1, 2, 3, 4]" into lists, and returns
    pydantic's raw text for arguments of the wrong type.
    """

    tool_log: ToolCallLog | None = None  # every answered call is recorded here, for the evaluation

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> Sequence[ContentBlock] | dict[str, Any]:
        arrival = _Arrival(current_trace_id(), current_request_id(), datetime.now(UTC))
        try:
            result = await self._checked_call(name, arguments)
        except (Exception, asyncio.CancelledError) as err:  # a cancelled call is recorded too
            self._record(arrival, name, arguments, ok=False, kind=error_kind(err))
            raise  # the tool's own error, even if it could not be recorded
        if not self._record(arrival, name, arguments, ok=True, ids=returned_ids(result)):
            raise ToolError(UNRECORDED)
        return result

    async def _checked_call(
        self, name: str, arguments: dict[str, Any]
    ) -> Sequence[ContentBlock] | dict[str, Any]:
        tools = await self.list_tools()
        tool = next((t for t in tools if t.name == name), None)
        if tool is None:
            known = ", ".join(sorted(t.name for t in tools))
            raise UnknownToolError(f"unknown tool {name}; the tools are: {known}")
        allowed = set(tool.inputSchema.get("properties", {}))
        if unknown := sorted(set(arguments) - allowed):
            raise ArgumentError(
                f"unknown argument(s) for {name}: {', '.join(unknown)}; "
                f"the arguments are: {', '.join(sorted(allowed))}"
            )
        if problems := text_instead_of_values(arguments, tool.inputSchema):
            raise ArgumentError(f"invalid arguments for {name}: {'; '.join(problems)}")
        try:
            return await super().call_tool(name, arguments)
        except ToolError as err:
            if isinstance(err.__cause__, ValidationError):
                cause = err.__cause__
                raise ArgumentError(f"invalid arguments for {name}: {explain(cause)}") from cause
            raise

    def _record(
        self,
        arrival: _Arrival,
        name: str,
        arguments: dict[str, Any],
        *,
        ok: bool,
        kind: ErrorKind | None = None,
        ids: list[str] | None = None,
    ) -> bool:
        """Write one record. False, with the reason in the server's log, if the log could not be written."""
        if self.tool_log is None:
            return True
        record = ToolCallRecord(
            trace_id=arrival.trace_id,
            request_id=arrival.request_id,
            at=arrival.at,
            tool=name,
            arguments=arguments,
            ok=ok,
            error_kind=kind,
            returned_ids=ids or [],
        )
        try:
            self.tool_log.write(record)
        except OSError:
            logger.exception("could not write the tool-call log at %s", self.tool_log.path)
            return False
        return True


LOCAL = ToolAnnotations(readOnlyHint=True, openWorldHint=False)  # answers from local data only
CATALOGUE = ToolAnnotations(readOnlyHint=True, openWorldHint=True)  # queries the external catalogue

INSTRUCTIONS = (
    "Tools to find Sentinel-1 and Sentinel-2 satellite scenes (metadata only, no imagery). "
    "Resolve place names with resolve_place before searching; never invent coordinates or scene ids."
)


def tool_descriptions_hash(tools: Iterable[Tool]) -> str:
    """SHA-256 of what a model reads about the given tools, recorded with every evaluation run.

    It covers each tool's name, description and input schema (parameter descriptions included), in name order,
    so a changed docstring or parameter description changes the hash.
    """
    described = [
        {"name": t.name, "description": t.description, "input_schema": t.inputSchema}
        for t in sorted(tools, key=lambda t: t.name)
    ]
    text = json.dumps(described, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


async def _ask_catalogue[T](what: str, ask: Callable[[], T], deadline_s: float | None) -> T:
    """Run a blocking catalogue call in a worker thread and stop waiting after `deadline_s` seconds.

    A late answer is dropped. The thread itself cannot be stopped; it ends when its HTTP timeouts do.
    """
    try:
        async with asyncio.timeout(deadline_s):
            return await asyncio.to_thread(ask)
    except TimeoutError:
        raise CatalogError(f"{what} failed: the catalogue did not answer within {deadline_s:g} s") from None


def build_server(
    catalog: Catalog,
    gazetteer: Gazetteer,
    tool_log: ToolCallLog | None = None,
    *,
    catalog_deadline_s: float | None = None,
) -> FastMCP:
    """The MCP server with its four tools, using the given catalogue and gazetteer; calls go to `tool_log`.

    Each call to the catalogue gets at most `catalog_deadline_s` seconds (None: no limit), so a tool call
    always answers before the agent's MCP timeout.
    """
    mcp = CatalogMCP(
        "eo-catalog", instructions=INSTRUCTIONS, host=HOST, port=PORT, stateless_http=True, json_response=True
    )
    mcp.tool_log = tool_log

    @mcp.tool(annotations=LOCAL)
    def list_products() -> ProductList:
        """List the satellite products you can search, with what each is for and its first available date."""
        return ProductList(
            products=[
                ProductInfo(
                    id=Product.SENTINEL_2_L2A,
                    description=(
                        "Sentinel-2 optical imagery (Level-2A). Clouds block it; supports a cloud filter."
                    ),
                    cloud_filter=True,
                    first_date=MISSION_START[Product.SENTINEL_2_L2A],
                ),
                ProductInfo(
                    id=Product.SENTINEL_1_GRD,
                    description="Sentinel-1 radar imagery (GRD). Sees through clouds; no cloud filter.",
                    cloud_filter=False,
                    first_date=MISSION_START[Product.SENTINEL_1_GRD],
                ),
            ]
        )

    @mcp.tool(annotations=LOCAL)
    def resolve_place(
        name: Annotated[str, Field(description="A place name in any language, e.g. 'Milan' or 'Napoli'.")],
    ) -> PlaceResult:
        """Find a known place by name and return its bounding box. If it is unknown, the result says so and
        suggests similar known names: ask the user which one they meant instead of guessing coordinates."""
        place = gazetteer.resolve(name)
        if place:
            info = PlaceInfo(key=place.key, name=place.name, bbox=place.bbox.as_list())
            return PlaceResult(found=True, place=info)
        else:
            return PlaceResult(found=False, suggestions=gazetteer.suggest(name))

    @mcp.tool(annotations=CATALOGUE)
    async def search_scenes(
        product: Annotated[Product, Field(description="Which product to search; see list_products.")],
        bbox: Annotated[
            list[Annotated[float, Field(strict=True)]],
            Field(
                min_length=4,
                max_length=4,
                description="west, south, east, north in degrees, from resolve_place",
            ),
        ],
        start_date: Annotated[str, Field(pattern=ISO_DAY, description="First day, inclusive (YYYY-MM-DD).")],
        end_date: Annotated[str, Field(pattern=ISO_DAY, description="Last day, inclusive (YYYY-MM-DD).")],
        max_cloud_cover: Annotated[
            float | None,
            Field(strict=True, description="Sentinel-2 only: maximum cloud cover in percent (0-100)."),
            # Advertised as a plain optional number: small models send "20" (text) for a "number or null"
            # parameter but numbers for plain ones. Leaving it out still means no cloud filter.
            WithJsonSchema(CLOUD_SCHEMA),
        ] = None,
        limit: Annotated[int, Field(strict=True, description="Maximum number of scenes, 1-50.")] = 10,
    ) -> SearchResult:
        """Search satellite scenes over a bounding box and date range, newest first. Returns scene ids with
        acquisition time and cloud cover. Only cite scene ids that this tool returned."""
        try:
            request = SearchRequest.model_validate(
                {
                    "product": product,
                    "bbox": dict(zip(("west", "south", "east", "north"), bbox, strict=True)),
                    "dates": {"start": start_date, "end": end_date},
                    "max_cloud_cover": max_cloud_cover,
                    "limit": limit,
                }
            )
        except ValidationError as err:
            raise ToolError(explain(err)) from err

        try:
            scenes = await _ask_catalogue("scene search", lambda: catalog.search(request), catalog_deadline_s)
        except CatalogError as err:
            raise ToolError(str(err)) from err

        note = None
        if not scenes:
            note = (
                "No scenes match this search. Tell the user so; do not widen the dates or the cloud limit "
                "unless they ask."
            )
        return SearchResult(count=len(scenes), scenes=scenes, note=note)

    @mcp.tool(annotations=CATALOGUE)
    async def get_scene(
        product: Annotated[Product, Field(description="The product the scene belongs to.")],
        scene_id: Annotated[
            str, Field(pattern=SCENE_ID, max_length=200, description="A scene id returned by search_scenes.")
        ],
    ) -> SceneResult:
        """Get one scene by its id. Use it only with ids that search_scenes returned."""
        try:
            scene = await _ask_catalogue(
                "scene lookup", lambda: catalog.get_scene(product, scene_id), catalog_deadline_s
            )
        except CatalogError as err:
            raise ToolError(str(err)) from err
        return SceneResult(found=scene is not None, scene=scene)

    return mcp


def main() -> None:
    """Serve the tools over streamable HTTP on 127.0.0.1:8001, with Earth Search and the gazetteer."""
    settings = Settings()
    catalog = StacCatalog(
        settings.stac_api_url, timeout_s=settings.stac_timeout_s, max_retries=settings.stac_max_retries
    )
    log = ToolCallLog(settings.tool_log)
    server = build_server(catalog, Gazetteer.load(), log, catalog_deadline_s=settings.stac_deadline_s)
    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
