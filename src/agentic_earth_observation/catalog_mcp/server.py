"""MCP server exposing the satellite catalogue as four tools.

No LLM runs here: each tool is a plain function over the validated domain types, the gazetteer and the
catalogue client, so it can be tested in-process and used by any MCP client. The tool docstrings and
parameter descriptions are what a model reads; they are part of what the evaluation measures, so change
them deliberately.

Run it with `make serve-mcp`; it listens on http://127.0.0.1:8001/mcp only.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import date
from typing import Annotated, Any, Protocol

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ContentBlock, ToolAnnotations
from pydantic import BaseModel, Field, ValidationError

from agentic_earth_observation.catalog.gazetteer import Gazetteer
from agentic_earth_observation.catalog.stac import CatalogError, SceneSummary, StacCatalog
from agentic_earth_observation.domain import MISSION_START, Product, SearchRequest

HOST = "127.0.0.1"
PORT = 8001
ISO_DAY = r"^\d{4}-\d{2}-\d{2}$"  # dates as YYYY-MM-DD only, so a number is never read as a timestamp
SCENE_ID = r"^[A-Za-z0-9_.-]+$"


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
    """One readable line from a pydantic error: field and message, without pydantic internals or the input."""
    parts = []
    for error in err.errors():
        message = str(error["msg"]).removeprefix("Value error, ")
        where = ".".join(str(part) for part in error["loc"])
        parts.append(f"{where}: {message}" if where else message)
    return "; ".join(parts)


class CatalogMCP(FastMCP):
    """FastMCP that refuses unknown argument names and explains every argument error in plain words.

    By default FastMCP drops arguments it does not know (a misspelt `cloud_cover` would search with no
    cloud limit, and nobody would notice) and returns pydantic's raw text for arguments of the wrong type.
    """

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> Sequence[ContentBlock] | dict[str, Any]:
        tool = next((t for t in await self.list_tools() if t.name == name), None)
        if tool is not None:
            allowed = set(tool.inputSchema.get("properties", {}))
            if unknown := sorted(set(arguments) - allowed):
                raise ToolError(
                    f"unknown argument(s) for {name}: {', '.join(unknown)}; "
                    f"the arguments are: {', '.join(sorted(allowed))}"
                )
        try:
            return await super().call_tool(name, arguments)
        except ToolError as err:
            if isinstance(err.__cause__, ValidationError):
                raise ToolError(f"invalid arguments for {name}: {explain(err.__cause__)}") from err.__cause__
            raise


LOCAL = ToolAnnotations(readOnlyHint=True, openWorldHint=False)  # answers from local data only
CATALOGUE = ToolAnnotations(readOnlyHint=True, openWorldHint=True)  # queries the external catalogue

INSTRUCTIONS = (
    "Tools to find Sentinel-1 and Sentinel-2 satellite scenes (metadata only, no imagery). "
    "Resolve place names with resolve_place before searching; never invent coordinates or scene ids."
)


def build_server(catalog: Catalog, gazetteer: Gazetteer) -> FastMCP:
    """The MCP server with its four tools, using the given catalogue and gazetteer."""
    mcp = CatalogMCP(
        "eo-catalog", instructions=INSTRUCTIONS, host=HOST, port=PORT, stateless_http=True, json_response=True
    )

    # Worked example: a tool is a typed function. FastMCP turns the type hints into the input schema, the
    # docstring into the description, and the returned pydantic model into structured output.
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
            scenes = await asyncio.to_thread(catalog.search, request)
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
            scene = await asyncio.to_thread(catalog.get_scene, product, scene_id)
        except CatalogError as err:
            raise ToolError(str(err)) from err
        return SceneResult(found=scene is not None, scene=scene)

    return mcp


def main() -> None:
    """Serve the tools over streamable HTTP on 127.0.0.1:8001, with Earth Search and the gazetteer."""
    build_server(StacCatalog(), Gazetteer.load()).run(transport="streamable-http")


if __name__ == "__main__":
    main()
