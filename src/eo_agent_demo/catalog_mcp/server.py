"""MCP server exposing the satellite catalogue as four tools.

No LLM runs here: each tool is a plain function over the validated domain types, the gazetteer and the
catalogue client, so it can be tested in-process and used by any MCP client. The tool docstrings and
parameter descriptions are what a model reads; they are part of what the evaluation measures, so change
them deliberately.

Run it with `make serve-mcp`; it listens on http://127.0.0.1:8001/mcp only.
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Annotated, Protocol

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field, ValidationError

from eo_agent_demo.catalog.gazetteer import Gazetteer, Place
from eo_agent_demo.catalog.stac import CatalogError, SceneSummary, StacCatalog
from eo_agent_demo.domain import MISSION_START, Product, SearchRequest

HOST = "127.0.0.1"
PORT = 8001


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


class PlaceResult(BaseModel):
    found: bool
    place: Place | None = None
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


LOCAL = ToolAnnotations(readOnlyHint=True, openWorldHint=False)  # answers from local data only
CATALOGUE = ToolAnnotations(readOnlyHint=True, openWorldHint=True)  # queries the external catalogue

INSTRUCTIONS = (
    "Tools to find Sentinel-1 and Sentinel-2 satellite scenes (metadata only, no imagery). "
    "Resolve place names with resolve_place before searching; never invent coordinates or scene ids."
)


def build_server(catalog: Catalog, gazetteer: Gazetteer) -> FastMCP:
    """The MCP server with its four tools, using the given catalogue and gazetteer."""
    mcp = FastMCP(
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
        name: Annotated[str, Field(description="A place name in any language, e.g. 'Rome' or 'Gardasee'.")],
    ) -> PlaceResult:
        """Find a known place by name and return its bounding box. If it is unknown, the result says so and
        suggests similar known names: ask the user which one they meant instead of guessing coordinates."""
        place = gazetteer.resolve(name)
        if place:
            return PlaceResult(found=True, place=place)
        else:
            return PlaceResult(found=False, suggestions=gazetteer.suggest(name))

    @mcp.tool(annotations=CATALOGUE)
    async def search_scenes(
        product: Annotated[Product, Field(description="Which product to search; see list_products.")],
        bbox: Annotated[
            list[float],
            Field(
                min_length=4,
                max_length=4,
                description="west, south, east, north in degrees, from resolve_place",
            ),
        ],
        start_date: Annotated[date, Field(description="First day, inclusive (YYYY-MM-DD).")],
        end_date: Annotated[date, Field(description="Last day, inclusive (YYYY-MM-DD).")],
        max_cloud_cover: Annotated[
            float | None, Field(description="Sentinel-2 only: maximum cloud cover in percent (0-100).")
        ] = None,
        limit: Annotated[int, Field(description="Maximum number of scenes, 1-50.")] = 10,
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
            note = "No scenes match. Try a longer date range, or a higher cloud limit for Sentinel-2."
        return SearchResult(count=len(scenes), scenes=scenes, note=note)

    @mcp.tool(annotations=CATALOGUE)
    async def get_scene(
        product: Annotated[Product, Field(description="The product the scene belongs to.")],
        scene_id: Annotated[str, Field(description="A scene id returned by search_scenes.")],
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
