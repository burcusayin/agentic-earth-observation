"""STAC catalogue client: turns a validated SearchRequest into a search and returns safe scene summaries.

The catalogue is an external, untrusted source. Only whitelisted fields leave this module, every returned
scene is re-checked against the request, and every failure becomes a CatalogError with a readable message.
pystac-client is synchronous; async callers should run these methods in a thread (asyncio.to_thread).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pystac_client import Client
from pystac_client.exceptions import APIError
from pystac_client.stac_api_io import StacApiIO

from agentic_earth_observation.domain import BBox, Product, SearchRequest

logger = logging.getLogger(__name__)

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
USER_AGENT = "agentic-earth-observation/0.1 (+https://github.com/burcusayin/agentic-earth-observation)"

# Product names used by the tools, mapped to Earth Search collection ids. Every product must be listed: a
# missing entry is a bug, never a reason to pass the product name through to the catalogue.
EARTH_SEARCH_COLLECTIONS: dict[Product, str] = {
    Product.SENTINEL_2_L2A: "sentinel-2-c1-l2a",  # not the legacy "sentinel-2-l2a": its ids differ
    Product.SENTINEL_1_GRD: "sentinel-1-grd",
}

# Ask the catalogue for these fields only: smaller responses, and nothing we would have to discard anyway.
# Earth Search still sends assets (download links, ~15 KB per scene) unless they are excluded explicitly.
_FIELDS = [
    "id",
    "collection",
    "bbox",
    "properties.datetime",
    "properties.eo:cloud_cover",
    "properties.platform",
    "-assets",
    "-links",
    "-geometry",
]
_NEWEST_FIRST = [{"field": "properties.datetime", "direction": "desc"}]


class CatalogError(Exception):
    """The catalogue could not answer: network error, timeout or server error. The message is safe to show."""


class SceneSummary(BaseModel):
    """The whitelisted fields of one scene. Nothing else from the catalogue reaches a model."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$", max_length=200)
    product: Product
    acquired: datetime
    cloud_cover: float | None = Field(default=None, ge=0, le=100)
    platform: str | None = Field(default=None, pattern=r"^[a-z0-9-]+$", max_length=40)


class StacCatalog:
    """Searches one STAC API. Construction makes no network call; the connection opens on first use."""

    def __init__(self, url: str = EARTH_SEARCH_URL, *, timeout_s: float = 20.0, max_retries: int = 3) -> None:
        self._url = url
        self._io = StacApiIO(headers={"User-Agent": USER_AGENT}, timeout=timeout_s, max_retries=max_retries)
        self._client: Client | None = None

    def search(self, request: SearchRequest) -> list[SceneSummary]:
        """Scenes matching the request, newest first, at most `request.limit`."""
        collection = EARTH_SEARCH_COLLECTIONS[request.product]
        query = None
        if request.max_cloud_cover is not None:
            # `query`, not CQL2 `filter`: Earth Search ignores `filter` without any error.
            query = {"eo:cloud_cover": {"lte": request.max_cloud_cover}}
        items = self._fetch(
            "scene search",
            {
                "collections": [collection],
                "bbox": request.bbox.as_list(),
                "datetime": request.dates.to_stac_interval(),
                "query": query,
                "sortby": _NEWEST_FIRST,
                "max_items": request.limit,
                "limit": request.limit,
            },
        )
        scenes = [
            s
            for item in items
            if (s := summarise(item, request.product, collection)) and matches(s, item, request)
        ]
        if len(scenes) < len(items):
            logger.warning(
                "dropped %d of %d catalogue items that did not match the request",
                len(items) - len(scenes),
                len(items),
            )
        return sorted(scenes, key=lambda s: (-s.acquired.timestamp(), s.id))

    def get_scene(self, product: Product, scene_id: str) -> SceneSummary | None:
        """One scene by id, or None when the catalogue has no such scene for this product."""
        collection = EARTH_SEARCH_COLLECTIONS[product]
        items = self._fetch("scene lookup", {"collections": [collection], "ids": [scene_id], "max_items": 1})
        scenes = [s for item in items if (s := summarise(item, product, collection)) and s.id == scene_id]
        return scenes[0] if scenes else None

    def _fetch(self, what: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            if self._client is None:
                self._client = Client.open(self._url, stac_io=self._io)
            return list(self._client.search(fields=_FIELDS, **params).items_as_dicts())
        except APIError as err:
            raise CatalogError(f"{what} failed: the catalogue at {self._url} did not answer ({err})") from err


def summarise(item: dict[str, Any], product: Product, collection: str) -> SceneSummary | None:
    """Whitelist the fields of one raw STAC item; None if it is from another collection or malformed."""
    if item.get("collection") != collection:
        return None
    properties = item.get("properties") or {}
    try:
        return SceneSummary.model_validate(
            {
                "id": item.get("id"),
                "product": product,
                "acquired": properties.get("datetime"),
                "cloud_cover": properties.get("eo:cloud_cover"),
                "platform": properties.get("platform"),
            }
        )
    except ValidationError:
        return None


def matches(scene: SceneSummary, item: dict[str, Any], request: SearchRequest) -> bool:
    """Re-check a returned scene against the request, in case the catalogue ignored part of it."""
    if not request.dates.start <= scene.acquired.date() <= request.dates.end:
        return False
    if request.max_cloud_cover is not None and (
        scene.cloud_cover is None or scene.cloud_cover > request.max_cloud_cover
    ):
        return False
    try:
        footprint = BBox.model_validate(
            dict(zip(("west", "south", "east", "north"), item["bbox"], strict=True))
        )
    except (KeyError, TypeError, ValueError):
        return False
    return _intersects(footprint, request.bbox)


def _intersects(a: BBox, b: BBox) -> bool:
    return a.west < b.east and b.west < a.east and a.south < b.north and b.south < a.north
