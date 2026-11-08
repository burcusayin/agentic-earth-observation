"""STAC catalogue client: turns a validated SearchRequest into a search and returns safe scene summaries.

The catalogue is an external, untrusted source. Only whitelisted fields leave this module, every returned
scene is re-checked against the request, and every failure becomes a CatalogError with a fixed, readable
message (the catalogue's own error text is logged, never passed on, because it could carry instructions).
Requests go to the configured catalogue only: pystac-client follows the links in a catalogue's answers, with
the method, headers and body they name, so a link or a redirect to anywhere else is refused.
pystac-client is synchronous; async callers should run these methods in a thread (asyncio.to_thread).
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, field_validator
from pystac import STACError
from pystac_client import Client
from pystac_client.errors import ClientTypeError
from pystac_client.exceptions import APIError
from pystac_client.stac_api_io import StacApiIO
from pystac_client.warnings import DoesNotConformTo
from requests import Request
from urllib3.util import parse_url
from urllib3.util.retry import Retry

from agentic_earth_observation.domain import BBox, Product, SearchRequest

logger = logging.getLogger(__name__)

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
USER_AGENT = "agentic-earth-observation/1.0 (+https://github.com/burcusayin/agentic-earth-observation)"
RETRY_AFTER_MAX_S = 10  # the longest pause a Retry-After header gets before the next try

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

# Everything pystac-client and the JSON parser raise for a failed or malformed answer (JSON decode errors are
# ValueErrors; an unexpected shape surfaces as a TypeError, KeyError or AttributeError).
_CATALOGUE_FAILURES = (APIError, ClientTypeError, STACError, ValueError, TypeError, KeyError, AttributeError)


class CatalogError(Exception):
    """The catalogue could not answer: network error, timeout or server error. The message is safe to show."""


class CatalogRefused(CatalogError):
    """The catalogue answered that our request is wrong (a 4xx): a bug or a wrong setting of ours, which must
    not count as a failure of the catalogue. The message is safe to show."""


class SceneSummary(BaseModel):
    """The whitelisted fields of one scene. Nothing else from the catalogue reaches a model.

    The id, the one text a model and the answer repeat, has the shape of a Sentinel product id, so that it
    cannot carry a sentence.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^S[12][A-Z]_[A-Z0-9_]+$", max_length=100)
    product: Product
    acquired: AwareDatetime  # a time without a time zone is ambiguous, so it is refused
    cloud_cover: float | None = Field(default=None, ge=0, le=100, strict=True)  # true is not 1, "5" is not 5
    platform: str | None = Field(default=None, pattern=r"^[a-z0-9-]+$", max_length=40)

    @field_validator("acquired", mode="before")
    @classmethod
    def _a_time_is_not_a_number(cls, value: object) -> object:
        if isinstance(value, bool | int | float):
            raise ValueError("a time is written as text")  # pydantic would read a number as a timestamp
        return value

    @field_validator("acquired")
    @classmethod
    def _in_utc(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)  # dates are compared in UTC, like the request's date range


def retry_policy(max_retries: int) -> Retry:
    """How failed catalogue requests are retried: on rate limits and server errors, with a short backoff.

    Searches are read-only, so retrying a POST is safe. A server's Retry-After header is honoured, so a public
    service that asks for a pause gets one, but only up to RETRY_AFTER_MAX_S: urllib3 would otherwise wait as
    long as the server asks (hours), far past every timeout in Settings. The STAC deadline still bounds the
    whole call.
    """
    return Retry(
        total=max_retries,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=None,
        raise_on_status=False,
        respect_retry_after_header=True,
        retry_after_max=RETRY_AFTER_MAX_S,
    )


def _origin(url: str) -> tuple[str | None, str | None, int | None, str | None]:
    """The scheme, host, port and user information of a URL, read by urllib3, which is what sends the request.

    `urllib.parse` reads some URLs differently (a backslash before an `@`, for one), so a check made with it
    could pass a URL that then goes elsewhere.
    """
    parts = parse_url(url)
    port = parts.port or {"http": 80, "https": 443}.get(parts.scheme or "")
    return parts.scheme, parts.host, port, parts.auth


def only_to(url: str) -> Callable[[Request], None]:
    """A check on every request pystac-client is about to send: it must go to the origin of `url`.

    The answers of a catalogue name the next page and the search endpoint, with a method, headers and a body.
    Followed blindly, a hostile answer could send a chosen request to the servers on this machine, which trust
    local callers without an Origin header.
    """
    allowed = _origin(url)

    def check(request: Request) -> None:
        *origin, auth = _origin(str(request.url))
        # A link may carry no user information, or the configured URL's own; never another.
        if origin != list(allowed[:3]) or auth not in (None, allowed[3]):
            raise ValueError(f"refused a link away from the configured catalogue: {request.url!s}")

    return check


class StacCatalog:
    """Searches one STAC API. Construction makes no network call; the connection opens on first use."""

    def __init__(self, url: str = EARTH_SEARCH_URL, *, timeout_s: float = 20.0, max_retries: int = 3) -> None:
        self._url = url
        self._timeout_s = timeout_s
        self._io = StacApiIO(
            headers={"User-Agent": USER_AGENT}, timeout=timeout_s, max_retries=retry_policy(max_retries)
        )
        self._io.session.max_redirects = 0  # a redirect could lead anywhere; Earth Search sends none
        self._client: Client | None = None
        self._lock = threading.Lock()  # tools call the catalogue from several threads

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

    def _open(self) -> Client:
        with self._lock:
            if self._client is None:
                # Pass both again: Client.open resets the timeout and the request check of the StacApiIO to
                # its own arguments.
                self._client = Client.open(
                    self._url, stac_io=self._io, timeout=self._timeout_s, request_modifier=only_to(self._url)
                )
            return self._client

    def _fetch(self, what: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            client = self._open()
        except _CATALOGUE_FAILURES as err:
            raise _failure(what, err) from err
        # Outside the `try`s: this only checks our own parameters, so an error here is a bug of ours, which
        # must not be taken for a failure of the catalogue. The one exception is a catalogue whose landing
        # page no longer declares item search: that is the catalogue's doing.
        try:
            search = client.search(fields=_FIELDS, **params)
        except DoesNotConformTo as err:
            raise _failure(what, err) from err
        try:
            items = list(search.items_as_dicts())
        except _CATALOGUE_FAILURES as err:
            raise _failure(what, err) from err
        return [item for item in items if isinstance(item, dict)]


def _failure(what: str, err: Exception) -> CatalogError:
    """The error for a failed call: the catalogue's own text is logged, never passed on."""
    status = getattr(err, "status_code", None)
    logger.warning("%s failed: %s: %.300r", what, type(err).__name__, str(err))  # %r: one line in the log
    detail = f" (HTTP {status})" if status else ""
    message = f"{what} failed: the catalogue did not answer correctly{detail}"
    # 408 and 429 are the catalogue's load, not our request; the retries have already been spent on them.
    if isinstance(status, int) and 400 <= status < 500 and status not in (408, 429):
        return CatalogRefused(message)
    return CatalogError(message)


def summarise(item: dict[str, Any], product: Product, collection: str) -> SceneSummary | None:
    """Whitelist the fields of one raw STAC item; None if it is from another collection or malformed."""
    properties = item.get("properties")
    if item.get("collection") != collection or not isinstance(properties, dict):
        return None
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
    except (ValidationError, OverflowError):  # a time so far from UTC that it leaves the calendar
        return None


def matches(scene: SceneSummary, item: dict[str, Any], request: SearchRequest) -> bool:
    """Re-check a returned scene against the request, in case the catalogue ignored part of it."""
    if not request.dates.start <= scene.acquired.date() <= request.dates.end:
        return False
    if request.max_cloud_cover is not None and (
        scene.cloud_cover is None or scene.cloud_cover > request.max_cloud_cover
    ):
        return False
    footprint = footprint_boxes(item.get("bbox"))
    return footprint is not None and any(_intersects(part, request.bbox) for part in footprint)


Box = tuple[float, float, float, float]  # west, south, east, north


def footprint_boxes(raw: object) -> list[Box] | None:
    """A scene footprint as one box, or two when it crosses the antimeridian; None if malformed.

    Footprints are not search boxes: they may be large, cross 180 degrees, or carry heights (6 values), so
    they get their own lenient check instead of the BBox validator.
    """
    if not isinstance(raw, list) or len(raw) not in (4, 6):
        return None
    try:
        values = [float(v) for v in raw]
    except (TypeError, ValueError, OverflowError):  # OverflowError: an integer too large for a float
        return None
    west, south, east, north = values if len(values) == 4 else (values[0], values[1], values[3], values[4])
    if not all(math.isfinite(v) for v in (west, south, east, north)):
        return None
    if not (-180 <= west <= 180 and -180 <= east <= 180 and -90 <= south <= north <= 90):
        return None
    if west <= east:
        return [(west, south, east, north)]
    return [(west, south, 180.0, north), (-180.0, south, east, north)]  # crosses the antimeridian


def _intersects(part: Box, box: BBox) -> bool:
    west, south, east, north = part
    return west <= box.east and box.west <= east and south <= box.north and box.south <= north
