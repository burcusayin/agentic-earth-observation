"""Shared test doubles: a fake catalogue (no network) and sample data."""

from __future__ import annotations

from datetime import UTC, datetime

from agentic_earth_observation.catalog.stac import CatalogError, SceneSummary
from agentic_earth_observation.domain import Product, SearchRequest

SCENE = SceneSummary(
    id="S2A_T33TTG_20250730T100227_L2A",
    product=Product.SENTINEL_2_L2A,
    acquired=datetime(2025, 7, 30, 10, 9, 29, tzinfo=UTC),
    cloud_cover=4.1,
    platform="sentinel-2a",
)
ROME_JULY = {
    "product": "sentinel-2-l2a",
    "bbox": [12.2344, 41.6556, 12.8558, 42.141],
    "start_date": "2025-07-01",
    "end_date": "2025-07-31",
    "max_cloud_cover": 10,
}


class FakeCatalog:
    """Answers from a fixed list of scenes and records every search request it receives."""

    def __init__(self, scenes: list[SceneSummary] | None = None, *, fail: bool = False) -> None:
        self.scenes = scenes if scenes is not None else [SCENE]
        self.fail = fail
        self.requests: list[SearchRequest] = []
        self.lookups: list[str] = []

    def search(self, request: SearchRequest) -> list[SceneSummary]:
        self.requests.append(request)
        if self.fail:
            raise CatalogError("scene search failed: the catalogue did not answer (Read timed out)")
        return self.scenes[: request.limit]

    def get_scene(self, product: Product, scene_id: str) -> SceneSummary | None:
        self.lookups.append(scene_id)
        if self.fail:
            raise CatalogError("scene lookup failed: the catalogue did not answer (Read timed out)")
        return next((s for s in self.scenes if s.id == scene_id and s.product is product), None)
