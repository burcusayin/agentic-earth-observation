"""Contract tests: the STAC client against real Earth Search answers, recorded once and replayed offline.

The cassettes live in tests/cassettes/. Re-record only when the client's requests change, and ask first:
    uv run pytest tests/test_stac_contract.py --record-mode=rewrite
"""

from __future__ import annotations

import re
from datetime import date

import pytest

from eo_agent_demo.catalog.stac import StacCatalog
from eo_agent_demo.domain import Product, SearchRequest

pytestmark = pytest.mark.vcr

S2_ID = re.compile(r"^S2[A-D]_T\d{2}[A-Z]{3}_\d{8}T\d{6}_L2A$")
S1_ID = re.compile(r"^S1[A-D]_IW_GRD[HM]_\w+$")
ROME_S2_ID = "S2A_T33TTG_20250730T100227_L2A"


def test_sentinel_2_search_with_a_cloud_filter() -> None:
    request = SearchRequest.model_validate(
        {
            "product": "sentinel-2-l2a",
            "bbox": {"west": 12.35, "south": 41.78, "east": 12.65, "north": 42.00},
            "dates": {"start": "2025-07-01", "end": "2025-07-31"},
            "max_cloud_cover": 10,
            "limit": 3,
        }
    )
    scenes = StacCatalog().search(request)
    assert 1 <= len(scenes) <= 3
    for scene in scenes:
        assert S2_ID.match(scene.id), scene.id
        assert scene.product is Product.SENTINEL_2_L2A
        assert scene.cloud_cover is not None
        assert scene.cloud_cover <= 10
        assert date(2025, 7, 1) <= scene.acquired.date() <= date(2025, 7, 31)
        assert scene.platform is not None
        assert scene.platform.startswith("sentinel-2")
    assert [s.acquired for s in scenes] == sorted((s.acquired for s in scenes), reverse=True)


def test_sentinel_1_search_without_a_cloud_filter() -> None:
    request = SearchRequest.model_validate(
        {
            "product": "sentinel-1-grd",
            "bbox": {"west": 14.85, "south": 37.65, "east": 15.15, "north": 37.85},
            "dates": {"start": "2025-02-01", "end": "2025-02-07"},
        }
    )
    scenes = StacCatalog().search(request)
    assert scenes
    for scene in scenes:
        assert S1_ID.match(scene.id), scene.id
        assert scene.product is Product.SENTINEL_1_GRD
        assert scene.cloud_cover is None  # radar scenes carry no cloud cover
        assert scene.platform is not None
        assert scene.platform.startswith("sentinel-1")


def test_get_scene_by_id() -> None:
    scene = StacCatalog().get_scene(Product.SENTINEL_2_L2A, ROME_S2_ID)
    assert scene is not None
    assert scene.id == ROME_S2_ID
    assert scene.acquired.date() == date(2025, 7, 30)
