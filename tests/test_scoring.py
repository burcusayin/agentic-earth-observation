from __future__ import annotations

import pytest

from eo_agent_demo.evals.scoring import bbox_iou, date_range_matches, score_search_call

ROME = (12.35, 41.78, 12.65, 42.00)


def test_iou_identical_and_disjoint() -> None:
    assert bbox_iou(ROME, ROME) == pytest.approx(1.0)
    assert bbox_iou(ROME, (0.0, 0.0, 1.0, 1.0)) == 0.0


def test_iou_rejects_inverted_box() -> None:
    with pytest.raises(ValueError, match="invalid bbox"):
        bbox_iou((2.0, 0.0, 1.0, 1.0), ROME)


def test_date_range_tolerance() -> None:
    assert date_range_matches(("2025-07-01", "2025-07-31"), ("2025-07-01", "2025-07-31"))
    assert not date_range_matches(("2025-07-02", "2025-07-31"), ("2025-07-01", "2025-07-31"))
    assert date_range_matches(("2025-07-02", "2025-07-31"), ("2025-07-01", "2025-07-31"), tolerance_days=1)


def test_score_search_call() -> None:
    expected = {
        "collection": "sentinel-2-l2a",
        "bbox": list(ROME),
        "start_date": "2025-07-01",
        "end_date": "2025-07-31",
        "max_cloud_cover": 20,
    }
    assert score_search_call(dict(expected), expected)["all"]
    wrong = dict(expected, collection="sentinel-1-grd")
    scored = score_search_call(wrong, expected)
    assert not scored["collection"]
    assert scored["bbox"]
    assert not scored["all"]
    assert not score_search_call({"collection": "sentinel-2-l2a"}, expected)["bbox"]
