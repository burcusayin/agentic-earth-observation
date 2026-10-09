"""Pure scoring functions for tool-use accuracy. No network, no LLM."""
from __future__ import annotations

from datetime import date

BBox = tuple[float, float, float, float]  # west, south, east, north


def bbox_iou(a: BBox, b: BBox) -> float:
    """Intersection over union of two lon/lat boxes (0.0 when they do not overlap)."""
    for box in (a, b):
        if len(box) != 4 or box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError(f"invalid bbox: {box!r}")
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


def date_range_matches(got: tuple[str, str], expected: tuple[str, str], tolerance_days: int = 0) -> bool:
    """True when both ends of the range are within tolerance_days of the expected ones."""
    g0, g1, e0, e1 = (date.fromisoformat(d) for d in (*got, *expected))
    return abs((g0 - e0).days) <= tolerance_days and abs((g1 - e1).days) <= tolerance_days


def score_search_call(call: dict, expected: dict, min_iou: float = 0.5) -> dict[str, bool]:
    """Compare one search_scenes call with the golden expectation, argument by argument."""
    try:
        bbox_ok = bbox_iou(tuple(call["bbox"]), tuple(expected["bbox"])) >= min_iou
    except (KeyError, TypeError, ValueError):
        bbox_ok = False
    try:
        dates_ok = date_range_matches(
            (call["start_date"], call["end_date"]), (expected["start_date"], expected["end_date"])
        )
    except (KeyError, TypeError, ValueError):
        dates_ok = False
    cloud_expected = expected.get("max_cloud_cover")
    result = {
        "collection": call.get("collection") == expected.get("collection"),
        "bbox": bbox_ok,
        "dates": dates_ok,
        "cloud_cover": cloud_expected is None or call.get("max_cloud_cover") == cloud_expected,
    }
    result["all"] = all(result.values())
    return result
