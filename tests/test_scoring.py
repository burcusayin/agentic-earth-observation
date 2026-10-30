"""Tests for turn scoring: each rule in the Scoring section of docs/evaluation.md has a test here."""

from __future__ import annotations

from typing import Any

import pytest

from eo_agent_demo.catalog.gazetteer import Gazetteer
from eo_agent_demo.evals.golden import Expectation, ExpectedSearch, Route
from eo_agent_demo.evals.scoring import (
    ToolCall,
    TurnTrace,
    bbox_iou,
    retrieval_metrics,
    score_arguments,
    score_turn,
    tools_in_order,
)

FLORENCE = [11.1498, 43.7259, 11.3389, 43.8355]  # the gazetteer box
EXPECT = Expectation.model_validate(
    {
        "route": "delegate",
        "required_tools": ["resolve_place", "search_scenes"],
        "search": {
            "product": "sentinel-2-l2a",
            "place": "florence",
            "start": "2025-07-01",
            "end": "2025-07-31",
            "cloud": [0, 10],
        },
    }
)


def _expected_search() -> ExpectedSearch:
    assert EXPECT.search is not None
    return EXPECT.search


SEARCH = _expected_search()
GOOD_ARGUMENTS: dict[str, Any] = {
    "product": "sentinel-2-l2a",
    "bbox": FLORENCE,
    "start_date": "2025-07-01",
    "end_date": "2025-07-31",
    "max_cloud_cover": 10,
}
IDS = ["S2A_T32TPP_20250729T100051_L2A", "S2B_T32TPP_20250724T100049_L2A"]


@pytest.fixture(scope="module")
def gazetteer() -> Gazetteer:
    return Gazetteer.load()


def trace(
    *, arguments: dict[str, Any] = GOOD_ARGUMENTS, cited: list[str] | None = None, **fields: object
) -> TurnTrace:
    """A turn where the system resolved the place and searched, with any part replaced."""
    calls = [
        ToolCall(name="resolve_place", arguments={"name": "Florence"}),
        ToolCall(name="search_scenes", arguments=arguments, returned_ids=IDS),
    ]
    values: dict[str, Any] = {
        "route": "delegate",
        "calls": calls,
        "status": "ok",
        "cited_ids": cited or IDS[:1],
    }
    values.update(fields)
    return TurnTrace.model_validate(values)


def test_bbox_iou_of_identical_and_disjoint_boxes() -> None:
    box = (12.35, 41.78, 12.65, 42.00)
    assert bbox_iou(box, box) == pytest.approx(1.0)
    assert bbox_iou(box, (0.0, 0.0, 1.0, 1.0)) == 0.0


def test_a_correct_turn_passes(gazetteer: Gazetteer) -> None:
    score = score_turn(EXPECT, trace(), gazetteer)
    assert score.passed
    assert score.arguments is not None
    assert score.arguments.all


@pytest.mark.parametrize(
    ("change", "failing"),
    [
        ({"product": "sentinel-1-grd"}, "product"),
        ({"bbox": [11.0, 43.6, 11.5, 44.0]}, "bbox"),  # overlaps Florence, but not 95%
        ({"start_date": "2025-07-02"}, "dates"),
        ({"max_cloud_cover": 20}, "cloud"),  # cloud-free accepts 0 to 10 only
        ({"surprise": True}, "no_extra"),
    ],
)
def test_each_wrong_argument_is_caught(gazetteer: Gazetteer, change: dict[str, Any], failing: str) -> None:
    score = score_arguments(SEARCH, {**GOOD_ARGUMENTS, **change}, gazetteer)
    assert not getattr(score, failing)
    assert not score.all


def test_any_cloud_limit_inside_the_accepted_range_is_correct(gazetteer: Gazetteer) -> None:
    assert score_arguments(SEARCH, {**GOOD_ARGUMENTS, "max_cloud_cover": 5}, gazetteer).cloud


def test_a_cloud_limit_where_none_was_asked_for_is_wrong(gazetteer: Gazetteer) -> None:
    no_cloud = EXPECT.model_copy(update={"search": SEARCH.model_copy(update={"cloud": None})})
    assert no_cloud.search is not None
    assert not score_arguments(no_cloud.search, GOOD_ARGUMENTS, gazetteer).cloud
    without = {k: v for k, v in GOOD_ARGUMENTS.items() if k != "max_cloud_cover"}
    assert score_arguments(no_cloud.search, without, gazetteer).cloud


def test_malformed_arguments_fail_instead_of_crashing(gazetteer: Gazetteer) -> None:
    bad = {"product": "sentinel-2-l2a", "bbox": "Florence", "start_date": "July", "end_date": None}
    score = score_arguments(SEARCH, bad, gazetteer)
    assert not score.bbox
    assert not score.dates


def test_the_last_successful_search_is_scored_and_the_first_reported(gazetteer: Gazetteer) -> None:
    calls = [
        ToolCall(name="resolve_place"),
        ToolCall(name="search_scenes", arguments={**GOOD_ARGUMENTS, "max_cloud_cover": 50}),
        ToolCall(name="search_scenes", arguments=GOOD_ARGUMENTS, returned_ids=IDS),
    ]
    score = score_turn(EXPECT, trace(calls=calls), gazetteer)
    assert score.passed
    assert score.first_arguments is not None
    assert not score.first_arguments.cloud


def test_a_search_that_never_succeeded_fails(gazetteer: Gazetteer) -> None:
    calls = [
        ToolCall(name="resolve_place"),
        ToolCall(name="search_scenes", arguments=GOOD_ARGUMENTS, ok=False),
    ]
    assert not score_turn(EXPECT, trace(calls=calls, cited=[]), gazetteer).passed


def test_a_fabricated_scene_id_fails_the_turn(gazetteer: Gazetteer) -> None:
    score = score_turn(EXPECT, trace(cited=["S2A_T32TPP_20250101T000000_L2A"]), gazetteer)
    assert not score.grounded
    assert not score.passed


def test_ids_from_an_earlier_turn_count_as_grounded(gazetteer: Gazetteer) -> None:
    earlier = "S2A_T32TPP_20250601T100051_L2A"
    assert score_turn(EXPECT, trace(cited=[earlier]), gazetteer, known_ids=frozenset({earlier})).grounded


def test_tools_must_come_in_order_and_forbidden_tools_fail(gazetteer: Gazetteer) -> None:
    assert tools_in_order(
        ["list_products", "resolve_place", "search_scenes"], ["resolve_place", "search_scenes"]
    )
    assert not tools_in_order(["search_scenes", "resolve_place"], ["resolve_place", "search_scenes"])
    decline = Expectation.model_validate(
        {"route": "decline", "forbidden_tools": ["search_scenes", "get_scene"]}
    )
    assert not score_turn(decline, trace(route="decline", status="declined", cited=[]), gazetteer).tools


def test_route_and_outcome_accept_any_listed_alternative(gazetteer: Gazetteer) -> None:
    either = Expectation.model_validate({"route": ["delegate", "decline"], "outcomes": ["error", "declined"]})
    declined = TurnTrace(route=Route.DECLINE, status="declined")
    assert score_turn(either, declined, gazetteer).passed
    invented = TurnTrace(route=Route.DELEGATE, status="ok", cited_ids=["S2A_T31UDQ_20140110T000000_L2A"])
    assert not score_turn(either, invented, gazetteer).passed


def test_retrieval_metrics() -> None:
    assert retrieval_metrics(IDS, IDS, limit=10) == (1.0, 1.0)
    assert retrieval_metrics(IDS[:1], IDS, limit=10) == (1.0, 0.5)
    assert retrieval_metrics(IDS[:1], IDS, limit=1) == (1.0, 1.0)  # recall only up to the requested limit
    assert retrieval_metrics(["not-in-oracle"], IDS, limit=10) == (0.0, 0.0)
    assert retrieval_metrics([], [], limit=10) == (1.0, 1.0)  # nothing exists, nothing returned: correct
