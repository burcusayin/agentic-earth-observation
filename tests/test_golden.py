"""Tests for the golden-set format and its validation rules."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from eo_agent_demo.catalog.gazetteer import Gazetteer
from eo_agent_demo.evals.golden import Task, check_tasks, load_golden


def task(**overrides: object) -> dict[str, Any]:
    """A valid single-search task as a dict, with any top-level field replaced."""
    fields: dict[str, Any] = {
        "id": "s2-florence-july",
        "split": "test",
        "category": "single_search",
        "turns": [
            {
                "user": "Find cloud-free Sentinel-2 images of Florence from July 2025.",
                "expect": {
                    "route": "delegate",
                    "required_tools": ["resolve_place", "search_scenes"],
                    "search": {
                        "product": "sentinel-2-l2a",
                        "place": "florence",
                        "start": "2025-07-01",
                        "end": "2025-07-31",
                        "cloud": [0, 10],
                    },
                },
            }
        ],
    }
    fields.update(overrides)
    return fields


def expect(**overrides: object) -> dict[str, Any]:
    return {**task()["turns"][0]["expect"], **overrides}


def one_turn(expectation: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"user": "a question", "expect": expectation}]


@pytest.fixture(scope="module")
def gazetteer() -> Gazetteer:
    return Gazetteer.load()


def test_accepts_a_valid_task(gazetteer: Gazetteer) -> None:
    assert check_tasks([Task.model_validate(task())], gazetteer) == []


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"id": "Has Spaces"}, "string_pattern_mismatch"),
        ({"split": "train"}, "literal_error"),
        ({"category": "trivia"}, "enum"),
        ({"surprise": 1}, "extra_forbidden"),
        ({"category": "follow_up"}, "two turns"),
    ],
)
def test_rejects_malformed_tasks(overrides: dict[str, object], reason: str) -> None:
    with pytest.raises(ValidationError, match=reason):
        Task.model_validate(task(**overrides))


def test_rejects_a_cloud_limit_on_radar() -> None:
    search = {**expect()["search"], "product": "sentinel-1-grd"}
    with pytest.raises(ValidationError, match="Sentinel-1"):
        Task.model_validate(task(turns=one_turn(expect(search=search))))


def test_rejects_an_upside_down_cloud_range() -> None:
    search = {**expect()["search"], "cloud": [20, 10]}
    with pytest.raises(ValidationError, match="range"):
        Task.model_validate(task(turns=one_turn(expect(search=search))))


def test_a_decline_turn_must_forbid_searching() -> None:
    declined = {"route": "decline", "forbidden_tools": []}
    with pytest.raises(ValidationError, match="must forbid search_scenes"):
        Task.model_validate(task(category="out_of_scope", turns=one_turn(declined)))


def test_a_tool_cannot_be_both_required_and_forbidden() -> None:
    contradictory = expect(forbidden_tools=["resolve_place"])
    with pytest.raises(ValidationError, match="both required and forbidden"):
        Task.model_validate(task(turns=one_turn(contradictory)))


def test_finds_an_unknown_place_key(gazetteer: Gazetteer) -> None:
    search = {**expect()["search"], "place": "atlantis"}
    problems = check_tasks([Task.model_validate(task(turns=one_turn(expect(search=search))))], gazetteer)
    assert problems == ["s2-florence-july turn 1: 'atlantis' is not a gazetteer key"]


def test_finds_an_expected_search_that_the_tools_would_reject(gazetteer: Gazetteer) -> None:
    search = {**expect()["search"], "start": "2015-01-01", "end": "2015-01-31"}  # before Sentinel-2 launched
    problems = check_tasks([Task.model_validate(task(turns=one_turn(expect(search=search))))], gazetteer)
    assert len(problems) == 1
    assert "2015-06-23" in problems[0]


def test_finds_duplicate_ids(gazetteer: Gazetteer) -> None:
    same = Task.model_validate(task())
    assert "used 2 times" in check_tasks([same, same], gazetteer)[0]


def test_the_committed_golden_set_is_valid_and_complete() -> None:
    tasks = load_golden()
    assert sum(t.split == "dev" for t in tasks) == 4
    assert sum(t.split == "test" for t in tasks) == 16


def test_dev_and_test_use_different_places() -> None:
    # Prompts may be tuned on dev, so dev must not share places with test (no leakage into the prompts).
    def places(split: str) -> set[str]:
        return {
            turn.expect.search.place
            for t in load_golden()
            if t.split == split
            for turn in t.turns
            if turn.expect.search is not None
        }

    assert places("dev").isdisjoint(places("test"))
