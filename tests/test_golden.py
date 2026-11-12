"""Tests for the golden-set format and its validation rules."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from agentic_earth_observation.catalog.gazetteer import Gazetteer
from agentic_earth_observation.config import ROOT
from agentic_earth_observation.evals import golden
from agentic_earth_observation.evals.golden import (
    GoldenError,
    Task,
    check_tasks,
    evaluation_code_hash,
    golden_hash,
    load_golden,
    load_tasks,
)
from agentic_earth_observation.evals.groups import added_in_v2, relative_period
from agentic_earth_observation.evals.run_eval import (
    FROZEN_EVALUATION_CODE,
    FROZEN_EVALUATION_RULES,
    FROZEN_GAZETTEER,
    FROZEN_TEST_SPLIT,
    FROZEN_TEST_SPLIT_V1,
    TEST_TASKS_V1,
)


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
    assert sum(t.split == "test" for t in tasks) == 50


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


def test_the_frozen_files_are_as_they_were_at_the_tag() -> None:
    # A fix to any of them needs a dated decision and a new tag, and then new values in run_eval.py. The
    # literal digests here are the ones taken at the tag; the constants must stay equal to them. The dev split
    # is not frozen, so the hash over both splits is pinned here only to show when it last changed.
    assert golden_hash() == "fe7aa7979bcdd77ecdfe1ee40dc34ae9304dd949048c26d577df8658baa7907b"
    assert (
        evaluation_code_hash()
        == FROZEN_EVALUATION_CODE
        == "72d140b004d9a42eaa017ea53db68334407cea46a82b5d4b871ff45d94c3b9cb"
    )
    for path, constant, frozen in (
        (
            "docs/evaluation.md",
            FROZEN_EVALUATION_RULES,
            "98cd80541fcd11e77834a9bda3edd4868af3a2c747df11996d3d4a33112dc467",
        ),
        (
            "data/gazetteer.json",
            FROZEN_GAZETTEER,
            "af181e26cb48c0631f8e3e70bbe71aa7ef2e12d59b1bae0f1b141d76f842d91e",
        ),
        (
            "data/golden/test.jsonl",
            FROZEN_TEST_SPLIT,
            "7f4847562d2e6c43f91b74945201d3cc03cb99091fdcb923764060a41582ea6c",
        ),
    ):
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == constant == frozen, path


def test_the_test_split_of_v2_begins_with_that_of_v1_unchanged() -> None:
    # The results of v1.0 are on these 18 tasks, and a comparison with them needs the very same ones.
    lines = (ROOT / "data/golden/test.jsonl").read_bytes().splitlines(keepends=True)
    first = hashlib.sha256(b"".join(lines[:TEST_TASKS_V1])).hexdigest()
    assert first == FROZEN_TEST_SPLIT_V1 == "86123f066e640af1464347956058eac5a9c04799f5ce7400549fb9422b16f784"
    assert len(lines) == 50


def test_the_ids_of_the_test_split_tell_the_tasks_of_v1_from_those_added_in_v2() -> None:
    # A report groups the tasks by their ids (evals/groups.py), since a task has no field for it.
    test = [task for task in load_golden() if task.split == "test"]
    assert [added_in_v2(task.id) for task in test] == [False] * TEST_TASKS_V1 + [True] * 32
    periods = Counter(period for task in test if (period := relative_period(task.id)) is not None)
    assert periods == {
        "last-month": 2,
        "last-spring": 1,
        "last-summer": 1,
        "last-autumn": 2,
        "last-winter": 2,
    }
    assert not any(added_in_v2(task.id) for task in load_golden() if task.split == "dev")


def test_the_labelling_rules_and_the_scoring_are_those_of_v1_word_for_word() -> None:
    # docs/evaluation.md gained the tasks of v2 and the rules for comparing setups. What decides an expected
    # answer and a score did not change: this is the digest of that part of the file at tag golden-test-v1.
    text = (ROOT / "docs/evaluation.md").read_text(encoding="utf-8")
    part = text[text.index("## Labelling rules") : text.index("## Runs and statistics")]
    assert "## Task format" in part
    assert "## Scoring" in part
    assert hashlib.sha256(part.encode()).hexdigest() == (
        "66524fdd71953d2e77a239dce5a29fca8ea119c20aea5cfc90a8480791c67f62"
    )


def test_the_evaluation_code_hash_is_stable_and_well_formed() -> None:
    first = evaluation_code_hash()
    assert first == evaluation_code_hash()
    assert len(first) == 64
    assert all(c in "0123456789abcdef" for c in first)


class TestGapsFoundInTheNovemberReview:
    """Validation rules that had no test when they were frozen."""

    @pytest.mark.parametrize(
        ("change", "reason"),
        [
            ({"cloud": [0, 120]}, "within 0-100"),
            ({"cloud": [-5, 10]}, "within 0-100"),
        ],
    )
    def test_rejects_a_cloud_range_outside_0_to_100(self, change: dict[str, object], reason: str) -> None:
        search = {**expect()["search"], **change}
        with pytest.raises(ValidationError, match=reason):
            Task.model_validate(task(turns=one_turn(expect(search=search))))

    @pytest.mark.parametrize(
        ("expectation", "reason"),
        [
            ({"required_tools": ["resolve_place"]}, "needs search_scenes in required_tools"),
            ({"route": []}, "at least one route"),
            (
                {"route": "decline", "required_tools": [], "forbidden_tools": ["search_scenes"]},
                "cannot expect a search",
            ),
        ],
    )
    def test_rejects_an_expectation_that_contradicts_itself(
        self, expectation: dict[str, object], reason: str
    ) -> None:
        with pytest.raises(ValidationError, match=reason):
            Task.model_validate(task(turns=one_turn(expect(**expectation))))

    def test_only_a_follow_up_has_more_than_one_turn(self) -> None:
        turn = task()["turns"][0]
        with pytest.raises(ValidationError, match="exactly one turn"):
            Task.model_validate(task(turns=[turn, turn]))

    def test_finds_an_alias_used_as_a_place_key(self, gazetteer: Gazetteer) -> None:
        search = {**expect()["search"], "place": "firenze"}  # an alias of the key "florence"
        problems = check_tasks([Task.model_validate(task(turns=one_turn(expect(search=search))))], gazetteer)
        assert problems == ["s2-florence-july turn 1: 'firenze' is not a gazetteer key"]

    def test_finds_a_search_that_ends_after_the_reference_date(self, gazetteer: Gazetteer) -> None:
        search = {**expect()["search"], "start": "2025-12-01", "end": "2025-12-31"}
        problems = check_tasks([Task.model_validate(task(turns=one_turn(expect(search=search))))], gazetteer)
        assert problems == ["s2-florence-july turn 1: ends after the reference date 2025-12-15"]

    def test_finds_a_task_in_the_wrong_split_file(self, tmp_path: Path) -> None:
        (tmp_path / "dev.jsonl").write_text(json.dumps(task(split="test")) + "\n", encoding="utf-8")
        (tmp_path / "test.jsonl").write_text("", encoding="utf-8")
        with pytest.raises(GoldenError, match=r"dev\.jsonl contains tasks of another split"):
            load_golden(tmp_path)

    def test_a_bad_file_names_every_invalid_line(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        path.write_text(json.dumps(task()) + "\n\n" + '{"id": "x"}\n' + "not json\n", encoding="utf-8")
        with pytest.raises(GoldenError) as raised:
            load_tasks(path)
        assert "test.jsonl:3" in str(raised.value)
        assert "test.jsonl:4" in str(raised.value)
        assert "test.jsonl:1" not in str(raised.value)
        with pytest.raises(GoldenError, match="cannot read"):
            load_tasks(tmp_path / "missing.jsonl")

    def test_the_golden_hash_names_the_exact_files(self, tmp_path: Path) -> None:
        for split in ("dev", "test"):
            (tmp_path / f"{split}.jsonl").write_text(json.dumps(task(split=split)) + "\n", encoding="utf-8")
        both = (tmp_path / "dev.jsonl").read_bytes() + (tmp_path / "test.jsonl").read_bytes()
        assert golden_hash(tmp_path) == hashlib.sha256(both).hexdigest()
        (tmp_path / "test.jsonl").write_text(
            json.dumps(task(split="test", notes="edited")) + "\n", encoding="utf-8"
        )
        assert golden_hash(tmp_path) != hashlib.sha256(both).hexdigest()

    def test_the_evaluation_code_hash_covers_validation_and_scoring(self) -> None:
        folder = Path(golden.__file__).parent
        code = (folder / "golden.py").read_bytes() + (folder / "scoring.py").read_bytes()
        assert evaluation_code_hash() == hashlib.sha256(code).hexdigest()
