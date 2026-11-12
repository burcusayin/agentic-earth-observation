"""Tests for the checks on a file of candidate tasks (evals/candidates.py)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from agentic_earth_observation.catalog.gazetteer import Gazetteer
from agentic_earth_observation.config import Settings
from agentic_earth_observation.domain import Product
from agentic_earth_observation.evals import candidates
from agentic_earth_observation.evals.candidates import (
    SecondLabel,
    differences,
    live_check,
    period_dates,
    problems,
    questions,
)
from agentic_earth_observation.evals.golden import (
    REFERENCE_DATE,
    ExpectedSearch,
    GoldenError,
    Task,
    load_golden,
)
from agentic_earth_observation.evals.groups import PERIODS
from agentic_earth_observation.evals.run_eval import ORACLE_LIMIT, oracle_request
from tests.fakes import FakeCatalog

GAZETTEER = Gazetteer.load()
ROME = [12.2344, 41.6556, 12.8558, 42.141]
QUESTION = "Find cloud-free Sentinel-2 images of Rome from July 2025."
TASK_ID = "v2-single-example-for-the-tests"
NOTE = "Live check 2026-11-12: the expected search returns 7 scenes."
SEARCH = {
    "product": "sentinel-2-l2a",
    "place": "rome",
    "start": "2025-07-01",
    "end": "2025-07-31",
    "cloud": [0, 10],
}


def expectation(**overrides: object) -> dict[str, Any]:
    """What a turn that searches expects, with any field replaced."""
    fields: dict[str, Any] = {
        "route": "delegate",
        "required_tools": ["resolve_place", "search_scenes"],
        "search": dict(SEARCH),
    }
    fields.update(overrides)
    return fields


def candidate(**overrides: object) -> dict[str, Any]:
    """A valid new single-search task as a dict, with any top-level field replaced."""
    fields: dict[str, Any] = {
        "id": TASK_ID,
        "split": "test",
        "category": "single_search",
        "turns": [{"user": QUESTION, "expect": expectation()}],
        "notes": NOTE,
    }
    fields.update(overrides)
    return fields


def declined(**overrides: object) -> dict[str, Any]:
    """A valid new task that expects no search."""
    expect = {"route": "decline", "forbidden_tools": ["search_scenes", "get_scene"], "outcomes": ["declined"]}
    turns = [{"user": "What is the weather in Rome?", "expect": expect}]
    return candidate(id="v2-oos-example-for-the-tests", category="out_of_scope", turns=turns, notes="")


def follow_up() -> dict[str, Any]:
    """A valid new task of two turns, each with its search."""
    second = expectation(required_tools=["search_scenes"], search={**SEARCH, "end": "2025-07-15", "limit": 3})
    turns = [{"user": QUESTION, "expect": expectation()}, {"user": "Only three.", "expect": second}]
    return candidate(id="v2-follow-example-for-the-tests", category="follow_up", turns=turns)


def relative(
    period: str = "last-month", start: str = "2025-11-01", end: str = "2025-11-30", **overrides: object
) -> dict[str, Any]:
    """A valid new relative-date task: its search has the dates its period means on the reference date."""
    expect = expectation(search={**SEARCH, "start": start, "end": end})
    fields: dict[str, Any] = {
        "id": f"v2-relative-{period}-rome",
        "category": "robustness",
        "turns": [{"user": f"Sentinel-2 images of Rome from {period.replace('-', ' ')}.", "expect": expect}],
    }
    fields.update(overrides)
    return candidate(**fields)


def tasks(*fields: dict[str, Any]) -> list[Task]:
    return [Task.model_validate(one) for one in fields]


def write(folder: Path, *fields: dict[str, Any], name: str = "candidates.jsonl") -> Path:
    path = folder / name
    path.write_text("".join(json.dumps(one) + "\n" for one in fields), encoding="utf-8")
    return path


class TestTheOracleRequest:
    def test_it_asks_for_the_top_of_the_cloud_range_and_up_to_fifty_scenes(self) -> None:
        request = oracle_request(ExpectedSearch.model_validate(SEARCH), GAZETTEER)
        assert request.product is Product.SENTINEL_2_L2A
        assert request.bbox.as_list() == ROME
        assert (str(request.dates.start), str(request.dates.end)) == ("2025-07-01", "2025-07-31")
        assert request.max_cloud_cover == 10
        assert request.limit == ORACLE_LIMIT == 50

    def test_it_keeps_the_number_of_scenes_a_task_asks_for(self) -> None:
        request = oracle_request(ExpectedSearch.model_validate({**SEARCH, "limit": 3}), GAZETTEER)
        assert request.limit == 3

    def test_no_cloud_range_means_no_cloud_limit(self) -> None:
        radar = {**SEARCH, "product": "sentinel-1-grd", "cloud": None}
        assert oracle_request(ExpectedSearch.model_validate(radar), GAZETTEER).max_cloud_cover is None

    def test_a_place_the_gazetteer_does_not_have_is_refused(self) -> None:
        search = ExpectedSearch.model_validate({**SEARCH, "place": "atlantis"})
        with pytest.raises(LookupError, match="'atlantis' is not in the gazetteer"):
            oracle_request(search, GAZETTEER)


class TestPeriodDates:
    """docs/evaluation.md: "last month" is the previous calendar month, "last <season>" the most recent
    complete meteorological season before today."""

    @pytest.mark.parametrize(
        ("period", "start", "end"),
        [
            ("last-month", date(2025, 11, 1), date(2025, 11, 30)),
            ("last-spring", date(2025, 3, 1), date(2025, 5, 31)),
            ("last-summer", date(2025, 6, 1), date(2025, 8, 31)),
            ("last-autumn", date(2025, 9, 1), date(2025, 11, 30)),
            ("last-winter", date(2024, 12, 1), date(2025, 2, 28)),
        ],
    )
    def test_the_periods_on_the_reference_date(self, period: str, start: date, end: date) -> None:
        assert date(2025, 12, 15) == REFERENCE_DATE
        assert period_dates(period, REFERENCE_DATE) == (start, end)

    @pytest.mark.parametrize(
        ("today", "start", "end"),
        [
            (date(2026, 1, 10), date(2025, 12, 1), date(2025, 12, 31)),  # across the turn of the year
            (
                date(2024, 3, 31),
                date(2024, 2, 1),
                date(2024, 2, 29),
            ),  # a leap year, from a day February lacks
            (date(2025, 12, 1), date(2025, 11, 1), date(2025, 11, 30)),
        ],
    )
    def test_last_month_is_the_calendar_month_before_todays(
        self, today: date, start: date, end: date
    ) -> None:
        assert period_dates("last-month", today) == (start, end)

    @pytest.mark.parametrize(
        ("period", "today", "start", "end"),
        [
            (
                "last-summer",
                date(2025, 9, 1),
                date(2025, 6, 1),
                date(2025, 8, 31),
            ),  # complete since yesterday
            ("last-summer", date(2025, 8, 31), date(2024, 6, 1), date(2024, 8, 31)),  # its last day: not yet
            ("last-summer", date(2025, 7, 15), date(2024, 6, 1), date(2024, 8, 31)),  # in the middle of one
            ("last-spring", date(2025, 3, 10), date(2024, 3, 1), date(2024, 5, 31)),
            ("last-autumn", date(2025, 12, 1), date(2025, 9, 1), date(2025, 11, 30)),
            ("last-autumn", date(2025, 11, 30), date(2024, 9, 1), date(2024, 11, 30)),
            ("last-winter", date(2025, 3, 1), date(2024, 12, 1), date(2025, 2, 28)),
            # In January the winter that began in December is not over, so the one before it is the last.
            ("last-winter", date(2025, 1, 15), date(2023, 12, 1), date(2024, 2, 29)),
        ],
    )
    def test_last_season_is_the_latest_one_whose_last_day_is_before_today(
        self, period: str, today: date, start: date, end: date
    ) -> None:
        assert period_dates(period, today) == (start, end)

    def test_every_period_an_id_can_name_has_dates(self) -> None:
        for period in PERIODS:
            start, end = period_dates(period, REFERENCE_DATE)
            assert start < end < REFERENCE_DATE

    def test_a_period_the_rules_do_not_define_is_refused(self) -> None:
        with pytest.raises(ValueError, match="last-week is not one of the periods last-month, last-spring"):
            period_dates("last-week", REFERENCE_DATE)


class TestProblems:
    def test_valid_candidates_have_none(self) -> None:
        assert problems(tasks(candidate(), declined(), follow_up()), load_golden(), GAZETTEER) == []

    @pytest.mark.parametrize(
        ("changes", "problem"),
        [
            ({"id": "single-example-for-the-tests"}, "the id of a new task starts with v2-"),
            ({"split": "dev"}, "a candidate is a test task, not a dev one"),
            ({"notes": ""}, "its notes do not say what the live check found"),
            ({"notes": "The expected search returns 7 scenes."}, "its notes do not say what the live check"),
            ({"notes": "Live check: 7 scenes."}, "its notes do not say what the live check"),  # no date
            ({"id": "v2-relative-next-week-rome"}, "names none of the periods"),
        ],
    )
    def test_a_candidate_that_breaks_a_rule_is_named_with_the_rule(
        self, changes: dict[str, object], problem: str
    ) -> None:
        (found,) = problems(tasks(candidate(**changes)), load_golden(), GAZETTEER)
        assert problem in found
        assert str(changes.get("id", TASK_ID)) in found

    @pytest.mark.parametrize(
        ("period", "start", "end"),
        [
            ("last-month", "2025-11-01", "2025-11-30"),
            ("last-spring", "2025-03-01", "2025-05-31"),
            ("last-summer", "2025-06-01", "2025-08-31"),
            ("last-autumn", "2025-09-01", "2025-11-30"),
            ("last-winter", "2024-12-01", "2025-02-28"),
        ],
    )
    def test_a_relative_date_task_with_its_periods_dates_is_fine(
        self, period: str, start: str, end: str
    ) -> None:
        assert problems(tasks(relative(period, start, end)), load_golden(), GAZETTEER) == []

    def test_a_relative_date_task_of_another_category_is_refused(self) -> None:
        (found,) = problems(tasks(relative(category="single_search")), load_golden(), GAZETTEER)
        assert found == (
            "v2-relative-last-month-rome: a relative-date task is a robustness task, not a single_search one"
        )

    @pytest.mark.parametrize(
        ("start", "end"),
        [
            ("2024-06-01", "2024-08-31"),  # the summer a year early
            ("2025-06-02", "2025-08-31"),
            ("2025-06-01", "2025-08-30"),
        ],
    )
    def test_a_relative_date_task_with_other_dates_than_its_periods_is_refused(
        self, start: str, end: str
    ) -> None:
        (found,) = problems(tasks(relative("last-summer", start, end)), load_golden(), GAZETTEER)
        assert found == (
            "v2-relative-last-summer-rome: last-summer is 2025-06-01 to 2025-08-31 on 2025-12-15, "
            f"but its search expects {start} to {end}"
        )

    def test_a_relative_date_task_that_expects_no_search_is_refused(self) -> None:
        asks = {"route": "clarify", "forbidden_tools": ["search_scenes"], "outcomes": ["clarified"]}
        turns = [{"user": "Images of Rome from last summer.", "expect": asks}]
        (found,) = problems(tasks(relative("last-summer", turns=turns, notes="")), load_golden(), GAZETTEER)
        assert found == "v2-relative-last-summer-rome: a relative-date task expects a search"

    def test_a_place_of_the_dev_split_is_refused(self) -> None:
        # Prompts are tuned on the dev split, so a test task must not share a place with it.
        milan = expectation(search={**SEARCH, "place": "milan"})
        (found,) = problems(
            tasks(candidate(turns=[{"user": QUESTION, "expect": milan}])), load_golden(), GAZETTEER
        )
        assert found == f"{TASK_ID}: milan is a place of the dev split"

    def test_an_id_that_a_task_already_has_is_refused(self) -> None:
        existing = tasks(candidate())
        (found,) = problems(tasks(candidate()), existing, GAZETTEER)
        assert found == f"{TASK_ID}: the id is already a task's"

    def test_the_golden_sets_own_checks_apply(self) -> None:
        unknown = expectation(search={**SEARCH, "place": "atlantis"})
        twice = tasks(candidate(), candidate(turns=[{"user": QUESTION, "expect": unknown}]))
        found = problems(twice, [], GAZETTEER)
        assert f"{TASK_ID}: the id is used 2 times" in found
        assert f"{TASK_ID} turn 1: 'atlantis' is not a gazetteer key" in found

    def test_a_task_without_a_search_needs_no_note(self) -> None:
        assert problems(tasks(declined()), load_golden(), GAZETTEER) == []


class TestLiveCheck:
    def test_every_expected_search_is_asked_as_the_oracle_asks_it(self) -> None:
        catalog = FakeCatalog()
        pauses: list[None] = []
        results = live_check(
            tasks(candidate(), declined(), follow_up()), catalog, GAZETTEER, pause=lambda: pauses.append(None)
        )
        assert [(r.task_id, r.turn, r.scenes, r.failed) for r in results] == [
            (TASK_ID, 1, 1, None),
            ("v2-follow-example-for-the-tests", 1, 1, None),
            ("v2-follow-example-for-the-tests", 2, 1, None),
        ]
        first, _, last = catalog.requests
        assert first == oracle_request(ExpectedSearch.model_validate(SEARCH), GAZETTEER)
        assert (str(last.dates.end), last.limit) == ("2025-07-15", 3)
        assert len(pauses) == 2  # between the searches, not before the first: a public service

    def test_a_search_that_finds_nothing_is_counted_as_such(self) -> None:
        (result,) = live_check(tasks(candidate()), FakeCatalog([]), GAZETTEER, pause=lambda: None)
        assert (result.scenes, result.failed) == (0, None)

    def test_a_catalogue_that_fails_is_reported_not_raised(self) -> None:
        results = live_check(
            tasks(candidate(), follow_up()), FakeCatalog(fail=True), GAZETTEER, pause=lambda: None
        )
        assert [r.scenes for r in results] == [None, None, None]
        assert all(r.failed and "the catalogue did not answer" in r.failed for r in results)


class TestQuestions:
    def test_they_are_the_users_words_under_a_number_and_nothing_else(self) -> None:
        asked = questions(tasks(candidate(), follow_up()))
        assert asked == [{"n": 1, "turns": [QUESTION]}, {"n": 2, "turns": [QUESTION, "Only three."]}]
        # The id of a relative-date task names its period, which is the answer the second labelling looks for.
        assert TASK_ID not in json.dumps(asked)


def label(number: int, *turns: dict[str, Any]) -> SecondLabel:
    return SecondLabel.model_validate({"n": number, "turns": list(turns)})


class TestDifferences:
    def test_the_same_labels_do_not_differ(self) -> None:
        both = tasks(candidate(), declined())
        second = [
            label(
                2,
                {
                    "route": "decline",
                    "forbidden_tools": ["get_scene", "search_scenes"],
                    "outcomes": ["declined"],
                },
            ),
            label(1, expectation(route=["delegate"])),  # a list of one route is that route
        ]
        assert differences(both, second) == []

    def test_several_accepted_routes_are_the_same_in_any_order(self) -> None:
        # An unknown place may be checked with the retrieval agent first, or asked about at once.
        asks = {"required_tools": ["resolve_place"], "forbidden_tools": ["search_scenes"]}
        first = {"route": ["delegate", "clarify"], "outcomes": ["clarified"], **asks}
        second = {"route": ["clarify", "delegate"], "outcomes": ["clarified"], **asks}
        unknown = candidate(category="clarify", turns=[{"user": "Scenes of Atlantis?", "expect": first}])
        assert differences(tasks(unknown), [label(1, second)]) == []

    @pytest.mark.parametrize(
        ("changes", "field"),
        [
            ({"search": {**SEARCH, "end": "2025-07-30"}}, "search"),
            ({"search": {**SEARCH, "cloud": [0, 20]}}, "search"),
            ({"route": ["delegate", "clarify"]}, "route"),
            ({"required_tools": ["search_scenes"]}, "required tools"),
            ({"required_tools": ["search_scenes", "resolve_place"]}, "required tools"),  # the order counts
            ({"forbidden_tools": ["get_scene"]}, "forbidden tools"),
            ({"outcomes": ["ok"]}, "outcomes"),
        ],
    )
    def test_a_label_that_differs_is_named_by_its_task_turn_and_field(
        self, changes: dict[str, object], field: str
    ) -> None:
        (found,) = differences(tasks(candidate()), [label(1, expectation(**changes))])
        assert (found.number, found.text.split(":")[0]) == (1, f"task 1 ({TASK_ID}) turn 1, {field}")

    def test_a_task_the_second_labelling_left_out_differs(self) -> None:
        (found,) = differences(tasks(candidate(), declined()), [label(1, expectation())])
        assert (found.number, found.text) == (
            2,
            "task 2 (v2-oos-example-for-the-tests): the second labelling has no entry for it",
        )

    def test_another_number_of_turns_differs(self) -> None:
        (found,) = differences(tasks(follow_up()), [label(1, expectation())])
        assert "1 turn(s) where the task has 2" in found.text

    def test_an_entry_without_a_task_and_an_entry_given_twice_are_named(self) -> None:
        second = [label(1, expectation()), label(1, expectation()), label(7, expectation())]
        found = differences(tasks(candidate()), second)
        assert [(d.number, d.text) for d in found] == [
            (1, "entry 1 of the second labelling is given 2 times"),
            (7, "entry 7 of the second labelling has no task"),
        ]


def run(*arguments: str | Path, catalog: FakeCatalog | None = None) -> int:
    return candidates.main(
        [str(argument) for argument in arguments], catalog=catalog or FakeCatalog(), pause_s=0
    )


class TestTheCommands:
    """Every command prints totals only: whoever writes prompts runs it and must not see a task."""

    def test_check_passes_valid_candidates_and_counts_them(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        file = write(tmp_path, candidate(), relative(), declined())
        details = tmp_path / "out" / "details.txt"
        assert run("check", file, "--details", details) == 0
        assert capsys.readouterr().out.splitlines() == [
            "tasks: 3 (out_of_scope 1, robustness 1, single_search 1)",
            "relative dates: 1 (last-month 1)",
            "problems: 0",
        ]
        assert details.read_text(encoding="utf-8") == "no problems\n"

    def test_check_counts_the_problems_and_keeps_them_out_of_what_it_prints(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        file = write(tmp_path, candidate(notes=""), candidate(id="v2-relative-rome"))
        details = tmp_path / "details.txt"
        assert run("check", file, "--details", details) == 1
        out = capsys.readouterr().out
        assert out.splitlines() == [
            "tasks: 2 (single_search 2)",
            "relative dates: cannot be counted, an id names no period",
            f"problems: 2; the details are in {details}",
        ]
        assert "rome" not in out.replace(str(details), "")
        written = details.read_text(encoding="utf-8")
        assert f"{TASK_ID}: its notes do not say" in written
        assert "v2-relative-rome starts with v2-relative- but names none of the periods" in written

    def test_a_file_that_cannot_be_read_as_tasks_is_said_without_its_text(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        file = tmp_path / "candidates.jsonl"
        file.write_text('{"id": "v2-broken", "user": "A secret question about Rome"}\n', encoding="utf-8")
        details = tmp_path / "details.txt"
        for command in (("check", file), ("live", file), ("compare", file, file)):
            assert run(*command, "--details", details) == 1
            out = capsys.readouterr().out
            assert out == f"the file cannot be read as tasks; the details are in {details}\n"
            assert "candidates.jsonl:1" in details.read_text(encoding="utf-8")
        assert run("questions", file, "--out", tmp_path / "questions.jsonl") == 1
        assert capsys.readouterr().out == "the file cannot be read as tasks; `check` writes the details\n"
        assert not (tmp_path / "questions.jsonl").exists()

    def test_live_counts_what_the_catalogue_found(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        file = write(tmp_path, candidate(), follow_up(), declined())
        details = tmp_path / "details.txt"
        assert run("live", file, "--details", details) == 0
        assert capsys.readouterr().out == (
            f"searches: 3; found scenes: 3; found none: 0; failed: 0; the details are in {details}\n"
        )
        assert details.read_text(encoding="utf-8").splitlines() == [
            f"{TASK_ID} turn 1: 1 scene(s)",
            "v2-follow-example-for-the-tests turn 1: 1 scene(s)",
            "v2-follow-example-for-the-tests turn 2: 1 scene(s)",
        ]

    @pytest.mark.parametrize(
        ("catalog", "counted", "written"),
        [
            (FakeCatalog([]), "found scenes: 0; found none: 1; failed: 0", "turn 1: 0 scene(s)"),
            (
                FakeCatalog(fail=True),
                "found scenes: 0; found none: 0; failed: 1",
                "turn 1: failed: scene search",
            ),
        ],
    )
    def test_live_fails_when_a_search_finds_nothing_or_the_catalogue_fails(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        catalog: FakeCatalog,
        counted: str,
        written: str,
    ) -> None:
        details = tmp_path / "details.txt"
        assert run("live", write(tmp_path, candidate()), "--details", details, catalog=catalog) == 1
        assert f"searches: 1; {counted}" in capsys.readouterr().out
        assert written in details.read_text(encoding="utf-8")

    def test_live_asks_nothing_of_the_catalogue_for_candidates_the_golden_set_refuses(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        unknown = expectation(search={**SEARCH, "place": "atlantis"})
        file = write(tmp_path, candidate(turns=[{"user": QUESTION, "expect": unknown}]))
        catalog = FakeCatalog()
        details = tmp_path / "details.txt"
        assert run("live", file, "--details", details, catalog=catalog) == 1
        assert capsys.readouterr().out == (
            f"problems: 1; nothing was searched; the details are in {details}\n"
        )
        assert catalog.requests == []
        assert "'atlantis' is not a gazetteer key" in details.read_text(encoding="utf-8")

    def test_questions_writes_the_questions_alone(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "second" / "questions.jsonl"
        assert run("questions", write(tmp_path, candidate(), follow_up()), "--out", out) == 0
        assert capsys.readouterr().out == f"tasks: 2; the questions are in {out}\n"
        lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        assert lines == [{"n": 1, "turns": [QUESTION]}, {"n": 2, "turns": [QUESTION, "Only three."]}]

    def test_compare_counts_the_tasks_that_agree_and_differ(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        file = write(tmp_path, candidate(), declined())
        other_dates = expectation(search={**SEARCH, "start": "2025-07-02", "cloud": [0, 20]})
        second = tmp_path / "second.jsonl"
        second.write_text(json.dumps({"n": 1, "turns": [other_dates]}) + "\n", encoding="utf-8")
        details = tmp_path / "details.txt"
        assert run("compare", file, second, "--details", details) == 1
        assert capsys.readouterr().out == (
            f"tasks compared: 2; agree: 0; differ: 2; the details are in {details}\n"
        )
        written = details.read_text(encoding="utf-8").splitlines()
        assert written[0].startswith(f"task 1 ({TASK_ID}) turn 1, search: ")
        assert written[1] == "task 2 (v2-oos-example-for-the-tests): the second labelling has no entry for it"

    def test_compare_passes_labels_that_agree(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        second = tmp_path / "second.jsonl"
        second.write_text(json.dumps({"n": 1, "turns": [expectation()]}) + "\n", encoding="utf-8")
        details = tmp_path / "details.txt"
        assert run("compare", write(tmp_path, candidate()), second, "--details", details) == 0
        assert capsys.readouterr().out == "tasks compared: 1; agree: 1; differ: 0\n"
        assert details.read_text(encoding="utf-8") == "no differences\n"

    def test_compare_fails_on_an_entry_without_a_task_and_counts_it_apart(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        second = tmp_path / "second.jsonl"
        entries = [{"n": 1, "turns": [expectation()]}, {"n": 7, "turns": [expectation()]}]
        second.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
        details = tmp_path / "details.txt"
        assert run("compare", write(tmp_path, candidate()), second, "--details", details) == 1
        assert capsys.readouterr().out == (
            "tasks compared: 1; agree: 1; differ: 0; entries without a task: 1; "
            f"the details are in {details}\n"
        )
        assert details.read_text(encoding="utf-8") == "entry 7 of the second labelling has no task\n"

    def test_second_labels_that_cannot_be_read_are_said_without_their_text(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        second = tmp_path / "second.jsonl"
        second.write_text('{"n": 1, "turns": "a secret answer"}\n', encoding="utf-8")
        details = tmp_path / "details.txt"
        assert run("compare", write(tmp_path, candidate()), second, "--details", details) == 1
        assert capsys.readouterr().out == (
            f"the second labelling cannot be read; the details are in {details}\n"
        )
        assert "second.jsonl:1" in details.read_text(encoding="utf-8")

    def test_a_second_labelling_may_have_empty_lines_and_a_missing_one_is_said(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        file = write(tmp_path, candidate())
        second = tmp_path / "second.jsonl"
        second.write_text("\n" + json.dumps({"n": 1, "turns": [expectation()]}) + "\n\n", encoding="utf-8")
        details = tmp_path / "details.txt"
        assert run("compare", file, second, "--details", details) == 0
        assert run("compare", file, tmp_path / "no-such-file.jsonl", "--details", details) == 1
        assert capsys.readouterr().out.splitlines()[-1] == (
            f"the second labelling cannot be read; the details are in {details}"
        )
        assert "cannot read" in details.read_text(encoding="utf-8")

    def test_details_that_cannot_be_written_are_an_error_not_a_traceback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        taken = tmp_path / "taken"
        taken.write_text("a file where a folder is needed", encoding="utf-8")
        assert run("check", write(tmp_path, candidate()), "--details", taken / "details.txt") == 1
        captured = capsys.readouterr()
        assert "cannot be written" in captured.err
        assert TASK_ID not in captured.err + captured.out

    def test_live_asks_the_configured_catalogue_unless_given_another(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asked = FakeCatalog()
        configured: list[Settings] = []

        def the_configured_one(settings: Settings) -> FakeCatalog:
            configured.append(settings)
            return asked

        without_env_file = Settings(_env_file=None)  # type: ignore[call-arg]  # a pydantic-settings init option
        monkeypatch.setattr(candidates, "Settings", lambda: without_env_file)
        monkeypatch.setattr(candidates, "live_catalog", the_configured_one)
        arguments = ["live", str(write(tmp_path, candidate())), "--details", str(tmp_path / "details.txt")]
        assert candidates.main(arguments, pause_s=0) == 0
        assert len(configured) == len(asked.requests) == 1

    def test_a_golden_set_that_cannot_be_loaded_is_said_without_its_text(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(gazetteer: Gazetteer) -> list[Task]:
            raise GoldenError("test.jsonl:3: a secret question about Rome")

        monkeypatch.setattr(candidates, "load_golden", broken)
        assert run("check", write(tmp_path, candidate()), "--details", tmp_path / "details.txt") == 1
        captured = capsys.readouterr()
        assert "the golden set or the gazetteer cannot be loaded" in captured.err
        assert "secret" not in captured.err + captured.out
