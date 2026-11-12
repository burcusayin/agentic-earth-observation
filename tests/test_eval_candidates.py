"""Tests for the checks on a file of candidate tasks (evals/candidates.py)."""

from __future__ import annotations

import json
import logging
import time
import warnings
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from agentic_earth_observation.catalog.gazetteer import Gazetteer, GazetteerError
from agentic_earth_observation.catalog.stac import CatalogError, SceneSummary
from agentic_earth_observation.config import Settings
from agentic_earth_observation.domain import Product, SearchRequest
from agentic_earth_observation.evals import candidates
from agentic_earth_observation.evals.candidates import (
    SecondLabel,
    differences,
    live_check,
    period_dates,
    places_named,
    problems,
    question_key,
    questions,
)
from agentic_earth_observation.evals.golden import (
    GOLDEN_DIR,
    REFERENCE_DATE,
    ExpectedSearch,
    GoldenError,
    Task,
    load_tasks,
)
from agentic_earth_observation.evals.groups import PERIODS
from agentic_earth_observation.evals.run_eval import ORACLE_LIMIT, oracle_request
from tests.fakes import FakeCatalog

GAZETTEER = Gazetteer.load()
# The golden set as these tests know it: the dev split. The ids below end in a way no task's would, so that
# a test task added later cannot have one of them and be named by a failing test.
DEV = load_tasks(GOLDEN_DIR / "dev.jsonl")
ROME = [12.2344, 41.6556, 12.8558, 42.141]
QUESTION = "Find cloud-free Sentinel-2 images of Rome from July 2025."
FIRST_OF_TWO = "Sentinel-2 images of Rome from July 2025 without clouds, please."
TASK_ID = "v2-single-example-for-the-tests"
NOTE = "Live check 2026-11-12: the expected search returns 7 scenes."
SEARCH = {
    "product": "sentinel-2-l2a",
    "place": "rome",
    "start": "2025-07-01",
    "end": "2025-07-31",
    "cloud": [0, 10],
}
# Twelve hexadecimal digits, as a key of the questions file has. Written apart from the word it is used
# under: next to it, the scan for secrets takes it for an API key.
HEX_12 = "0123456789ab"
ASKED_TWICE = "one of its questions is asked a second time, by it or by another task"
SECRETS = (TASK_ID, "for-the-tests", "Rome", "rome", "2025-07", "secret")  # what no command may print


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


def asking(question: str, **overrides: object) -> dict[str, Any]:
    """The valid single-search task with other words, and any top-level field replaced."""
    return candidate(turns=[{"user": question, "expect": expectation()}], **overrides)


def declined(**overrides: object) -> dict[str, Any]:
    """A valid new task that expects no search."""
    expect = {"route": "decline", "forbidden_tools": ["search_scenes", "get_scene"], "outcomes": ["declined"]}
    fields: dict[str, Any] = {
        "id": "v2-oos-example-for-the-tests",
        "category": "out_of_scope",
        "turns": [{"user": "What is the weather in Rome?", "expect": expect}],
        "notes": "",
    }
    fields.update(overrides)
    return candidate(**fields)


def follow_up(**overrides: object) -> dict[str, Any]:
    """A valid new task of two turns, each with its search."""
    second = expectation(required_tools=["search_scenes"], search={**SEARCH, "end": "2025-07-15", "limit": 3})
    fields: dict[str, Any] = {
        "id": "v2-follow-example-for-the-tests",
        "category": "follow_up",
        "turns": [{"user": FIRST_OF_TWO, "expect": expectation()}, {"user": "Only three.", "expect": second}],
        "notes": f"{NOTE} Turn 2: 3.",
    }
    fields.update(overrides)
    return candidate(**fields)


def relative(
    period: str = "last-month", start: str = "2025-11-01", end: str = "2025-11-30", **overrides: object
) -> dict[str, Any]:
    """A valid new relative-date task: its search has the dates its period means on the reference date."""
    expect = expectation(search={**SEARCH, "start": start, "end": end})
    fields: dict[str, Any] = {
        "id": f"v2-relative-{period}-for-the-tests",
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


class TestPlacesNamed:
    @pytest.mark.parametrize(
        "text",
        [
            "Radar scenes of Milan, please.",
            "Radar scenes (Milan)?",
            "Show me Milan's surroundings from space.",
            "Show me Milan\u2019s surroundings from space.",
            "Bilder von Mailand im Oktober",
            "scenes of MILANO",
        ],
    )
    def test_a_name_the_gazetteer_resolves_is_found_whatever_stands_around_it(self, text: str) -> None:
        assert places_named(text, GAZETTEER) == {"milan"}

    def test_every_name_the_gazetteer_has_is_found_however_many_words_it_has(self) -> None:
        names = [
            (place.key, name)
            for place in map(GAZETTEER.resolve, GAZETTEER.known_places())
            if place is not None
            for name in GAZETTEER.names_of(place.key)
        ]
        assert max(len(name.split()) for _, name in names) > 1  # or this test shows nothing about long names
        missed = [
            key for key, name in names if key not in places_named(f"Radar over {name}, please.", GAZETTEER)
        ]
        assert not missed

    def test_every_place_a_text_names_is_found(self) -> None:
        assert places_named("From Milan to Naples by train.", GAZETTEER) == {"milan", "naples"}

    @pytest.mark.parametrize("text", ["Radar scenes of Milaan.", "Scenes of the Milanese plain.", "", "?!"])
    def test_a_name_the_gazetteer_does_not_resolve_is_not_a_place(self, text: str) -> None:
        # A misspelt name is unknown by the labelling rules, however obvious it looks.
        assert places_named(text, GAZETTEER) == set()


class TestProblems:
    def test_valid_candidates_have_none(self) -> None:
        assert problems(tasks(candidate(), declined(), follow_up(), relative()), DEV, GAZETTEER) == []

    @pytest.mark.parametrize(
        ("changes", "problem"),
        [
            ({"id": "single-example-for-the-tests"}, "the id of a new task starts with v2-"),
            ({"split": "dev"}, "a candidate is a test task, not a dev one"),
            ({"id": "v2-relative-next-week-for-the-tests"}, "names none of the periods"),
        ],
    )
    def test_a_candidate_that_breaks_a_rule_is_named_with_the_rule(
        self, changes: dict[str, object], problem: str
    ) -> None:
        (found,) = problems(tasks(candidate(**changes)), DEV, GAZETTEER)
        assert problem in found
        assert str(changes.get("id", TASK_ID)) in found

    @pytest.mark.parametrize(
        "notes",
        [
            "",
            "The expected search returns 7 scenes.",
            "Live check: the expected search returns 7 scenes.",  # no date
            "Live check 2026-11-12",  # no count
            "Live check 2026-11-12: 7 scenes.",  # not the form the golden set uses
            "Live check 2026-11-12: the expected search returns 0 scenes.",  # a new search must find some
        ],
    )
    def test_notes_that_do_not_say_what_the_live_check_found_are_refused(self, notes: str) -> None:
        (found,) = problems(tasks(candidate(notes=notes)), DEV, GAZETTEER)
        assert found.startswith(f"{TASK_ID}: its notes do not say what the live check found")

    def test_a_live_check_on_a_day_that_does_not_exist_is_refused(self) -> None:
        notes = "Live check 9999-99-99: the expected search returns 7 scenes."
        (found,) = problems(tasks(candidate(notes=notes)), DEV, GAZETTEER)
        assert found == f"{TASK_ID}: the date of the live check in its notes is not a date"

    @pytest.mark.parametrize(
        "notes",
        [
            f"Cloud-free means a cloud limit of at most 10. {NOTE}",  # a reason first, as the dev split has
            "Live check 2026-11-12: the expected search returns 1 scene.",
        ],
    )
    def test_notes_may_say_more_and_count_a_single_scene(self, notes: str) -> None:
        assert problems(tasks(candidate(notes=notes)), DEV, GAZETTEER) == []

    @pytest.mark.parametrize(
        "notes", [NOTE, f"{NOTE} Turn 2: 0.", f"Turn 2: 3. {NOTE}", f"{NOTE} Turn 3: 3."]
    )
    def test_every_expected_search_needs_its_count_in_the_notes(self, notes: str) -> None:
        (found,) = problems(tasks(follow_up(notes=notes)), DEV, GAZETTEER)
        assert found == (
            "v2-follow-example-for-the-tests: its notes give no count for turn 2, "
            'as "Turn 2: N." with N at least 1'
        )

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
        assert problems(tasks(relative(period, start, end)), DEV, GAZETTEER) == []

    def test_a_relative_date_task_of_another_category_is_refused(self) -> None:
        (found,) = problems(tasks(relative(category="single_search")), DEV, GAZETTEER)
        assert found == (
            "v2-relative-last-month-for-the-tests: a relative-date task is a robustness task, "
            "not a single_search one"
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
        (found,) = problems(tasks(relative("last-summer", start, end)), DEV, GAZETTEER)
        assert found == (
            "v2-relative-last-summer-for-the-tests: last-summer is 2025-06-01 to 2025-08-31 on 2025-12-15, "
            f"but its search expects {start} to {end}"
        )

    def test_a_relative_date_task_that_expects_no_search_is_refused(self) -> None:
        asks = {"route": "clarify", "forbidden_tools": ["search_scenes"], "outcomes": ["clarified"]}
        turns = [{"user": "Images of Rome from last summer.", "expect": asks}]
        (found,) = problems(tasks(relative("last-summer", turns=turns, notes="")), DEV, GAZETTEER)
        assert found == "v2-relative-last-summer-for-the-tests: a relative-date task expects a search"

    @pytest.mark.parametrize(
        "question",
        [
            "Sentinel-2 images of Rome from November 2025.",  # the period is not named
            "Sentinel-2 images of Rome from last autumn.",  # another period is
            "Sentinel-2 images of Rome from last month, or from last autumn.",  # two are
        ],
    )
    def test_a_relative_date_task_names_its_period_in_the_rules_words_and_no_other(
        self, question: str
    ) -> None:
        turns = [
            {
                "user": question,
                "expect": expectation(search={**SEARCH, "start": "2025-11-01", "end": "2025-11-30"}),
            }
        ]
        (found,) = problems(tasks(relative(turns=turns)), DEV, GAZETTEER)
        assert found == (
            'v2-relative-last-month-for-the-tests: its questions name "last month", '
            "the rules' words for its period, and no other period"
        )

    def test_the_period_may_be_written_in_any_case_and_spacing(self) -> None:
        turns = [
            {
                "user": "Rome, LAST   Month, Sentinel-2.",
                "expect": expectation(search={**SEARCH, "start": "2025-11-01", "end": "2025-11-30"}),
            }
        ]
        assert problems(tasks(relative(turns=turns)), DEV, GAZETTEER) == []

    @pytest.mark.parametrize("words", ["last month", "Last Summer", "last  winter"])
    def test_a_task_that_names_a_relative_period_has_a_relative_date_id(self, words: str) -> None:
        # Otherwise its dates would not be checked, and a report would not count it with its period.
        (found,) = problems(
            tasks(asking(f"Sentinel-2 images of Rome from {words}, no clouds.")), DEV, GAZETTEER
        )
        assert found == (
            f"{TASK_ID}: a question names a relative period, so its id starts with v2-relative-<period>"
        )

    def test_a_relative_period_in_a_later_question_counts_too(self) -> None:
        turns = [follow_up()["turns"][0], {**follow_up()["turns"][1], "user": "And last month?"}]
        (found,) = problems(tasks(follow_up(turns=turns)), DEV, GAZETTEER)
        assert found == (
            "v2-follow-example-for-the-tests: a question names a relative period, "
            "so its id starts with v2-relative-<period>"
        )

    def test_a_place_of_the_dev_split_is_refused(self) -> None:
        # Prompts are tuned on the dev split, so a test task must not share a place with it.
        milan = expectation(search={**SEARCH, "place": "milan"})
        turns = [{"user": "Find cloud-free Sentinel-2 images of Milan from July 2025.", "expect": milan}]
        (found,) = problems(tasks(candidate(turns=turns)), DEV, GAZETTEER)
        assert found == f"{TASK_ID} turn 1: milan is a place of the dev split"

    def test_a_place_of_the_dev_split_is_refused_in_a_later_turn_too(self) -> None:
        first = {"user": QUESTION, "expect": expectation()}
        naples = expectation(
            required_tools=["resolve_place", "search_scenes"], search={**SEARCH, "place": "naples"}
        )
        (found,) = problems(
            tasks(follow_up(turns=[first, {"user": "And Naples?", "expect": naples}])), DEV, GAZETTEER
        )
        assert found == "v2-follow-example-for-the-tests turn 2: naples is a place of the dev split"

    @pytest.mark.parametrize(
        "question",
        [
            "Find cloud-free Sentinel-2 images of Rmoe from July 2025.",  # not a name the gazetteer has
            "Find cloud-free Sentinel-2 images of Milan from July 2025.",  # another place
            "Find cloud-free Sentinel-2 images from July 2025.",
        ],
    )
    def test_a_search_whose_place_no_question_names_is_refused(self, question: str) -> None:
        # By the labelling rules a name the gazetteer does not resolve is asked about, not searched.
        (found,) = problems(tasks(asking(question)), DEV, GAZETTEER)
        assert found == f"{TASK_ID} turn 1: no question up to it names rome in a way the gazetteer resolves"

    def test_a_later_turn_may_leave_the_place_to_an_earlier_one(self) -> None:
        assert problems(tasks(follow_up()), DEV, GAZETTEER) == []

    def test_a_question_an_existing_task_asks_is_refused(self) -> None:
        asked_already = DEV[3].turns[0].user.upper() + "  "  # the same words, whatever the case and spacing
        again = declined(turns=[{"user": asked_already, "expect": declined()["turns"][0]["expect"]}])
        (found,) = problems(tasks(again), DEV, GAZETTEER)
        assert found == (f"v2-oos-example-for-the-tests: {ASKED_TWICE}")

    def test_a_question_two_candidates_ask_is_refused_for_both(self) -> None:
        found = problems(tasks(candidate(), candidate(id="v2-single-again-for-the-tests")), DEV, GAZETTEER)
        assert found == [
            f"{TASK_ID}: {ASKED_TWICE}",
            f"v2-single-again-for-the-tests: {ASKED_TWICE}",
        ]

    def test_an_id_that_a_task_already_has_is_refused(self) -> None:
        existing = tasks(asking("Sentinel-2 over Rome in July 2025, without clouds."))
        (found,) = problems(tasks(candidate()), existing, GAZETTEER)
        assert found == f"{TASK_ID}: the id is already a task's"

    def test_the_golden_sets_own_checks_apply(self) -> None:
        unknown = expectation(search={**SEARCH, "place": "atlantis"})
        twice = tasks(
            candidate(), candidate(turns=[{"user": "Scenes of Atlantis in July 2025?", "expect": unknown}])
        )
        found = problems(twice, [], GAZETTEER)
        assert f"{TASK_ID}: the id is used 2 times" in found
        assert f"{TASK_ID} turn 1: 'atlantis' is not a gazetteer key" in found

    def test_a_task_without_a_search_needs_no_note(self) -> None:
        assert problems(tasks(declined()), DEV, GAZETTEER) == []


class FailsFrom(FakeCatalog):
    """A catalogue that answers until its `at`-th search, which fails as `error` says."""

    def __init__(self, at: int, error: BaseException) -> None:
        super().__init__()
        self.at = at
        self.error = error

    def search(self, request: SearchRequest) -> list[SceneSummary]:
        self.requests.append(request)
        if len(self.requests) >= self.at:
            raise self.error
        return self.scenes[: request.limit]


class TestLiveCheck:
    def test_every_expected_search_is_asked_as_the_oracle_asks_it(self) -> None:
        catalog = FakeCatalog()
        pauses: list[None] = []
        results = list(
            live_check(
                tasks(candidate(), declined(), follow_up()),
                catalog,
                GAZETTEER,
                pause=lambda: pauses.append(None),
            )
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

    def test_it_stops_at_the_first_search_that_fails(self) -> None:
        # A search of a valid task fails when the catalogue does not answer: nothing more is asked of it.
        catalog = FailsFrom(2, CatalogError("scene search failed: the catalogue did not answer"))
        pauses: list[None] = []
        results = list(
            live_check(tasks(candidate(), follow_up()), catalog, GAZETTEER, pause=lambda: pauses.append(None))
        )
        assert [(r.task_id, r.turn, r.scenes) for r in results] == [
            (TASK_ID, 1, 1),
            ("v2-follow-example-for-the-tests", 1, None),
        ]
        assert results[-1].failed == "scene search failed: the catalogue did not answer"
        assert (len(catalog.requests), len(pauses)) == (2, 1)


class TestQuestions:
    def test_they_are_the_users_words_under_a_key_and_nothing_else(self) -> None:
        both = tasks(candidate(), follow_up())
        asked = questions(both)
        assert sorted(asked, key=lambda entry: len(entry["turns"])) == [
            {"key": question_key(both[0]), "turns": [QUESTION]},
            {"key": question_key(both[1]), "turns": [FIRST_OF_TWO, "Only three."]},
        ]
        # The id of a relative-date task names its period, which is the answer the second labelling looks for.
        assert "for-the-tests" not in json.dumps(asked)

    def test_they_are_in_the_order_of_their_keys_not_of_the_file(self) -> None:
        # The file follows the categories, so a task's place in it says what kind of task it is.
        many = tasks(*(asking(f"Sentinel-2 images of Rome, {day} July 2025.") for day in range(1, 13)))
        keys = [entry["key"] for entry in questions(many)]
        assert keys == sorted(keys) != [question_key(task) for task in many]

    def test_a_key_is_made_of_the_words_alone(self) -> None:
        one, same_words, other_words, two_turns = tasks(
            candidate(),
            candidate(id="v2-single-again-for-the-tests", category="robustness", notes=f"Another. {NOTE}"),
            asking(QUESTION.replace("July", "June")),
            follow_up(),
        )
        other_end = [follow_up()["turns"][0], {**follow_up()["turns"][1], "user": "Only two."}]
        assert question_key(one) == question_key(same_words)
        assert len({question_key(one), question_key(other_words), question_key(two_turns)}) == 3
        assert question_key(two_turns) != question_key(Task.model_validate(follow_up(turns=other_end)))
        assert len(question_key(one)) == 12
        assert set(question_key(one)) <= set("0123456789abcdef")


def label(of: dict[str, Any], *turns: dict[str, Any]) -> SecondLabel:
    """What the second labelling says of the task `of`: the key of its questions and each turn's answer."""
    return SecondLabel.model_validate({"key": question_key(Task.model_validate(of)), "turns": list(turns)})


class TestDifferences:
    def test_the_same_labels_do_not_differ_in_whatever_order_they_come(self) -> None:
        both = tasks(candidate(), declined())
        second = [
            label(
                declined(),
                {
                    "route": "decline",
                    "forbidden_tools": ["get_scene", "search_scenes"],
                    "outcomes": ["declined"],
                },
            ),
            label(candidate(), expectation(route=["delegate"])),  # a list of one route is that route
        ]
        assert differences(both, second) == []

    def test_several_accepted_routes_and_outcomes_are_the_same_in_any_order(self) -> None:
        # An unknown place may be checked with the retrieval agent first, or asked about at once.
        asks = {"required_tools": ["resolve_place"], "forbidden_tools": ["search_scenes"]}
        first = {"route": ["delegate", "clarify"], "outcomes": ["clarified", "declined"], **asks}
        second = {"route": ["clarify", "delegate"], "outcomes": ["declined", "clarified"], **asks}
        unknown = candidate(category="clarify", turns=[{"user": "Scenes of Atlantis?", "expect": first}])
        assert differences(tasks(unknown), [label(unknown, second)]) == []

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
        (found,) = differences(tasks(candidate()), [label(candidate(), expectation(**changes))])
        assert (found.number, found.text.split(":")[0]) == (1, f"task 1 ({TASK_ID}) turn 1, {field}")

    def test_a_task_the_second_labelling_left_out_differs(self) -> None:
        (found,) = differences(tasks(candidate(), declined()), [label(candidate(), expectation())])
        assert (found.number, found.text) == (
            2,
            "task 2 (v2-oos-example-for-the-tests): the second labelling has no entry for its questions",
        )

    def test_another_number_of_turns_differs(self) -> None:
        (found,) = differences(tasks(follow_up()), [label(follow_up(), expectation())])
        assert "1 turn(s) where the task has 2" in found.text

    def test_a_label_made_for_other_words_is_of_no_task(self) -> None:
        # The labels answer the questions as they were when they were exported. A task reworded since then
        # has no label, although one stands where its label stood.
        before, after = candidate(), asking(QUESTION.replace("July", "June"))
        found = differences(tasks(after), [label(before, expectation())])
        assert [(d.number, d.text) for d in found] == [
            (1, f"task 1 ({TASK_ID}): the second labelling has no entry for its questions"),
            (
                None,
                f"entry {question_key(Task.model_validate(before))} of the second labelling "
                "answers the questions of no task",
            ),
        ]

    def test_an_entry_given_twice_is_named(self) -> None:
        second = [label(candidate(), expectation()), label(candidate(), expectation())]
        (found,) = differences(tasks(candidate()), second)
        assert (found.number, found.text) == (
            1,
            f"task 1 ({TASK_ID}): the second labelling has 2 entries for it",
        )

    def test_two_tasks_with_the_same_words_cannot_be_told_apart(self) -> None:
        twins = tasks(candidate(), candidate(id="v2-single-again-for-the-tests"))
        found = differences(twins, [label(candidate(), expectation())])
        assert [d.number for d in found] == [1, 2]
        assert all("another task asks the same words" in d.text for d in found)


def run(*arguments: str | Path, catalog: FakeCatalog | None = None) -> int:
    return candidates.main(
        [str(argument) for argument in arguments], catalog=catalog or FakeCatalog(), pause_s=0
    )


def printed(capsys: pytest.CaptureFixture[str]) -> str:
    """What a command printed, once it is certain that nothing went to the error stream and that neither
    stream says anything of a task."""
    captured = capsys.readouterr()
    assert captured.err == ""
    assert not [secret for secret in SECRETS if secret in captured.out]
    return captured.out


def second_labelling(folder: Path, *entries: SecondLabel | dict[str, Any]) -> Path:
    path = folder / "second.jsonl"
    lines = [
        entry.model_dump_json() if isinstance(entry, SecondLabel) else json.dumps(entry) for entry in entries
    ]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


class TestTheCommands:
    """Every command prints totals only, on both streams: whoever writes prompts runs it and must not see a
    task."""

    def test_check_passes_valid_candidates_and_counts_them(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        again = relative(
            id="v2-relative-last-month-again-for-the-tests",
            turns=[
                {
                    "user": "Rome last month, Sentinel-2, please.",
                    "expect": expectation(search={**SEARCH, "start": "2025-11-01", "end": "2025-11-30"}),
                }
            ],
        )
        file = write(tmp_path, candidate(), relative(), again, declined())
        details = tmp_path / "out" / "details.txt"
        assert run("check", file, "--details", details) == 0
        assert printed(capsys).splitlines() == [
            "tasks: 4 (out_of_scope 1, robustness 2, single_search 1)",
            "relative dates: 2 (last-month 2)",  # tasks, not periods
            "problems: 0",
        ]
        assert details.read_text(encoding="utf-8") == "no problems\n"

    def test_check_counts_the_problems_and_keeps_them_out_of_what_it_prints(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        other = asking("Sentinel-2 over Rome in June 2025.", id="v2-relative-for-the-tests")
        file = write(tmp_path, candidate(notes=""), other)
        details = tmp_path / "details.txt"
        assert run("check", file, "--details", details) == 1
        assert printed(capsys).splitlines() == [
            "tasks: 2 (single_search 2)",
            "relative dates: cannot be counted, an id names no period",
            f"problems: 2; the details are in {details}",
        ]
        written = details.read_text(encoding="utf-8")
        assert f"{TASK_ID}: its notes do not say" in written
        assert "v2-relative-for-the-tests starts with v2-relative- but names none of the periods" in written

    def test_check_knows_the_golden_set(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        # A place of the dev split is one only by the golden set: the command has to load it.
        milan = expectation(search={**SEARCH, "place": "milan"})
        turns = [{"user": "Find cloud-free Sentinel-2 images of Milan from July 2025.", "expect": milan}]
        details = tmp_path / "details.txt"
        assert run("check", write(tmp_path, candidate(turns=turns)), "--details", details) == 1
        assert f"problems: 1; the details are in {details}" in printed(capsys)
        assert "milan is a place of the dev split" in details.read_text(encoding="utf-8")

    def test_a_file_that_cannot_be_read_as_tasks_is_said_without_its_text(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        file = tmp_path / "candidates.jsonl"
        file.write_text('{"id": "v2-broken", "user": "A secret question about Rome"}\n', encoding="utf-8")
        details = tmp_path / "details.txt"
        for command in (("check", file), ("live", file), ("compare", file, tmp_path / "second.jsonl")):
            assert run(*command, "--details", details) == 1
            assert printed(capsys) == f"the file cannot be read as tasks; the details are in {details}\n"
            assert "candidates.jsonl:1" in details.read_text(encoding="utf-8")
        assert run("questions", file, "--out", tmp_path / "questions.jsonl") == 1
        assert printed(capsys) == "the file cannot be read as tasks; `check` writes the details\n"
        assert not (tmp_path / "questions.jsonl").exists()

    def test_a_file_that_is_not_text_is_said_too(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        file = tmp_path / "candidates.jsonl"
        file.write_bytes(b'{"id": "v2-secret-\xff"}\n')
        details = tmp_path / "details.txt"
        assert run("check", file, "--details", details) == 1
        assert printed(capsys) == f"the file cannot be read as tasks; the details are in {details}\n"
        assert details.read_text(encoding="utf-8") == "candidates.jsonl is not UTF-8 text\n"
        second = tmp_path / "second.jsonl"
        second.write_bytes(b'{"key": "\xff"}\n')
        assert (
            run("compare", write(tmp_path, candidate(), name="good.jsonl"), second, "--details", details) == 1
        )
        assert printed(capsys) == f"the second labelling cannot be read; the details are in {details}\n"

    @pytest.mark.parametrize("text", ["", "\n\n"])
    def test_a_file_without_tasks_passes_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], text: str
    ) -> None:
        file = tmp_path / "candidates.jsonl"
        file.write_text(text, encoding="utf-8")
        details = tmp_path / "details.txt"
        catalog = FakeCatalog()
        for command in (("check", file), ("live", file), ("compare", file, tmp_path / "second.jsonl")):
            assert run(*command, "--details", details, catalog=catalog) == 1
            assert printed(capsys) == "the file has no tasks\n"
        assert run("questions", file, "--out", tmp_path / "questions.jsonl") == 1
        assert printed(capsys) == "the file has no tasks\n"
        assert catalog.requests == []
        assert not (tmp_path / "questions.jsonl").exists()

    def test_a_command_does_not_write_over_a_file_it_reads(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        file = write(tmp_path, candidate())
        (tmp_path / "other").mkdir()
        before = file.read_text(encoding="utf-8")
        second = second_labelling(tmp_path, label(candidate(), expectation()))
        labels = second.read_text(encoding="utf-8")
        for command in (
            ("check", file, "--details", file),
            ("live", file, "--details", tmp_path / "other" / ".." / "candidates.jsonl"),  # another spelling
            ("questions", file, "--out", file),
            ("compare", file, second, "--details", second),
            ("compare", file, second, "--details", file),
        ):
            assert run(*command) == 1
            captured = capsys.readouterr()
            assert "is a file the command reads" in captured.err
            assert captured.out == ""
        assert (file.read_text(encoding="utf-8"), second.read_text(encoding="utf-8")) == (before, labels)

    def test_live_counts_what_the_catalogue_found(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        file = write(tmp_path, candidate(), follow_up(), declined())
        details = tmp_path / "out" / "details.txt"
        assert run("live", file, "--details", details) == 0
        assert printed(capsys) == (
            "searches: 3; asked: 3; found scenes: 3; found none: 0; failed: 0; "
            f"the details are in {details}\n"
        )
        assert details.read_text(encoding="utf-8").splitlines() == [
            f"{TASK_ID} turn 1: 1 scene(s)",
            "v2-follow-example-for-the-tests turn 1: 1 scene(s)",
            "v2-follow-example-for-the-tests turn 2: 1 scene(s)",
        ]

    def test_live_fails_when_a_search_finds_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        details = tmp_path / "details.txt"
        file = write(tmp_path, candidate(), follow_up())
        assert run("live", file, "--details", details, catalog=FakeCatalog([])) == 1
        assert "searches: 3; asked: 3; found scenes: 0; found none: 3; failed: 0" in printed(capsys)
        assert details.read_text(encoding="utf-8").count("0 scene(s)") == 3

    def test_live_stops_and_fails_when_the_catalogue_fails(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        details = tmp_path / "details.txt"
        catalog = FakeCatalog(fail=True)
        assert (
            run("live", write(tmp_path, candidate(), follow_up()), "--details", details, catalog=catalog) == 1
        )
        assert "searches: 3; asked: 1; found scenes: 0; found none: 0; failed: 1" in printed(capsys)
        assert details.read_text(encoding="utf-8") == (
            f"{TASK_ID} turn 1: failed: scene search failed: the catalogue did not answer (Read timed out)\n"
        )
        assert len(catalog.requests) == 1

    def test_live_keeps_what_it_learnt_when_it_is_stopped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        details = tmp_path / "details.txt"
        catalog = FailsFrom(2, KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt):
            run("live", write(tmp_path, candidate(), follow_up()), "--details", details, catalog=catalog)
        assert details.read_text(encoding="utf-8") == f"{TASK_ID} turn 1: 1 scene(s)\n"

    def test_live_asks_nothing_when_it_cannot_write_what_it_finds(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        catalog = FakeCatalog()
        assert run("live", write(tmp_path, candidate()), "--details", tmp_path, catalog=catalog) == 1
        captured = capsys.readouterr()
        assert "cannot be written" in captured.err
        assert captured.out == ""
        assert catalog.requests == []

    def test_live_pauses_a_second_between_two_searches(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        slept: list[float] = []
        monkeypatch.setattr(time, "sleep", slept.append)
        arguments = [
            "live",
            str(write(tmp_path, candidate(), follow_up())),
            "--details",
            str(tmp_path / "d.txt"),
        ]
        assert candidates.main(arguments, catalog=FakeCatalog()) == 0
        assert slept == [1.0, 1.0]  # three searches of a public service

    def test_live_asks_nothing_of_the_catalogue_for_candidates_the_golden_set_refuses(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        unknown = expectation(search={**SEARCH, "place": "atlantis"})
        file = write(tmp_path, candidate(turns=[{"user": QUESTION, "expect": unknown}]))
        catalog = FakeCatalog()
        details = tmp_path / "details.txt"
        assert run("live", file, "--details", details, catalog=catalog) == 1
        assert printed(capsys) == f"problems: 1; nothing was searched; the details are in {details}\n"
        assert catalog.requests == []
        assert "'atlantis' is not a gazetteer key" in details.read_text(encoding="utf-8")

    def test_live_asks_the_configured_catalogue_without_retries_unless_given_another(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asked = FakeCatalog()
        configured: list[Settings] = []

        def the_configured_one(settings: Settings) -> FakeCatalog:
            configured.append(settings)
            return asked

        without_env_file = Settings(_env_file=None, stac_timeout_s=7)  # type: ignore[call-arg]  # a pydantic-settings init option
        assert without_env_file.stac_max_retries > 0
        monkeypatch.setattr(candidates, "Settings", lambda: without_env_file)
        monkeypatch.setattr(candidates, "live_catalog", the_configured_one)
        arguments = ["live", str(write(tmp_path, candidate())), "--details", str(tmp_path / "details.txt")]
        assert candidates.main(arguments, pause_s=0) == 0
        assert len(configured) == len(asked.requests) == 1
        # The client does not ask a failed search again either: as many requests as searches.
        assert (configured[0].stac_max_retries, configured[0].stac_timeout_s) == (0, 7)

    def test_nothing_a_catalogue_client_logs_or_warns_reaches_the_screen(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        caplog: pytest.LogCaptureFixture,
        recwarn: pytest.WarningsRecorder,
    ) -> None:
        log = logging.getLogger("agentic_earth_observation.catalog.stac")

        class Talkative(FakeCatalog):
            def search(self, request: SearchRequest) -> list[SceneSummary]:
                # As the real client does: it logs what the catalogue answered, which can quote the search.
                log.error("scene search failed: a secret box and secret dates")
                warnings.warn("a secret", stacklevel=1)
                return super().search(request)

        caplog.set_level(logging.DEBUG)
        details = tmp_path / "details.txt"
        catalog = Talkative(fail=True)
        assert run("live", write(tmp_path, candidate()), "--details", details, catalog=catalog) == 1
        assert "failed: 1" in printed(capsys)
        assert (caplog.records, len(recwarn)) == ([], 0)
        log.error("after the command, logging is as it was")
        assert [record.getMessage() for record in caplog.records] == [
            "after the command, logging is as it was"
        ]

    def test_questions_writes_the_questions_alone(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "second" / "questions.jsonl"
        both = tasks(candidate(), follow_up())
        assert run("questions", write(tmp_path, candidate(), follow_up()), "--out", out) == 0
        assert printed(capsys) == f"tasks: 2; the questions are in {out}\n"
        lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        assert lines == questions(both)
        assert {(entry["key"], tuple(entry["turns"])) for entry in lines} == {
            (question_key(both[0]), (QUESTION,)),
            (question_key(both[1]), (FIRST_OF_TWO, "Only three.")),
        }

    def test_a_question_is_one_line_whatever_it_contains(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # U+2028 and U+0085 may stand raw in a JSON string, and Python splits lines at them.
        odd = "Sentinel-2 over Rome,\u2028July\u0085 2025, ohne Wölkchen."
        out = tmp_path / "questions.jsonl"
        assert run("questions", write(tmp_path, asking(odd)), "--out", out) == 0
        (line,) = out.read_text(encoding="utf-8").splitlines()
        assert json.loads(line)["turns"] == [odd]
        assert "Wölkchen" in line  # other letters stay readable

    def test_compare_counts_the_tasks_that_agree_and_differ(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        file = write(tmp_path, candidate(), declined(), follow_up())
        other = expectation(search={**SEARCH, "start": "2025-07-02", "cloud": [0, 20]}, outcomes=["ok"])
        second = second_labelling(
            tmp_path,
            label(candidate(), other),
            label(follow_up(), *(turn["expect"] for turn in follow_up()["turns"])),
        )
        details = tmp_path / "details.txt"
        assert run("compare", file, second, "--details", details) == 1
        # Two fields of one task differ and another task has no label: two tasks differ, not three.
        assert printed(capsys) == f"tasks compared: 3; agree: 1; differ: 2; the details are in {details}\n"
        written = details.read_text(encoding="utf-8").splitlines()
        assert written[0].startswith(f"task 1 ({TASK_ID}) turn 1, search: ")
        assert written[1].startswith(f"task 1 ({TASK_ID}) turn 1, outcomes: ")
        assert written[2] == (
            "task 2 (v2-oos-example-for-the-tests): the second labelling has no entry for its questions"
        )

    def test_compare_passes_labels_that_agree(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        second = second_labelling(tmp_path, label(candidate(), expectation()))
        details = tmp_path / "details.txt"
        assert run("compare", write(tmp_path, candidate()), second, "--details", details) == 0
        assert printed(capsys) == "tasks compared: 1; agree: 1; differ: 0\n"
        assert details.read_text(encoding="utf-8") == "no differences\n"

    def test_compare_fails_on_an_entry_without_a_task_and_counts_it_apart(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        second = second_labelling(
            tmp_path, label(candidate(), expectation()), label(follow_up(), expectation())
        )
        details = tmp_path / "details.txt"
        assert run("compare", write(tmp_path, candidate()), second, "--details", details) == 1
        assert printed(capsys) == (
            "tasks compared: 1; agree: 1; differ: 0; entries without a task: 1; "
            f"the details are in {details}\n"
        )
        key = question_key(Task.model_validate(follow_up()))
        assert details.read_text(encoding="utf-8") == (
            f"entry {key} of the second labelling answers the questions of no task\n"
        )

    @pytest.mark.parametrize(
        "entry",
        [
            {"key": HEX_12, "turns": "a secret answer"},
            {"key": HEX_12, "turns": []},
            {"turns": [{"route": "decline", "forbidden_tools": ["search_scenes"]}]},
            {"key": HEX_12.upper(), "turns": [{"route": "decline", "forbidden_tools": ["search_scenes"]}]},
            {"key": HEX_12[:-1], "turns": [{"route": "decline", "forbidden_tools": ["search_scenes"]}]},
            {"key": 123456789012, "turns": [{"route": "decline", "forbidden_tools": ["search_scenes"]}]},
            {
                "key": HEX_12,
                "n": 1,
                "turns": [{"route": "decline", "forbidden_tools": ["search_scenes"]}],
            },
        ],
    )
    def test_second_labels_that_cannot_be_read_are_said_without_their_text(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], entry: dict[str, Any]
    ) -> None:
        second = second_labelling(tmp_path, entry)
        details = tmp_path / "details.txt"
        assert run("compare", write(tmp_path, candidate()), second, "--details", details) == 1
        assert printed(capsys) == f"the second labelling cannot be read; the details are in {details}\n"
        assert "second.jsonl:1" in details.read_text(encoding="utf-8")

    def test_a_second_labelling_may_have_empty_lines_and_a_missing_one_is_said(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        file = write(tmp_path, candidate())
        second = tmp_path / "second.jsonl"
        second.write_text(
            "\n" + label(candidate(), expectation()).model_dump_json() + "\n\n", encoding="utf-8"
        )
        details = tmp_path / "details.txt"
        assert run("compare", file, second, "--details", details) == 0
        assert run("compare", file, tmp_path / "no-such-file.jsonl", "--details", details) == 1
        assert printed(capsys).splitlines()[-1] == (
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
        assert not [secret for secret in SECRETS if secret in captured.err + captured.out]

    @pytest.mark.parametrize(
        ("what", "error"),
        [
            ("load_golden", GoldenError("test.jsonl:3: a secret question about Rome")),
            ("Gazetteer", GazetteerError("gazetteer.json: a secret")),
        ],
    )
    def test_a_golden_set_or_gazetteer_that_cannot_be_loaded_is_said_without_its_text(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        what: str,
        error: Exception,
    ) -> None:
        class Broken:
            @staticmethod
            def load() -> Gazetteer:
                raise error

        def broken(gazetteer: Gazetteer) -> list[Task]:
            raise error

        monkeypatch.setattr(candidates, what, broken if what == "load_golden" else Broken)
        assert run("check", write(tmp_path, candidate()), "--details", tmp_path / "details.txt") == 1
        captured = capsys.readouterr()
        assert "the golden set or the gazetteer cannot be loaded" in captured.err
        assert not [secret for secret in SECRETS if secret in captured.err + captured.out]
