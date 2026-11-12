"""Tests for the groups of test tasks that are read from their ids."""

from __future__ import annotations

import pytest

from agentic_earth_observation.evals.groups import PERIODS, added_in_v2, relative_period


@pytest.mark.parametrize(
    ("task_id", "added"),
    [
        ("v2-single-s1-rome-may", True),
        ("v2-relative-last-month-rome", True),
        ("s2-florence-july", False),
        ("robust-relative-garda-summer", False),  # the one relative-date task of v1: seen, not new
        ("v2", False),  # the prefix needs its hyphen
        ("dev-v2-milan", False),
    ],
)
def test_a_task_added_in_v2_has_the_prefix(task_id: str, added: bool) -> None:
    assert added_in_v2(task_id) is added


@pytest.mark.parametrize("period", PERIODS)
def test_a_relative_date_task_names_its_period(period: str) -> None:
    assert relative_period(f"v2-relative-{period}-rome-radar") == period
    assert relative_period(f"v2-relative-{period}") == period


def test_the_periods_are_the_five_the_labelling_rules_define() -> None:
    # docs/evaluation.md, "Reference date": last month, and the last spring, summer, autumn and winter.
    assert PERIODS == ("last-month", "last-spring", "last-summer", "last-autumn", "last-winter")


@pytest.mark.parametrize(
    "task_id",
    ["v2-single-s1-rome-may", "v2-follow-rome-next-week", "robust-relative-garda-summer", "s2-florence-july"],
)
def test_any_other_task_has_no_relative_period(task_id: str) -> None:
    assert relative_period(task_id) is None


@pytest.mark.parametrize(
    "task_id",
    [
        "v2-relative-rome",  # no period
        "v2-relative-next-week-rome",  # not one of the five
        "v2-relative-last-monthly-rome",  # a period must end at a hyphen or at the end of the id
        "v2-relative-",
    ],
)
def test_a_relative_date_id_without_one_of_the_periods_is_refused(task_id: str) -> None:
    with pytest.raises(ValueError, match="names none of the periods"):
        relative_period(task_id)
