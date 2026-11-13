"""Tests for the comparison of two setups of one model (evals/setups.py; docs/evaluation.md, "Comparing two
setups"): every number against one worked out by hand."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from agentic_earth_observation.catalog_mcp.tool_log import ToolCallRecord
from agentic_earth_observation.evals.records import AttemptOutcome, AttemptRecord, TurnRecord
from agentic_earth_observation.evals.setups import (
    Actions,
    TurnKey,
    change,
    determinism,
    period_change,
    turn_actions,
)
from agentic_earth_observation.evals.stats import Result
from tests.fakes import SCENE
from tests.test_eval_records import output, turn

P: Result = "passed"
F: Result = "failed"
E: Result = "errored"
L: Result = "lost"


class TestChange:
    def test_each_task_is_taken_by_the_majority_of_its_runs_on_each_side(self) -> None:
        before = {"a": [P, P, F], "b": [F, F, P], "c": [P, P, P], "d": [F, E, F], "e": [P, F, F]}
        after = {"a": [P, P, P], "b": [P, P, F], "c": [F, F, P], "d": [F, F, F], "e": [P, P, F]}
        found = change("All 5 tasks", ["a", "b", "c", "d", "e"], before, after)
        assert (found.scope, found.tasks) == ("All 5 tasks", 5)
        assert (found.both_passed, found.both_failed, found.left_out) == (1, 1, 0)
        assert (found.gained, found.lost) == (["b", "e"], ["c"])  # in the order of the tasks given
        assert found.p_value == 1.0  # 2 against 1: 2 * (1 + 3) / 8, capped at 1

    def test_a_task_without_a_majority_on_either_side_is_left_out(self) -> None:
        # A lost run has no result; an even split has no majority; errored counts as failed.
        before = {"a": [L, L, L], "b": [P, F], "c": [P, E, L], "d": [P, L, L], "e": [P, P, P], "f": [F, F, F]}
        after = {"a": [P, P, P], "b": [P, P], "c": [P, P, P], "d": [F, F, F], "e": [P, F, L], "f": [L, L, L]}
        found = change("All", ["a", "b", "c", "d", "e", "f"], before, after)
        assert (found.left_out, found.gained, found.lost) == (5, [], ["d"])
        assert found.p_value == 1.0

    def test_the_sign_test_is_on_gained_against_lost(self) -> None:
        before = {name: [F, F, F] for name in "abcdef"}
        after = {name: [P, P, P] for name in "abcdef"}
        found = change("All", list("abcdef"), before, after)
        assert (len(found.gained), found.lost) == (6, [])
        assert found.p_value == pytest.approx(0.03125)  # 2 / 2**6

    def test_more_runs_on_one_side_than_the_other_are_fine(self) -> None:
        # A restarted evaluation gives a setup more runs (docs/evaluation.md): 3 failed of 5 is a majority.
        found = change("All", ["a"], {"a": [F, F, F, P, P]}, {"a": [P, P, P]})
        assert (found.gained, found.lost, found.both_passed, found.both_failed, found.left_out) == (
            ["a"],
            [],
            0,
            0,
            0,
        )


def periods_of(before: dict[str, list[Result]], after: dict[str, list[Result]]) -> Any:  # noqa: ANN401
    found = period_change(list(before), before, after)
    assert found is not None
    return found


class TestPeriodChange:
    def test_no_relative_date_task_gives_nothing(self) -> None:
        assert (
            period_change(
                ["a", "v2-single-x"], {"a": [P], "v2-single-x": [P]}, {"a": [P], "v2-single-x": [P]}
            )
            is None
        )

    def test_a_period_passes_when_all_its_tasks_pass(self) -> None:
        before = {
            "v2-relative-last-winter-a": [P, P, P],
            "v2-relative-last-month-a": [P, P, P],
            "v2-relative-last-month-b": [F, F, F],
            "v2-single-x": [F, F, F],  # not a relative-date task: not in any period
        }
        after = {
            "v2-relative-last-winter-a": [F, F, P],
            "v2-relative-last-month-a": [P, P, P],
            "v2-relative-last-month-b": [P, P, F],
            "v2-single-x": [P, P, P],
        }
        found = periods_of(before, after)
        rows = [
            (r.period, r.tasks, r.passed_before, r.passed_after, r.before, r.after) for r in found.periods
        ]
        assert rows == [  # in the order of the rules, not of the tasks
            ("last-month", 2, 1, 2, "failed", "passed"),
            ("last-winter", 1, 1, 0, "passed", "failed"),
        ]
        assert (found.compared, found.gained, found.lost) == (2, 1, 1)
        assert found.p_value == 1.0
        assert found.floor == 0.5  # two periods all one way: 2 / 2**2

    def test_a_period_with_a_task_left_out_is_left_out_and_not_tested(self) -> None:
        before = {"v2-relative-last-summer-a": [P, P, P], "v2-relative-last-summer-b": [P, F]}
        after = {"v2-relative-last-summer-a": [P, P, P], "v2-relative-last-summer-b": [P, P, P]}
        found = periods_of(before, after)
        (row,) = found.periods
        assert (row.before, row.after, row.passed_before, row.passed_after) == ("left out", "passed", 1, 2)
        assert (found.compared, found.gained, found.lost, found.p_value, found.floor) == (0, 0, 0, 1.0, 1.0)

    def test_five_periods_cannot_go_below_one_in_sixteen(self) -> None:
        periods = ["last-month", "last-spring", "last-summer", "last-autumn", "last-winter"]
        before = {f"v2-relative-{period}-a": [F, F, F] for period in periods}
        after = {f"v2-relative-{period}-a": [P, P, P] for period in periods}
        found = periods_of(before, after)
        assert (found.gained, found.p_value, found.floor) == (5, 0.0625, 0.0625)


def calls(*pairs: tuple[str, dict[str, Any]]) -> list[ToolCallRecord]:
    return [
        ToolCallRecord(trace_id="t", request_id=number, tool=tool, arguments=arguments, ok=True)
        for number, (tool, arguments) in enumerate(pairs, start=1)
    ]


OTHER_SCENE = SCENE.model_copy(update={"id": "S2B_32TMR_20250312_0_L2A"})
THIRD_SCENE = SCENE.model_copy(update={"id": "S2A_32TMR_20250314_0_L2A"})


def acting(
    *, cites: bool = True, made: Sequence[ToolCallRecord] = (), cited: Sequence[str] | None = None
) -> TurnRecord:
    """A scored turn that made these calls, and cited the one scene its search returned, or none, or the
    scenes `cited` names among three. A turn may cite only a scene its tools returned (the records check
    it)."""
    base = turn()
    assert base.output is not None
    assert base.output.retrieval is not None
    if cited is None:
        return turn(
            calls=list(made), output=base.output.model_copy(update={"cited_ids": [SCENE.id] if cites else []})
        )
    retrieval = base.output.retrieval.model_copy(update={"scenes": [SCENE, OTHER_SCENE, THIRD_SCENE]})
    output = base.output.model_copy(update={"cited_ids": list(cited), "retrieval": retrieval})
    return turn(calls=list(made), output=output)


def attempt_of(task_id: str, *turns: TurnRecord, outcome: AttemptOutcome = "scored") -> AttemptRecord:
    passed = True if outcome == "scored" else (None if outcome == "infrastructure" else False)
    reason = None if outcome == "scored" else "why"
    return AttemptRecord(
        run_id="r1",
        task_id=task_id,
        attempt=1,
        outcome=outcome,
        passed=passed,
        reason=reason,
        turns=list(turns),
    )


SEARCH = ("search_scenes", {"product": "sentinel-2-l2a", "bbox": [1, 2, 3, 4], "limit": 3})
RESOLVE = ("resolve_place", {"name": "Milan"})


class TestTurnActions:
    def test_a_turn_is_its_calls_in_order_and_the_set_of_scenes_it_cited(self) -> None:
        actions = turn_actions([attempt_of("a", acting(made=calls(RESOLVE, SEARCH)))])
        assert actions == {
            ("a", 1): (
                (
                    ("resolve_place", '{"name": "Milan"}'),
                    ("search_scenes", '{"bbox": [1, 2, 3, 4], "limit": 3, "product": "sentinel-2-l2a"}'),
                ),
                frozenset({SCENE.id}),
            )
        }

    def test_arguments_in_another_order_are_the_same_call(self) -> None:
        reordered = ("search_scenes", {"limit": 3, "bbox": [1, 2, 3, 4], "product": "sentinel-2-l2a"})
        one = turn_actions([attempt_of("a", acting(made=calls(SEARCH)))])
        other = turn_actions([attempt_of("a", acting(made=calls(reordered)))])
        assert one == other

    def test_the_scenes_cited_are_a_set_of_all_of_them(self) -> None:
        two = [SCENE.id, OTHER_SCENE.id]
        one = turn_actions([attempt_of("a", acting(cited=two))])
        reversed_order = turn_actions([attempt_of("a", acting(cited=list(reversed(two))))])
        other = turn_actions([attempt_of("a", acting(cited=[SCENE.id, THIRD_SCENE.id]))])
        assert one[("a", 1)][1] == frozenset(two)
        assert one == reversed_order
        assert one != other

    def test_only_scored_attempts_count_and_turns_are_numbered(self) -> None:
        attempts = [
            attempt_of("a", turn(score=None, oracle=None), outcome="infrastructure"),
            attempt_of("b", acting(), acting(cites=False)),
            attempt_of("c", turn(score=None, oracle=None), outcome="errored"),
        ]
        assert sorted(turn_actions(attempts)) == [("b", 1), ("b", 2)]


def run(**turns: Actions) -> dict[TurnKey, Actions]:
    """A run's actions: keyword `a1` is task `a`, turn 1."""
    return {(key[:-1], int(key[-1])): actions for key, actions in turns.items()}


SEARCHED: Actions = ((("search_scenes", "{}"),), frozenset({"s1"}))
OTHER_SCENES: Actions = ((("search_scenes", "{}"),), frozenset({"s2"}))
OTHER_CALLS: Actions = ((("search_scenes", '{"limit": 3}'),), frozenset({"s1"}))
ASKED: Actions = ((), frozenset())


class TestDeterminism:
    def test_it_counts_the_turns_every_run_reached(self) -> None:
        runs = [
            run(a1=SEARCHED, a2=SEARCHED, b1=ASKED, c1=SEARCHED),
            run(a1=SEARCHED, a2=OTHER_SCENES, b1=ASKED, c1=OTHER_CALLS),
            run(a1=SEARCHED, b1=ASKED, c1=SEARCHED),  # stopped after a's first turn
        ]
        found = determinism(runs, ["a", "b", "c"])
        assert (found.runs, found.turns) == (3, 3)  # a1, b1, c1; a2 not reached by every run
        assert (found.same_calls, found.same_scenes) == (2, 2)  # a1 and b1; c1 has other calls

    def test_same_calls_with_other_scenes_count_for_the_first_only(self) -> None:
        found = determinism([run(a1=SEARCHED), run(a1=OTHER_SCENES)], ["a"])
        assert (found.turns, found.same_calls, found.same_scenes) == (1, 1, 0)

    def test_only_the_tasks_given_count(self) -> None:
        found = determinism([run(a1=SEARCHED, b1=SEARCHED), run(a1=SEARCHED, b1=OTHER_CALLS)], ["a"])
        assert (found.turns, found.same_calls) == (1, 1)

    def test_no_turn_reached_by_every_run_gives_zero_turns(self) -> None:
        found = determinism([run(a1=SEARCHED), run(b1=SEARCHED)], ["a", "b"])
        assert (found.turns, found.same_calls, found.same_scenes) == (0, 0, 0)

    def test_one_run_is_not_compared(self) -> None:
        with pytest.raises(ValueError, match="determinism compares two runs or more, not 1"):
            determinism([run(a1=SEARCHED)], ["a"])


def test_the_turns_here_are_built_with_the_records_helpers() -> None:
    # So they are turns a run can write, and the records' own checks apply to them.
    assert acting().output == output()
    assert acting(cites=False).output == output().model_copy(update={"cited_ids": []})
