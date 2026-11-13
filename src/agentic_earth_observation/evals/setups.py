"""How two setups of one model are compared (docs/evaluation.md, "Comparing two setups").

A setup is a set of runs with the same code, prompts, tool descriptions and model options; which runs make one
is the report's business (`evals/report.py`). This module takes what each task came to in each run of a setup,
and says what changed between two setups: task by task, per relative period, and how often the runs of a
setup did the same thing. Pure functions, no files.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agentic_earth_observation.evals.groups import PERIODS, relative_period
from agentic_earth_observation.evals.records import AttemptRecord
from agentic_earth_observation.evals.stats import Result, exact_sign_test, majority

PeriodVerdict = Literal["passed", "failed", "left out"]
# A turn's actions, as the determinism measures compare them: its tool calls in order, each its tool and its
# arguments, and the set of scene ids its answer cited.
Calls = tuple[tuple[str, str], ...]
Actions = tuple[Calls, frozenset[str]]
TurnKey = tuple[str, int]  # a task id and the turn's number in it, from 1


class Change(BaseModel):
    """One model's tasks before and after: each task by the majority of its runs in each setup."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scope: str
    tasks: int
    both_passed: int
    both_failed: int
    gained: list[str]  # failed before and passed after, by id
    lost: list[str]  # passed before and failed after, by id
    left_out: int  # tasks without a majority in either setup
    p_value: float  # the exact two-sided sign test on gained against lost


class PeriodResult(BaseModel):
    """One relative period before and after: how many of its tasks passed, and whether the period did."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    period: str
    tasks: int
    passed_before: int
    passed_after: int
    before: PeriodVerdict
    after: PeriodVerdict


class PeriodChange(BaseModel):
    """The relative periods before and after, and the sign test over those with a verdict on both sides."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    periods: list[PeriodResult]
    compared: int  # periods with a verdict before and after
    gained: int
    lost: int
    p_value: float
    floor: float  # the smallest p-value the test can give with `compared` periods


class Determinism(BaseModel):
    """How often every run of a setup did the same thing in a turn, over the turns every run reached."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    runs: int = Field(ge=2)
    turns: int = Field(ge=0)  # turns that every run reached in a scored attempt
    same_calls: int = Field(ge=0)  # ... where every run made the same tool calls: tools, arguments, order
    same_scenes: int = Field(ge=0)  # ... and where every run also cited the same set of scenes


def change(
    scope: str,
    task_ids: Sequence[str],
    before: Mapping[str, Sequence[Result]],
    after: Mapping[str, Sequence[Result]],
) -> Change:
    """`task_ids` before and after: each task by the majority of its runs in each setup (`stats.majority`:
    a lost run has no result, an errored run counts as failed). A task without a majority on either side is
    left out, and counted."""
    gained: list[str] = []
    lost: list[str] = []
    both_passed = both_failed = left_out = 0
    for task_id in task_ids:
        first, second = majority(before[task_id]), majority(after[task_id])
        if first is None or second is None:
            left_out += 1
        elif first == second == "passed":
            both_passed += 1
        elif first == second:
            both_failed += 1
        elif second == "passed":
            gained.append(task_id)
        else:
            lost.append(task_id)
    return Change(
        scope=scope,
        tasks=len(task_ids),
        both_passed=both_passed,
        both_failed=both_failed,
        gained=gained,
        lost=lost,
        left_out=left_out,
        p_value=exact_sign_test(len(gained), len(lost)),
    )


def period_change(
    task_ids: Sequence[str],
    before: Mapping[str, Sequence[Result]],
    after: Mapping[str, Sequence[Result]],
) -> PeriodChange | None:
    """The relative periods of `task_ids` before and after, or None when no task asks for one. A period
    passes when all its tasks pass; a period with a task left out is left out, and counted."""
    by_period: dict[str, list[str]] = {}
    for task_id in task_ids:
        if (period := relative_period(task_id)) is not None:
            by_period.setdefault(period, []).append(task_id)
    if not by_period:
        return None
    results = [
        _period_result(period, by_period[period], before, after) for period in PERIODS if period in by_period
    ]
    compared = [result for result in results if "left out" not in (result.before, result.after)]
    gained = sum(1 for result in compared if (result.before, result.after) == ("failed", "passed"))
    lost = sum(1 for result in compared if (result.before, result.after) == ("passed", "failed"))
    return PeriodChange(
        periods=results,
        compared=len(compared),
        gained=gained,
        lost=lost,
        p_value=exact_sign_test(gained, lost),
        floor=exact_sign_test(len(compared), 0),
    )


def _period_result(
    period: str,
    task_ids: Sequence[str],
    before: Mapping[str, Sequence[Result]],
    after: Mapping[str, Sequence[Result]],
) -> PeriodResult:
    first = [majority(before[task_id]) for task_id in task_ids]
    second = [majority(after[task_id]) for task_id in task_ids]
    return PeriodResult(
        period=period,
        tasks=len(task_ids),
        passed_before=first.count("passed"),
        passed_after=second.count("passed"),
        before=_verdict(first),
        after=_verdict(second),
    )


def _verdict(results: Sequence[Literal["passed", "failed"] | None]) -> PeriodVerdict:
    if None in results:
        return "left out"
    return "passed" if all(result == "passed" for result in results) else "failed"


def turn_actions(attempts: Sequence[AttemptRecord]) -> dict[TurnKey, Actions]:
    """What each turn of a run's scored attempts did: its tool calls, as the MCP server logged them, and the
    scenes its answer cited. Only a scored attempt is a task's last (an attempt lost to the infrastructure is
    tried again, an errored one is a bug of ours), so a task has at most one."""
    actions: dict[TurnKey, Actions] = {}
    for attempt in attempts:
        if attempt.outcome != "scored":
            continue
        for number, turn in enumerate(attempt.turns, start=1):
            calls = tuple((call.tool, json.dumps(call.arguments, sort_keys=True)) for call in turn.calls)
            assert turn.output is not None  # noqa: S101 - a scored turn has its output (the records check it)
            actions[(attempt.task_id, number)] = (calls, frozenset(turn.output.cited_ids))
    return actions


def determinism(runs: Sequence[Mapping[TurnKey, Actions]], task_ids: Sequence[str]) -> Determinism:
    """Over the turns of `task_ids` that every run reached in a scored attempt: in how many every run made
    the same tool calls, and in how many it also cited the same scenes. ValueError for fewer than two runs."""
    if len(runs) < 2:
        raise ValueError(f"determinism compares two runs or more, not {len(runs)}")
    wanted = set(task_ids)
    reached = set.intersection(*({key for key in run if key[0] in wanted} for run in runs))
    same_calls = same_scenes = 0
    for key in reached:
        if len({run[key][0] for run in runs}) == 1:
            same_calls += 1
            if len({run[key][1] for run in runs}) == 1:
                same_scenes += 1
    return Determinism(runs=len(runs), turns=len(reached), same_calls=same_calls, same_scenes=same_scenes)
