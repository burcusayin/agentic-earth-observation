"""The report's statistics: bootstrap intervals that resample whole tasks, the exact sign test for two models,
and how the runs of one task are combined. Pure functions on the standard library (docs/evaluation.md, "Runs
and statistics").

A task's result in one run is `passed`, `failed`, `errored` (a bug of ours; it counts as failed, and the
report lists it apart) or `lost` (every attempt fell to the infrastructure). Lost results are shown two ways:
left out, and counted as failed (decision of 2026-11-07).
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Literal

from pydantic import BaseModel, ConfigDict

Result = Literal["passed", "failed", "errored", "lost"]
LostAs = Literal["excluded", "failed"]

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 1  # fixed, so that the same records always give the same intervals
CONFIDENCE = 0.95


class Interval(BaseModel):
    """A statistic of the tasks as they are, and its percentile bootstrap interval."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    estimate: float
    low: float
    high: float
    confidence: float


def bootstrap_interval[T](
    tasks: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = CONFIDENCE,
) -> Interval:
    """`statistic` of `tasks`, with the interval from `resamples` resamples of whole tasks, drawn with
    replacement. A task is whatever the caller passes, its runs included, so the runs of a task stay together:
    three runs at temperature 0 are not three times the evidence."""
    if not tasks:
        raise ValueError("no task to resample: the interval needs at least one task")
    if resamples < 1:
        raise ValueError(f"resamples must be 1 or more, not {resamples}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, not {confidence}")
    rng = random.Random(seed)  # noqa: S311 - resampling for an interval, not secrecy
    values = sorted(statistic(rng.choices(tasks, k=len(tasks))) for _ in range(resamples))
    tail = (1 - confidence) / 2
    return Interval(
        estimate=statistic(tasks),
        low=quantile(values, tail),
        high=quantile(values, 1 - tail),
        confidence=confidence,
    )


def quantile(ordered: Sequence[float], q: float) -> float:
    """The q-quantile of sorted values, interpolated between the two nearest (numpy's default)."""
    if not ordered:
        raise ValueError("a quantile needs at least one value")
    if not 0 <= q <= 1:
        raise ValueError(f"q must be between 0 and 1, not {q}")
    if any(later < earlier for earlier, later in itertools.pairwise(ordered)):
        raise ValueError("pass the values sorted")
    position = q * (len(ordered) - 1)
    below, above = math.floor(position), math.ceil(position)
    return ordered[below] + (ordered[above] - ordered[below]) * (position - below)


def exact_sign_test(a_only: int, b_only: int) -> float:
    """The two-sided p-value of an exact sign test (McNemar's, exact) on the tasks where two models differ:
    `a_only` tasks only the first passed, `b_only` only the second. Under no difference each such task goes
    either way with probability 1/2."""
    if a_only < 0 or b_only < 0:
        raise ValueError(f"pass counts of tasks, 0 or more, not {a_only} and {b_only}")
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(Fraction(math.comb(n, i), 2**n) for i in range(min(a_only, b_only) + 1))
    return float(min(Fraction(1), 2 * tail))


def task_share(results: Sequence[Result], *, lost: LostAs) -> float | None:
    """The share of a task's runs that passed. With `lost="excluded"`, over the runs with a result, and None
    when no run has one; with `lost="failed"`, over every run."""
    if not results:
        raise ValueError("a task's share needs at least one run")
    counted = [r for r in results if r != "lost"] if lost == "excluded" else list(results)
    if not counted:
        return None
    return sum(1 for r in counted if r == "passed") / len(counted)


def majority(results: Sequence[Result]) -> Literal["passed", "failed"] | None:
    """The result most of a task's runs with a result had; errored counts as failed. None when no run has a
    result, or the runs with one are evenly split (left out of the comparison of two models, and counted)."""
    passed = sum(1 for r in results if r == "passed")
    failed = sum(1 for r in results if r in ("failed", "errored"))
    if passed > failed:
        return "passed"
    if failed > passed:
        return "failed"
    return None


def stability(tasks: Sequence[Sequence[Result]]) -> float:
    """The share of tasks with the same result in every run: all passed, or none passed. A lost run says
    nothing about the model, so a task with one is not counted as stable."""
    if not tasks:
        raise ValueError("stability needs at least one task")
    same = sum(1 for runs in tasks if "lost" not in runs and len({r == "passed" for r in runs}) == 1)
    return same / len(tasks)
