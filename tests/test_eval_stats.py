"""Tests for the report's statistics: the exact sign test against values computed by hand, the bootstrap's
properties, and how the runs of one task are combined (docs/evaluation.md, "Runs and statistics")."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from agentic_earth_observation.evals.stats import (
    Interval,
    Result,
    bootstrap_interval,
    exact_sign_test,
    majority,
    quantile,
    stability,
    task_share,
)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


class TestExactSignTest:
    """Two-sided and exact, on the tasks where two models differ: p = 2 P(X <= min), X ~ Binomial(n, 1/2)."""

    @pytest.mark.parametrize(
        ("a_only", "b_only", "p"),
        [
            (0, 6, 0.03125),  # 2 * 1/64
            (6, 0, 0.03125),  # symmetric
            (1, 5, 0.21875),  # 2 * (1 + 6)/64
            (0, 5, 0.0625),  # 2 * 1/32: five discordant tasks are not enough for 0.05
            (2, 10, 0.03857421875),  # 2 * (1 + 12 + 66)/4096
            (3, 3, 1.0),  # capped: the doubled tail is above 1
            (0, 1, 1.0),
        ],
    )
    def test_matches_the_binomial_computed_by_hand(self, a_only: int, b_only: int, p: float) -> None:
        assert exact_sign_test(a_only, b_only) == pytest.approx(p, abs=1e-12)

    def test_no_task_where_the_models_differ_is_no_evidence(self) -> None:
        assert exact_sign_test(0, 0) == 1.0

    @pytest.mark.parametrize(("a_only", "b_only"), [(-1, 3), (3, -1)])
    def test_a_negative_count_is_refused(self, a_only: int, b_only: int) -> None:
        with pytest.raises(ValueError, match="counts of tasks, 0 or more"):
            exact_sign_test(a_only, b_only)


class TestBootstrap:
    def test_constant_values_give_an_interval_of_no_width(self) -> None:
        found = bootstrap_interval([1.0] * 18, mean, resamples=500)
        assert found == Interval(estimate=1.0, low=1.0, high=1.0, confidence=0.95)

    @pytest.mark.parametrize(("confidence", "low", "high"), [(0.95, 24.975, 974.025), (0.90, 49.95, 949.05)])
    def test_the_interval_is_the_two_tailed_interpolated_percentile_of_the_resamples(
        self, confidence: float, low: float, high: float
    ) -> None:
        # The statistic returns 0, 1, ..., 999 for the resamples and -1 for the tasks as they are, so the
        # interval is the (1 - confidence)/2 and 1 - (1 - confidence)/2 quantiles of 0..999, interpolated
        # between neighbours: 0.025 * 999 = 24.975.
        tasks = [0.0, 1.0]
        drawn = iter(range(1000))

        def counter(resampled: Sequence[float]) -> float:
            return -1.0 if resampled is tasks else float(next(drawn))

        found = bootstrap_interval(tasks, counter, resamples=1000, confidence=confidence)
        assert found.estimate == -1.0
        assert found.low == pytest.approx(low)
        assert found.high == pytest.approx(high)

    def test_the_same_seed_gives_the_same_interval(self) -> None:
        values = [1.0] * 12 + [0.0] * 6
        first = bootstrap_interval(values, mean, resamples=2000, seed=7)
        assert bootstrap_interval(values, mean, resamples=2000, seed=7) == first

    def test_the_seed_decides_the_resamples(self) -> None:
        def means_seen(seed: int) -> list[float]:
            seen: list[float] = []

            def record(tasks: Sequence[int]) -> float:
                seen.append(mean(tasks))
                return 0.0

            bootstrap_interval(list(range(18)), record, resamples=50, seed=seed)
            return seen

        assert means_seen(7) == means_seen(7)
        assert means_seen(7) != means_seen(8)

    def test_the_estimate_is_the_statistic_of_the_tasks_as_they_are_and_lies_inside(self) -> None:
        values = [1.0] * 14 + [0.0] * 4
        found = bootstrap_interval(values, mean)
        assert found.estimate == pytest.approx(14 / 18)
        assert found.low <= found.estimate <= found.high

    def test_the_interval_is_roughly_the_one_the_rules_expect_for_eighteen_tasks(self) -> None:
        # docs/evaluation.md: "a single model's interval is roughly +-20 points" with 18 tasks.
        found = bootstrap_interval([1.0] * 9 + [0.0] * 9, mean)
        assert 0.15 <= (found.high - found.low) / 2 <= 0.25

    def test_more_spread_gives_a_wider_interval(self) -> None:
        narrow = bootstrap_interval([1.0] * 17 + [0.0], mean)
        wide = bootstrap_interval([1.0] * 9 + [0.0] * 9, mean)
        assert wide.high - wide.low > narrow.high - narrow.low

    def test_a_higher_confidence_gives_a_wider_interval(self) -> None:
        values = [1.0] * 10 + [0.0] * 8
        assert (
            bootstrap_interval(values, mean, confidence=0.99).high
            - bootstrap_interval(values, mean, confidence=0.99).low
            > bootstrap_interval(values, mean, confidence=0.80).high
            - bootstrap_interval(values, mean, confidence=0.80).low
        )

    def test_whole_tasks_are_resampled_not_their_runs(self) -> None:
        # Each task carries its three runs together; the statistic sees tasks, so a resample never mixes the
        # runs of different tasks.
        tasks = [("passed", "passed", "passed")] * 9 + [("failed", "failed", "failed")] * 9
        seen: list[int] = []

        def count_mixed(resampled: Sequence[tuple[str, ...]]) -> float:
            seen.append(sum(1 for runs in resampled if len(set(runs)) > 1))
            return 0.0

        bootstrap_interval(tasks, count_mixed, resamples=200)
        assert set(seen) == {0}

    @pytest.mark.parametrize(
        ("kwargs", "said"),
        [
            ({"resamples": 0}, "resamples must be 1 or more"),
            ({"confidence": 1.0}, "confidence must be between 0 and 1"),
            ({"confidence": 0.0}, "confidence must be between 0 and 1"),
        ],
    )
    def test_settings_out_of_range_are_refused(self, kwargs: dict[str, Any], said: str) -> None:
        with pytest.raises(ValueError, match=said):
            bootstrap_interval([1.0, 0.0], mean, **kwargs)

    def test_no_task_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no task to resample"):
            bootstrap_interval([], mean)


class TestTaskShare:
    """How much of one task's runs passed, with lost runs excluded or counted as failed."""

    @pytest.mark.parametrize(
        ("results", "excluded", "failed"),
        [
            (("passed", "passed", "passed"), 1.0, 1.0),
            (("passed", "failed", "passed"), 2 / 3, 2 / 3),
            (("passed", "errored", "passed"), 2 / 3, 2 / 3),  # errored counts as failed
            (("passed", "lost", "passed"), 1.0, 2 / 3),
            (("failed", "lost", "lost"), 0.0, 0.0),
            (("lost", "lost", "lost"), None, 0.0),  # no result: left out, or a failure
        ],
    )
    def test_the_share_of_runs_that_passed(
        self, results: tuple[Result, ...], excluded: float | None, failed: float
    ) -> None:
        assert task_share(results, lost="excluded") == (
            pytest.approx(excluded) if excluded is not None else None
        )
        assert task_share(results, lost="failed") == pytest.approx(failed)

    def test_a_task_without_runs_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one run"):
            task_share((), lost="excluded")


class TestMajority:
    @pytest.mark.parametrize(
        ("results", "expected"),
        [
            (("passed", "passed", "failed"), "passed"),
            (("passed", "failed", "failed"), "failed"),
            (("passed", "errored", "errored"), "failed"),
            (("passed", "passed", "lost"), "passed"),
            (("passed", "failed", "lost"), None),  # a tie among the runs with a result: no majority
            (("lost", "lost", "lost"), None),
            (("passed",), "passed"),
        ],
    )
    def test_the_result_most_runs_had(self, results: tuple[Result, ...], expected: str | None) -> None:
        assert majority(results) == expected


class TestStability:
    def test_the_share_of_tasks_with_the_same_result_in_every_run(self) -> None:
        tasks: list[tuple[Result, ...]] = [
            ("passed", "passed", "passed"),  # same
            ("failed", "errored", "failed"),  # same: not passed in any run
            ("passed", "failed", "passed"),  # differs
            ("passed", "passed", "lost"),  # a lost run breaks sameness: it says nothing
            ("failed", "lost", "failed"),  # even when the other runs agree
        ]
        assert stability(tasks) == pytest.approx(2 / 5)

    def test_no_task_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one task"):
            stability([])


class TestQuantile:
    @pytest.mark.parametrize(
        ("values", "q", "expected"),
        [
            ([5.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0], 0.5, 12.0),  # the middle one
            ([5.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0], 0.9, 21.4),  # 0.9 * 6 = 5.4: 21 + 0.4 * (22 - 21)
            ([1.0, 2.0], 0.5, 1.5),  # between the two
            ([7.0], 0.9, 7.0),
            ([1.0, 2.0, 3.0], 0.0, 1.0),
            ([1.0, 2.0, 3.0], 1.0, 3.0),
        ],
    )
    def test_interpolates_between_the_nearest_values(
        self, values: list[float], q: float, expected: float
    ) -> None:
        assert quantile(values, q) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("values", "q", "said"),
        [
            ([], 0.5, "at least one value"),
            ([1.0], 1.5, "q must be between 0 and 1"),
            ([2.0, 1.0], 0.5, "pass the values sorted"),
        ],
    )
    def test_refuses_what_has_no_quantile(self, values: list[float], q: float, said: str) -> None:
        with pytest.raises(ValueError, match=said):
            quantile(values, q)
