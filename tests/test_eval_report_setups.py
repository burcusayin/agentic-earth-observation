"""Tests for the report's comparison of setups (evals/report.py; docs/evaluation.md, "Comparing two setups";
decision of 2026-11-12): which runs make a setup, the runs of an earlier golden set, and what the report says
of two setups. The numbers themselves are checked in tests/test_eval_setups.py."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from agentic_earth_observation import cli
from agentic_earth_observation.config import ROOT
from agentic_earth_observation.evals import report
from agentic_earth_observation.evals.golden import Category, Task, golden_hash, load_golden
from agentic_earth_observation.evals.records import (
    AttemptRecord,
    CodeVersion,
    ModelBuild,
    TaskCounts,
    read_run_record,
)
from agentic_earth_observation.evals.report import (
    EARLIER_GOLDEN_SETS,
    EarlierGoldenSet,
    ListedRun,
    Report,
    ReportError,
    build_report,
    collect_runs,
    group_setups,
    mark_comparable,
    render_markdown,
    tasks_digest,
)
from agentic_earth_observation.evals.stats import Result
from tests.test_eval_records import STARTED, git
from tests.test_eval_report import (
    COMMIT,
    GOLDEN,
    ONE_TURN,
    TASKS,
    hashes,
    model_build,
    record,
    search_task,
    write_run,
)
from tests.test_eval_setups import acting, attempt_of


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository with a first commit of code."""
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    (root / "src.py").write_text("x = 1\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "code")
    return root


OLD_GOLDEN = "8" * 64
OLD_RULES = "7" * 64
# The start of why a run on it is not read
RAN_ON_OLD = (
    "the golden set and docs/evaluation.md differ from what this report reads: it ran on golden-old, "
)


def run(
    run_id: str,
    results: dict[str, Result] | None = None,
    *,
    model: str = "m1",
    build: ModelBuild | None = None,
    state: str = "reported",
    attempts: Sequence[AttemptRecord] = (),
    **changes: Any,  # noqa: ANN401 - fields of the record
) -> ListedRun:
    """A run as the report lists it, with its record, its results and its attempts."""
    passed: Result = "passed"
    found = results if results is not None else dict.fromkeys(TASKS, passed)
    tally = {what: sum(1 for result in found.values() if result == what) for what in report.COUNTED}
    fields: dict[str, Any] = {
        "model": build or model_build(model),
        "task_order": list(found),
        "tasks": TaskCounts(**tally),
    } | changes
    return ListedRun(
        folder=run_id,
        state=state,  # type: ignore[arg-type]  # a test names the state as text
        record=record(run_id, **fields),
        results=found,
        attempts=list(attempts),
    )


UNKNOWN = "fedcba9876543210" * 2 + "01234567"  # a commit no repository here has


def on(commit: str) -> CodeVersion:
    return CodeVersion(commit=commit, dirty=False)


def folders(groups: Sequence[Sequence[ListedRun]]) -> list[list[str]]:
    return [[one.folder for one in group] for group in groups]


class TestGroupSetups:
    """A model's runs, in the order they ran, cut into its setups."""

    def test_runs_on_one_setup_are_one_setup(self) -> None:
        assert folders(group_setups([run("r1"), run("r2"), run("r3")])) == [["r1", "r2", "r3"]]

    def test_a_change_of_setup_starts_a_new_one(self) -> None:
        runs = [run("r1"), run("r2"), run("r3", shuffle_seed=7), run("r4", shuffle_seed=7)]
        assert folders(group_setups(runs)) == [["r1", "r2"], ["r3", "r4"]]

    def test_a_setup_that_comes_back_is_a_new_one(self) -> None:
        # The runs of one phase come from one command: setups are cut where they change.
        runs = [run("r1"), run("r2", shuffle_seed=7), run("r3")]
        assert folders(group_setups(runs)) == [["r1"], ["r2"], ["r3"]]

    def test_another_build_of_the_model_starts_a_new_one(self) -> None:
        runs = [run("r1"), run("r2", build=model_build("m1", "another-digest"))]
        assert folders(group_setups(runs)) == [["r1"], ["r2"]]

    def test_another_golden_set_starts_a_new_one(self) -> None:
        # Two earlier golden sets are two setups, whatever else they share.
        runs = [run("r1", hashes=hashes(golden=OLD_GOLDEN)), run("r2", hashes=hashes(golden="6" * 64))]
        assert folders(group_setups(runs)) == [["r1"], ["r2"]]

    def test_runs_on_two_commits_that_differ_only_in_the_records_are_one_setup(self, repo: Path) -> None:
        first = git(repo, "rev-parse", "HEAD").strip()
        (repo / "results").mkdir()
        (repo / "results" / "run.json").write_text("{}", encoding="utf-8")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "records")
        second = git(repo, "rev-parse", "HEAD").strip()
        runs = [run("r1", code=on(first)), run("r2", code=on(second))]
        assert folders(group_setups(runs, repo)) == [["r1", "r2"]]

    def test_a_run_on_a_commit_this_repository_does_not_have_is_a_setup_of_its_own(self, repo: Path) -> None:
        known = git(repo, "rev-parse", "HEAD").strip()
        runs = [
            run("r1", code=on(known)),
            run("r2", code=on(UNKNOWN)),
            run("r3", code=on(UNKNOWN)),  # the same commit: comparable
            run("r4", code=on(known)),
        ]
        assert folders(group_setups(runs, repo)) == [["r1"], ["r2", "r3"], ["r4"]]

    def test_the_runs_of_one_model_only(self) -> None:
        with pytest.raises(ValueError, match="setups are cut for one model at a time, not for m1, m2"):
            group_setups([run("r1"), run("r2", model="m2")])

    def test_a_run_without_its_record_cannot_be_grouped(self) -> None:
        with pytest.raises(ValueError, match="r1 has no record to compare"):
            group_setups([ListedRun(folder="r1", state="aborted")])


class TestTheHeadlineIsAsBefore:
    """The numbers above the comparison count what they counted before Part 2 (decision of 2026-11-12)."""

    def test_a_rebuild_of_one_model_leaves_the_other_model_counted(self) -> None:
        runs = [run("a1"), run("b1", model="m2"), run("b2", build=model_build("m2", "another-digest"))]
        found = mark_comparable(runs)
        assert [(one.folder, one.state, one.reason) for one in found] == [
            ("a1", "reported", None),
            ("b1", "other_setup", "it ran another build of m2 than b2"),
            ("b2", "reported", None),
        ]

    def test_a_run_like_the_latest_counts_even_if_another_setup_ran_between(self) -> None:
        found = mark_comparable([run("r1"), run("r2", shuffle_seed=7), run("r3")])
        assert [one.state for one in found] == ["reported", "other_setup", "reported"]

    def test_the_runs_of_earlier_setups_keep_their_results_and_attempts(self) -> None:
        attempts = [attempt_of("dev-a", acting())]
        found = mark_comparable([run("r1", shuffle_seed=7, attempts=attempts), run("r2")])
        assert found[0].state == "other_setup"
        assert (found[0].results, found[0].attempts) == (dict.fromkeys(TASKS, "passed"), attempts)


class TestEarlierGoldenSets:
    """A run on an earlier golden set is read only when `EARLIER_GOLDEN_SETS` names it and its tasks are
    unchanged; otherwise it is listed as before, with what differs."""

    TASKS_OF_IT = ("dev-a", "dev-b")

    def tasks(self) -> list[Task]:
        return [search_task(task_id, Category.SINGLE_SEARCH) for task_id in TASKS]

    def known(self, monkeypatch: pytest.MonkeyPatch, tasks: Sequence[Task] | None = None) -> None:
        ran = [task for task in (tasks or self.tasks()) if task.id in self.TASKS_OF_IT]
        earlier = EarlierGoldenSet(
            tag="golden-old", golden=OLD_GOLDEN, evaluation_rules=OLD_RULES, tasks={"dev": tasks_digest(ran)}
        )
        monkeypatch.setattr(report, "EARLIER_GOLDEN_SETS", (earlier,))

    def write(self, runs: Path, **changes: Any) -> None:  # noqa: ANN401 - fields of the record
        fields: dict[str, Any] = {
            "hashes": hashes(golden=OLD_GOLDEN, evaluation_rules=OLD_RULES),
            "task_order": list(self.TASKS_OF_IT),
            "tasks": TaskCounts(passed=1, failed=1),
        } | changes
        write_run(runs, "r1", {"dev-a": [("scored", True)], "dev-b": [("scored", False)]}, **fields)

    def collect(self, runs: Path, tasks: Sequence[Task] | None = None) -> ListedRun:
        (found,) = collect_runs(
            runs, "dev", ONE_TURN, golden=GOLDEN, tasks=self.tasks() if tasks is None else tasks
        )
        return found

    def test_it_is_read_with_its_results_on_the_tasks_it_ran(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.known(monkeypatch)
        self.write(tmp_path)
        found = self.collect(tmp_path)
        assert found.state == "earlier_golden_set"
        assert (
            found.reason
            == "it ran on golden-old, on 2 of these tasks: it is read only for the comparison of setups"
        )
        assert found.results == {"dev-a": "passed", "dev-b": "failed"}
        assert len(found.attempts) == 2

    def test_a_task_that_changed_since_makes_it_another_golden_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.known(monkeypatch)
        self.write(tmp_path)
        changed = [
            task.model_copy(update={"notes": "changed"}) if task.id == "dev-a" else task
            for task in self.tasks()
        ]
        found = self.collect(tmp_path, changed)
        assert (found.state, found.results) == ("other_golden_set", {})
        assert found.reason == f"{RAN_ON_OLD}and its tasks have changed since"

    @pytest.mark.parametrize(
        ("changes", "reason"),
        [
            (
                {"hashes": hashes(golden=OLD_GOLDEN, evaluation_rules=OLD_RULES, evaluation_code="6" * 64)},
                "the golden set and the evaluation code and docs/evaluation.md differ from what this report "
                "reads: it ran on golden-old, on other evaluation code or another gazetteer than this "
                "report reads",
            ),
            (
                {"hashes": hashes(golden=OLD_GOLDEN, evaluation_rules=OLD_RULES, gazetteer="6" * 64)},
                "the golden set and the gazetteer and docs/evaluation.md differ from what this report "
                "reads: it ran on golden-old, on other evaluation code or another gazetteer than this "
                "report reads",
            ),
            (  # another version of the rules: not the pinned set
                {"hashes": hashes(golden=OLD_GOLDEN, evaluation_rules="6" * 64)},
                "the golden set and docs/evaluation.md differ from what this report reads",
            ),
            (
                {"hashes": hashes(golden="6" * 64, evaluation_rules=OLD_RULES)},
                "the golden set and docs/evaluation.md differ from what this report reads",
            ),
        ],
    )
    def test_other_frozen_files_make_it_another_golden_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changes: dict[str, Any], reason: str
    ) -> None:
        self.known(monkeypatch)
        self.write(tmp_path, **changes)
        found = self.collect(tmp_path)
        assert (found.state, found.reason) == ("other_golden_set", reason)

    def test_without_the_splits_tasks_nothing_earlier_is_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.known(monkeypatch)
        self.write(tmp_path)
        found = self.collect(tmp_path, tasks=[])
        assert (found.state, found.reason) == (
            "other_golden_set",
            f"{RAN_ON_OLD}and the report was not given the split's tasks to check them against",
        )

    def test_a_task_the_split_no_longer_has_makes_it_another_golden_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.known(monkeypatch)
        self.write(tmp_path, task_order=["dev-a", "dev-gone"])
        found = self.collect(tmp_path)
        assert (found.state, found.reason) == (
            "other_golden_set",
            f"{RAN_ON_OLD}and some of its tasks are not tasks of this split",
        )

    def test_an_earlier_golden_set_of_the_other_split_does_not_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ran = [task for task in self.tasks() if task.id in self.TASKS_OF_IT]
        earlier = EarlierGoldenSet(
            tag="g", golden=OLD_GOLDEN, evaluation_rules=OLD_RULES, tasks={"test": tasks_digest(ran)}
        )
        monkeypatch.setattr(report, "EARLIER_GOLDEN_SETS", (earlier,))
        self.write(tmp_path)
        assert self.collect(tmp_path).state == "other_golden_set"

    def test_its_results_must_add_up_as_a_counted_runs_do(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.known(monkeypatch)
        self.write(tmp_path, tasks=TaskCounts(passed=2))
        found = self.collect(tmp_path)
        assert (found.state, found.reason) == (
            "inconsistent",
            "run.json counts 2 passed, 0 failed, 0 errored, 0 lost; "
            "attempts.jsonl gives 1 passed, 1 failed, 0 errored, 0 lost",
        )

    def test_an_earlier_set_that_names_no_tasks_of_the_split_says_so(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        earlier_set = EarlierGoldenSet(
            tag="golden-old", golden=OLD_GOLDEN, evaluation_rules=OLD_RULES, tasks={}
        )
        monkeypatch.setattr(report, "EARLIER_GOLDEN_SETS", (earlier_set,))
        self.write(tmp_path)
        found = self.collect(tmp_path)
        assert (found.state, found.reason) == (
            "other_golden_set",
            f"{RAN_ON_OLD}which names no tasks of the dev split",
        )


class TestTheDigestOfTasks:
    def test_equal_tasks_give_equal_digests_in_any_order(self) -> None:
        a, b = search_task("dev-a", Category.SINGLE_SEARCH), search_task("dev-b", Category.EDGE)
        assert tasks_digest([a, b]) == tasks_digest([b, a])
        assert tasks_digest([a]) != tasks_digest([a.model_copy(update={"notes": "x"})])
        assert tasks_digest([a]) != tasks_digest([a, b])


class TestTheRunsOfV10:
    """The six runs of v1.0 in results/runs/ are on golden-test-v1. Checked by counts only, and every assert
    compares plain numbers or names: a failing assert must not print a run's record or attempts, which would
    show the test split to whoever reads the output. Runs on later golden sets may sit beside them."""

    def runs_on_v1(self) -> list[ListedRun]:
        tasks = [task for task in load_golden() if task.split == "test"]
        found = collect_runs(
            ROOT / "results" / "runs",
            "test",
            {task.id: len(task.turns) for task in tasks},
            golden=report_golden_hash(),
            tasks=tasks,
        )
        (v1,) = EARLIER_GOLDEN_SETS
        return [one for one in found if one.record is not None and one.record.hashes.golden == v1.golden]

    def test_they_are_read_as_an_earlier_golden_set_with_their_own_counts(self) -> None:
        runs = self.runs_on_v1()
        states = sorted({one.state for one in runs})
        tasks_ran = sorted({len(one.results) for one in runs})
        adds_up = [
            {what: sum(1 for result in one.results.values() if result == what) for what in report.COUNTED}
            == one.record.tasks.model_dump()
            for one in runs
            if one.record is not None
        ]
        number = len(runs)
        assert number == 6
        assert states == ["earlier_golden_set"]
        assert tasks_ran == [18]
        assert adds_up == [True] * 6

    def test_the_pinned_digest_is_that_of_the_tasks_they_ran(self) -> None:
        (v1,) = EARLIER_GOLDEN_SETS
        by_id = {task.id: task for task in load_golden()}
        folders = sorted(path for path in (ROOT / "results" / "runs").iterdir() if path.is_dir())
        records = [read_run_record(folder) for folder in folders]
        on_v1 = [found for found in records if found.hashes.golden == v1.golden]
        same_rules = all(found.hashes.evaluation_rules == v1.evaluation_rules for found in on_v1)
        digests = {
            tasks_digest([by_id[task_id] for task_id in found.task_order]) == v1.tasks["test"]
            for found in on_v1
        }
        number = len(on_v1)
        assert number == 6
        assert same_rules
        assert digests == {True}


def report_golden_hash() -> str:
    return golden_hash()


V1_TASK = "single-a"  # a task of v1: its id does not start with v2-
TASK_IDS = [
    V1_TASK,
    "v2-single-b",
    "v2-relative-last-month-a",
    "v2-relative-last-month-b",
    "v2-relative-last-winter-a",
]


def setup_tasks() -> list[Task]:
    return [search_task(task_id, Category.SINGLE_SEARCH) for task_id in TASK_IDS]


def results(*values: Result) -> dict[str, Result]:
    return dict(zip(TASK_IDS, values, strict=True))


BEFORE = results("passed", "failed", "failed", "passed", "passed")
AFTER = results("passed", "passed", "passed", "passed", "failed")


def three(prefix: str, found: dict[str, Result], **changes: Any) -> list[ListedRun]:  # noqa: ANN401
    return [run(f"{prefix}{number}", found, **changes) for number in (1, 2, 3)]


# A change to what the agents are given: another prompt. A setup made by another seed only is not a change.
CHANGED: dict[str, Any] = {"hashes": hashes(retrieval_prompt="0" * 64)}
CHANGED_AGAIN: dict[str, Any] = {"hashes": hashes(retrieval_prompt="0" * 64, decide_prompt="0" * 64)}


def build(runs: Sequence[ListedRun], **options: Any) -> Report:  # noqa: ANN401 - build_report's options
    return build_report("dev", runs, setup_tasks(), **options)


def setups_of(found: Report) -> list[tuple[str, int, str | None, list[str], str | None]]:
    return [(s.model, s.number, s.golden, s.runs, s.differs) for s in found.setups]


class TestChangesBetweenSetups:
    def test_the_first_change_is_shown_on_all_tasks_on_those_of_v1_and_on_those_added_in_v2(self) -> None:
        found = build([*three("a", BEFORE), *three("b", AFTER, **CHANGED)])
        assert setups_of(found) == [
            ("m1", 1, None, ["a1", "a2", "a3"], None),
            ("m1", 2, None, ["b1", "b2", "b3"], "its retrieval prompt differs from that of a1"),
        ]
        (pair,) = found.changes
        assert (pair.model, pair.before, pair.after, pair.first) == ("m1", 1, 2, True)
        rows = [
            (row.scope, row.both_passed, row.both_failed, row.gained, row.lost, row.left_out)
            for row in pair.rows
        ]
        assert rows == [
            (
                "All 5 tasks",
                2,
                0,
                ["v2-single-b", "v2-relative-last-month-a"],
                ["v2-relative-last-winter-a"],
                0,
            ),
            ("The 1 of v1, seen when the change was chosen", 1, 0, [], [], 0),
            (
                "The 4 added in v2",
                1,
                0,
                ["v2-single-b", "v2-relative-last-month-a"],
                ["v2-relative-last-winter-a"],
                0,
            ),
        ]
        assert [row.tasks for row in pair.rows] == [5, 1, 4]
        assert pair.periods is not None
        assert [(p.period, p.before, p.after) for p in pair.periods.periods] == [
            ("last-month", "failed", "passed"),
            ("last-winter", "passed", "failed"),
        ]

    def test_the_headline_counts_only_the_latest_setup(self) -> None:
        found = build([*three("a", BEFORE), *three("b", AFTER, **CHANGED)])
        (summary,) = found.models
        assert summary.runs == ["b1", "b2", "b3"]
        assert [one.state for one in found.listed] == ["other_setup"] * 3 + ["reported"] * 3

    def test_a_later_change_is_shown_on_all_tasks_only(self) -> None:
        found = build(
            [*three("a", BEFORE), *three("b", AFTER, **CHANGED), *three("c", BEFORE, **CHANGED_AGAIN)]
        )
        first, later = found.changes
        assert (first.first, len(first.rows)) == (True, 3)
        assert (later.first, later.before, later.after, [row.scope for row in later.rows]) == (
            False,
            2,
            3,
            ["All 5 tasks"],
        )
        assert later.periods is not None  # the relative periods are shown for every change

    def test_each_model_is_compared_with_its_own_setups(self) -> None:
        # One model's Ollama version, or its build, does not cut the other model's setups.
        q_build = model_build("q").model_copy(update={"server_version": "0.41.0"})
        runs = [
            *three("q", BEFORE, model="q"),
            *three("m", BEFORE, model="m"),
            *three("r", AFTER, build=q_build, **CHANGED),  # q after its change, on another Ollama
            *three("n", AFTER, model="m", **CHANGED),
        ]
        found = build(runs)
        assert [(pair.model, pair.before, pair.after, pair.first) for pair in found.changes] == [
            ("q", 1, 2, True),
            ("m", 1, 2, True),
        ]
        assert all(len(pair.rows) == 3 for pair in found.changes)

    def test_a_cut_in_the_environment_is_not_the_first_change(self) -> None:
        # The baseline restarted after an Ollama upgrade, then the real change: only the second is a change.
        upgraded = model_build("m1").model_copy(update={"server_version": "0.41.0"})
        runs = [
            run("a1", BEFORE),
            run("a2", BEFORE),
            run("a3", BEFORE, build=upgraded),
            *three("b", AFTER, build=upgraded, **CHANGED),
        ]
        found = build(runs)
        assert [(p.before, p.after, p.change, p.first, len(p.rows)) for p in found.changes] == [
            (1, 2, False, False, 1),
            (2, 3, True, True, 3),
        ]
        text = render_markdown(found)
        assert (
            "Nothing the agents are given or run differs between these two setups, only their environment"
            in text
        )
        assert "says nothing and is not shown" not in text  # neither pair is a change after the first

    @pytest.mark.parametrize(
        "environment",
        [
            {"shuffle_seed": 7},
            {"reference_date": date(2025, 12, 14)},
            {"build": model_build("m1", "another-digest")},
            {"build": model_build("m1").model_copy(update={"server_version": "0.41.0"})},
        ],
    )
    def test_the_environment_is_not_a_change(self, environment: dict[str, Any]) -> None:
        (pair,) = build([*three("a", BEFORE), *three("b", AFTER, **environment)]).changes
        assert (pair.change, pair.first) == (False, False)

    @pytest.mark.parametrize(
        "chosen",
        [
            {"hashes": hashes(decide_prompt="0" * 64)},
            {"hashes": hashes(tool_descriptions="0" * 64)},
            {
                "settings": {
                    "turn_deadline_s": 1.0,
                    "max_model_calls": 6,
                    "mcp_url": "http://127.0.0.1:8001/mcp",
                }
            },
            {"build": model_build("m1").model_copy(update={"options": {"seed": 7}})},
            {
                "code": CodeVersion(commit=COMMIT, dirty=True)
            },  # what it ran cannot be known: a change, to be safe
        ],
    )
    def test_what_the_agents_are_given_or_run_is_a_change(self, chosen: dict[str, Any]) -> None:
        (pair,) = build([*three("a", BEFORE), *three("b", AFTER, **chosen)]).changes
        assert (pair.change, pair.first) == (True, True)

    def test_code_the_agents_run_is_a_change_and_other_code_is_not(self, repo: Path) -> None:
        commits = [git(repo, "rev-parse", "HEAD").strip()]
        (repo / "README.md").write_text("words\n", encoding="utf-8")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "docs")
        commits.append(git(repo, "rev-parse", "HEAD").strip())
        (repo / "src" / "agentic_earth_observation").mkdir(parents=True)
        (repo / "src" / "agentic_earth_observation" / "catalog.py").write_text("x = 1\n", encoding="utf-8")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "code")
        commits.append(git(repo, "rev-parse", "HEAD").strip())
        runs = [
            run("a1", BEFORE, code=on(commits[0])),
            run("b1", AFTER, code=on(commits[1])),
            run("c1", BEFORE, code=on(commits[2])),
        ]
        found = build(runs, root=repo)
        assert [(p.change, p.first) for p in found.changes] == [(False, False), (True, True)]

    def test_a_commit_this_repository_does_not_have_may_be_a_change(self, repo: Path) -> None:
        known = git(repo, "rev-parse", "HEAD").strip()
        runs = [run("a1", BEFORE, code=on(UNKNOWN)), run("b1", AFTER, code=on(known))]
        (pair,) = build(runs, root=repo).changes
        assert (pair.change, pair.first) == (True, True)

    def test_a_restarted_evaluation_gives_a_setup_more_runs(self) -> None:
        found = build([*three("a", BEFORE), run("a4", BEFORE), *three("b", AFTER, **CHANGED)])
        assert setups_of(found)[0][3] == ["a1", "a2", "a3", "a4"]

    def test_without_tasks_added_in_v2_only_all_tasks_are_shown(self) -> None:
        tasks = [search_task(task_id, Category.SINGLE_SEARCH) for task_id in TASKS]
        found = build_report("dev", [run("a1"), run("b1", **CHANGED)], tasks)
        (pair,) = found.changes
        assert ([row.scope for row in pair.rows], pair.periods) == (["All 3 tasks"], None)

    def test_a_setup_after_one_on_a_commit_this_repository_does_not_have_says_so(self, repo: Path) -> None:
        known = git(repo, "rev-parse", "HEAD").strip()
        runs = [run("a1", BEFORE, code=on(UNKNOWN)), run("b1", AFTER, code=on(known))]
        found = build(runs, root=repo)
        assert [setup.differs for setup in found.setups] == [
            None,
            "the commit of a1 is not in this repository, so the two cannot be compared",
        ]

    def test_one_setup_has_nothing_to_compare(self) -> None:
        found = build(three("a", BEFORE))
        assert (len(found.setups), found.changes, found.session_noise) == (1, [], [])

    def test_a_dirty_run_that_is_only_listed_does_not_mark_the_report(self) -> None:
        dirty = run("d1", code=CodeVersion(commit=COMMIT, dirty=True)).model_copy(
            update={
                "state": "dirty",
                "reason": "the checkout had changes that were not committed",
                "results": {},
            }
        )
        assert not build([dirty, *three("a", BEFORE)]).scratch

    def test_runs_from_a_checkout_with_changes_mark_the_report_as_scratch(self) -> None:
        # Even when only an earlier setup has them: the comparison uses its runs.
        found = build(
            [
                *three("a", BEFORE, code=CodeVersion(commit=COMMIT, dirty=True)),
                *three("b", AFTER, **CHANGED),
            ]
        )
        assert found.scratch
        assert not build([*three("a", BEFORE), *three("b", AFTER, **CHANGED)]).scratch


def earlier(prefix: str, found: dict[str, Result], **changes: Any) -> list[ListedRun]:  # noqa: ANN401
    """Three runs on an earlier golden set, started a day before the others unless said otherwise."""
    fields: dict[str, Any] = {
        "hashes": hashes(golden=OLD_GOLDEN, evaluation_rules=OLD_RULES),
        "started_at": STARTED - timedelta(days=1),
    } | changes
    return [run(f"{prefix}{number}", found, state="earlier_golden_set", **fields) for number in (1, 2, 3)]


ON_V1: dict[str, Result] = {V1_TASK: "failed"}


@pytest.fixture
def known_earlier(monkeypatch: pytest.MonkeyPatch) -> None:
    sets = (
        EarlierGoldenSet(tag="golden-old", golden=OLD_GOLDEN, evaluation_rules=OLD_RULES, tasks={}),
        EarlierGoldenSet(tag="golden-mid", golden="6" * 64, evaluation_rules=OLD_RULES, tasks={}),
    )
    monkeypatch.setattr(report, "EARLIER_GOLDEN_SETS", sets)


@pytest.mark.usefixtures("known_earlier")
class TestSessionNoise:
    def test_each_models_last_earlier_setup_is_compared_with_its_first_on_this_golden_set(self) -> None:
        found = build([*earlier("v", ON_V1), *three("a", BEFORE), *three("b", AFTER, **CHANGED)])
        assert setups_of(found) == [
            ("m1", 1, "golden-old", ["v1", "v2", "v3"], None),
            ("m1", 2, None, ["a1", "a2", "a3"], "another golden set"),
            ("m1", 3, None, ["b1", "b2", "b3"], "its retrieval prompt differs from that of a1"),
        ]
        (noise,) = found.session_noise
        assert (noise.model, noise.tag, noise.earlier, noise.baseline, noise.refused) == (
            "m1",
            "golden-old",
            1,
            2,
            None,
        )
        assert (noise.differences, noise.agents) == ([], "same")
        assert noise.change is not None
        assert (noise.change.scope, noise.change.gained, noise.change.lost) == (
            "The 1 tasks of golden-old",
            [V1_TASK],
            [],
        )
        (pair,) = found.changes
        assert (pair.before, pair.after, pair.first) == (2, 3, True)

    def test_every_model_has_its_own_and_the_last_earlier_setup_is_used(self) -> None:
        # m2's earlier runs are in two setups (another Ollama version); its last one is compared.
        m2_old = model_build("m2").model_copy(update={"server_version": "0.39.0"})
        found = build(
            [
                *earlier("v", ON_V1),
                *earlier("w", {V1_TASK: "passed"}, build=m2_old),
                *earlier("x", ON_V1, model="m2"),
                *three("a", BEFORE),
                *three("b", BEFORE, model="m2"),
            ]
        )
        assert [(n.model, n.earlier, n.baseline) for n in found.session_noise] == [("m1", 1, 2), ("m2", 2, 3)]
        m2 = found.session_noise[1]
        assert m2.change is not None
        assert m2.change.gained == [V1_TASK]  # x failed it; w passed it, and is not the last

    def test_the_baseline_is_the_models_first_setup_on_this_golden_set(self) -> None:
        # The first setup on the 50 tasks passed the task, the later one failed it again: the noise is
        # measured against the first, and so is the determinism across the pair.
        searched = [attempt_of(V1_TASK, acting())]
        other = [attempt_of(V1_TASK, acting(cites=False))]
        first = [one.model_copy(update={"attempts": searched}) for one in three("a", BEFORE)]
        failed_again = BEFORE | {V1_TASK: "failed"}
        later = [one.model_copy(update={"attempts": other}) for one in three("b", failed_again, **CHANGED)]
        then = [one.model_copy(update={"attempts": searched}) for one in earlier("v", ON_V1)]
        found = build([*then, *first, *later])
        (noise,) = found.session_noise
        assert noise.change is not None
        assert (noise.baseline, noise.change.gained) == (2, [V1_TASK])
        pair = found.determinism[-1]
        assert pair.label == "setups 1 and 2, on the 1 tasks of golden-old"
        assert (pair.measure.runs, pair.measure.turns, pair.measure.same_scenes) == (6, 1, 1)

    def test_setups_are_numbered_in_the_order_they_ran(self) -> None:
        # A run on the earlier golden set that started after the first on this one comes after it.
        later = earlier("v", ON_V1, started_at=STARTED + timedelta(days=1))
        found = build([*three("a", BEFORE), *later])
        assert [(s.number, s.golden) for s in found.setups] == [(1, None), (2, "golden-old")]
        (noise,) = found.session_noise
        assert (noise.earlier, noise.baseline) == (2, 1)

    def test_a_checkout_with_changes_on_the_earlier_side_only_is_said(self) -> None:
        then = earlier("v", ON_V1, code=CodeVersion(commit=COMMIT, dirty=True))
        (noise,) = build([*then, *three("a", BEFORE)]).session_noise
        assert (noise.differences, noise.agents) == (["a checkout with uncommitted changes"], "unknown")

    def test_a_checkout_with_changes_stays_unknown_whatever_the_commits_say(self, repo: Path) -> None:
        first = git(repo, "rev-parse", "HEAD").strip()
        (repo / "README.md").write_text("words\n", encoding="utf-8")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "docs")
        second = git(repo, "rev-parse", "HEAD").strip()
        then = earlier("v", ON_V1, code=CodeVersion(commit=first, dirty=True))
        (noise,) = build([*then, *three("a", BEFORE, code=on(second))], root=repo).session_noise
        assert noise.agents == "unknown"

    def test_a_refused_pair_has_no_determinism_across_it(self) -> None:
        then = earlier("v", ON_V1, build=model_build("m1", "an-older-build"))
        found = build([*then, *three("a", BEFORE)])
        assert [row.label for row in found.determinism] == ["setup 1", "setup 2"]

    def test_two_earlier_golden_sets_are_two_setups(self) -> None:
        mid = earlier("w", {V1_TASK: "passed"}, hashes=hashes(golden="6" * 64, evaluation_rules=OLD_RULES))
        found = build([*earlier("v", ON_V1), *mid, *three("a", BEFORE)])
        assert [(s.number, s.golden, s.differs) for s in found.setups] == [
            (1, "golden-old", None),
            (2, "golden-mid", "another golden set"),
            (3, None, "another golden set"),
        ]
        (noise,) = found.session_noise
        assert (noise.tag, noise.earlier, noise.baseline) == ("golden-mid", 2, 3)

    def test_what_else_differs_is_listed(self) -> None:
        then = earlier(
            "v",
            ON_V1,
            settings={"turn_deadline_s": 1.0, "max_model_calls": 6, "mcp_url": "http://127.0.0.1:8001/mcp"},
            libraries={"langgraph": "1.0.0"},
            reference_date=date(2025, 12, 14),
            shuffle_seed=7,
            platform="Linux-6-x86_64",
            build=model_build("m1").model_copy(update={"server_version": "0.39.0"}),
        )
        (noise,) = build([*then, *three("a", BEFORE)]).session_noise
        assert noise.differences == [
            "settings: turn_deadline_s",
            "libraries: langgraph",
            "the task order's seed",
            "the reference date",
            "Ollama 0.39.0 and 0.40.1",
            "the platform",
        ]
        assert noise.change is not None

    @pytest.mark.parametrize(
        ("changes", "refused"),
        [
            (
                {"hashes": hashes(golden=OLD_GOLDEN, evaluation_rules=OLD_RULES, retrieval_prompt="0" * 64)},
                "the retrieval prompt differs",
            ),
            (
                {"hashes": hashes(golden=OLD_GOLDEN, evaluation_rules=OLD_RULES, decide_prompt="0" * 64)},
                "the decision prompt differs",
            ),
            (
                {"hashes": hashes(golden=OLD_GOLDEN, evaluation_rules=OLD_RULES, tool_descriptions="0" * 64)},
                "the tool descriptions differ",
            ),
            (
                {"build": model_build("m1").model_copy(update={"options": {"seed": 7}})},
                "the model options differ",
            ),
            ({"build": model_build("m1", "an-older-build")}, "the model's build differs"),
        ],
    )
    def test_another_model_or_other_questions_refuse_the_comparison(
        self, changes: dict[str, Any], refused: str
    ) -> None:
        (noise,) = build([*earlier("v", ON_V1, **changes), *three("a", BEFORE)]).session_noise
        assert (noise.refused, noise.change) == (
            f"{refused}, and the comparison is for the same model, asked the same way",
            None,
        )

    def test_a_model_without_a_setup_on_this_golden_set_yet_is_said(self) -> None:
        found = build([*earlier("v", ON_V1), *three("a", BEFORE, model="m2")])
        assert [(n.model, n.baseline, n.refused) for n in found.session_noise] == [
            ("m1", None, "m1 has no setup on this golden set yet")
        ]

    def test_a_checkout_with_changes_on_either_side_is_said(self) -> None:
        then = earlier("v", ON_V1)
        now = three("a", BEFORE, code=CodeVersion(commit=COMMIT, dirty=True))
        (noise,) = build([*then, *now]).session_noise
        assert (noise.differences, noise.agents) == (["a checkout with uncommitted changes"], "unknown")

    def test_the_code_that_differs_is_counted_and_the_agents_own_named(self, repo: Path) -> None:
        first = git(repo, "rev-parse", "HEAD").strip()
        (repo / "src" / "agentic_earth_observation" / "evals").mkdir(parents=True)
        (repo / "src" / "agentic_earth_observation" / "catalog.py").write_text("x = 1\n", encoding="utf-8")
        (repo / "src" / "agentic_earth_observation" / "evals" / "report.py").write_text(
            "x = 1\n", encoding="utf-8"
        )
        (repo / "results").mkdir()
        (repo / "results" / "run.json").write_text("{}", encoding="utf-8")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "change")
        second = git(repo, "rev-parse", "HEAD").strip()
        then = earlier("v", ON_V1, code=on(first))
        (noise,) = build([*then, *three("a", BEFORE, code=on(second))], root=repo).session_noise
        assert noise.differences == [
            "the code: 2 files outside results/, 1 of them in the code the agents run: "
            "src/agentic_earth_observation/catalog.py"
        ]
        assert noise.agents == "differs"

    def test_code_that_differs_outside_the_agents_does_not_count_as_theirs(self, repo: Path) -> None:
        first = git(repo, "rev-parse", "HEAD").strip()
        (repo / "README.md").write_text("words\n", encoding="utf-8")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "docs")
        second = git(repo, "rev-parse", "HEAD").strip()
        then = earlier("v", ON_V1, code=on(first))
        (noise,) = build([*then, *three("a", BEFORE, code=on(second))], root=repo).session_noise
        assert (noise.differences, noise.agents) == (
            ["the code: 1 file outside results/, 0 of them in the code the agents run"],
            "same",
        )

    def test_a_commit_this_repository_does_not_have_is_said_not_compared(self, repo: Path) -> None:
        known = git(repo, "rev-parse", "HEAD").strip()
        then = earlier("v", ON_V1, code=on(UNKNOWN))
        (noise,) = build([*then, *three("a", BEFORE, code=on(known))], root=repo).session_noise
        assert noise.differences == [
            f"the code: commit {UNKNOWN[:12]} or {known[:12]} is not in this repository"
        ]
        assert noise.agents == "unknown"  # it cannot be known, so it is not said to be the same

    def test_git_that_cannot_compare_the_two_commits_stops_the_report(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = git(repo, "rev-parse", "HEAD").strip()
        (repo / "src.py").write_text("x = 2\n", encoding="utf-8")
        git(repo, "commit", "-q", "-am", "change")
        second = git(repo, "rev-parse", "HEAD").strip()
        real = report._git

        def names_fail(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
            if arguments[:2] == ("diff", "--name-only"):
                return subprocess.CompletedProcess(["git"], 128, "", "fatal: bad object\n")
            return real(root, *arguments)

        monkeypatch.setattr(report, "_git", names_fail)
        then = earlier("v", ON_V1, code=on(first))
        with pytest.raises(
            ReportError, match=f"git cannot compare {first[:12]} and {second[:12]} in repo: fatal"
        ):
            build([*then, *three("a", BEFORE, code=on(second))], root=repo)

    def test_determinism_is_measured_within_each_setup_and_across_the_pair(self) -> None:
        searched = [attempt_of(V1_TASK, acting())]
        then = [one.model_copy(update={"attempts": searched}) for one in earlier("v", ON_V1)]
        now = [one.model_copy(update={"attempts": searched}) for one in three("a", BEFORE)]
        found = build([*then, *now])
        assert [
            (row.label, row.measure.runs, row.measure.turns, row.measure.same_calls)
            for row in found.determinism
        ] == [
            ("setup 1", 3, 1, 1),
            ("setup 2", 3, 1, 1),
            ("setups 1 and 2, on the 1 tasks of golden-old", 6, 1, 1),
        ]


@pytest.mark.usefixtures("known_earlier")
class TestTheMarkdown:
    def text(self) -> str:
        searched = [attempt_of(V1_TASK, acting())]
        runs = [
            *earlier("v", ON_V1),
            *three("a", BEFORE),
            *[one.model_copy(update={"attempts": searched}) for one in three("b", AFTER, **CHANGED)],
        ]
        return render_markdown(build(runs))

    def test_the_setups_are_listed_by_model(self) -> None:
        text = self.text()
        assert "## Between setups" in text
        assert "| m1 | 1 | golden-old | v1, v2, v3 |  |" in text
        assert "| m1 | 3 | this one | b1, b2, b3 | its retrieval prompt differs from that of a1 |" in text

    def test_a_change_is_shown_with_its_gained_and_lost_tasks(self) -> None:
        text = self.text()
        assert "### m1: setup 2, then setup 3" in text
        assert "| All 5 tasks | 2 | 0 | 2 | 1 | 0 | 1 |" in text
        assert "| The 4 added in v2 | 1 | 0 | 2 | 1 | 0 | 1 |" in text
        assert "- Gained: v2-single-b, v2-relative-last-month-a." in text
        assert "- Lost: v2-relative-last-winter-a." in text

    def test_the_periods_are_shown_with_the_floor_of_their_test(self) -> None:
        text = self.text()
        assert "| last-month | 2 | 1 | 2 | failed | passed |" in text
        assert (
            "A period passes when all its tasks pass; one with a task left out is left out (0 periods "
            "here). Sign test over the 2 periods with a result on both sides: 1 gained, 1 lost, p = 1. With "
            "2 periods it cannot go below 0.5, so nothing is called significant on its strength."
        ) in text

    def test_the_floor_of_five_periods_is_printed_exactly(self) -> None:
        ids = [
            f"v2-relative-{period}-a"
            for period in ("last-month", "last-spring", "last-summer", "last-autumn", "last-winter")
        ]
        tasks = [search_task(task_id, Category.ROBUSTNESS) for task_id in ids]
        failed: Result = "failed"
        passed: Result = "passed"
        runs = [run("a1", dict.fromkeys(ids, failed)), run("b1", dict.fromkeys(ids, passed), **CHANGED)]
        text = render_markdown(build_report("dev", runs, tasks))
        assert "5 gained, 0 lost, p = 0.063. With 5 periods it cannot go below 0.0625," in text

    def test_the_session_noise_is_shown_with_what_else_differs(self) -> None:
        text = self.text()
        assert "### Session noise, m1: setup 1 (golden-old) against setup 2" in text
        assert "A difference here is not the effect of a change to them, to the model or to the code" in text
        assert "What else differs: nothing else that the run records name." in text
        assert "| The 1 tasks of golden-old | 0 | 0 | 1 | 0 | 0 | 1 |" in text

    def test_the_session_noise_says_when_the_agents_code_cannot_be_known(self) -> None:
        then = earlier("v", ON_V1)
        now = three("a", BEFORE, code=CodeVersion(commit=COMMIT, dirty=True))
        text = render_markdown(build([*then, *now]))
        said = (
            "Whether the code the agents run is the same cannot be told, so a difference here may also come"
        )
        assert said in text
        assert "A difference here is not the effect" not in text

    def test_the_session_noise_says_when_the_agents_code_differs(self, repo: Path) -> None:
        first = git(repo, "rev-parse", "HEAD").strip()
        (repo / "src" / "agentic_earth_observation").mkdir(parents=True)
        (repo / "src" / "agentic_earth_observation" / "catalog.py").write_text("x = 1\n", encoding="utf-8")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "change")
        second = git(repo, "rev-parse", "HEAD").strip()
        runs = [*earlier("v", ON_V1, code=on(first)), *three("a", BEFORE, code=on(second))]
        text = render_markdown(build(runs, root=repo))
        assert "The code the agents run differs too, so a difference here may also come from it." in text

    def test_determinism_has_a_table(self) -> None:
        text = self.text()
        assert "## Determinism" in text
        assert "| m1 | setup 3 | 3 | 1 | 1 of 1 (100%) | 1 of 1 (100%) |" in text

    def test_a_later_change_says_why_seen_and_new_are_not_shown(self) -> None:
        runs = [*three("a", BEFORE), *three("b", AFTER, **CHANGED), *three("c", BEFORE, **CHANGED_AGAIN)]
        assert "the split into seen and new tasks says nothing and is not shown" in render_markdown(
            build(runs)
        )

    def test_a_refusal_is_said(self) -> None:
        then = earlier(
            "v",
            ON_V1,
            hashes=hashes(golden=OLD_GOLDEN, evaluation_rules=OLD_RULES, tool_descriptions="0" * 64),
        )
        text = render_markdown(build([*then, *three("a", BEFORE)]))
        said = "Not compared: the tool descriptions differ, and the comparison is for the same model, asked "
        assert f"{said}the same way." in text

    def test_the_sentence_on_the_numbers_above_needs_numbers_above(self) -> None:
        said = "The numbers above count the runs marked counted in the table of every run"
        assert said in self.text()
        assert said not in render_markdown(build(earlier("v", ON_V1)))

    def test_a_report_with_one_setup_has_no_setup_section(self) -> None:
        assert "## Between setups" not in render_markdown(build(three("a", BEFORE)))


def test_the_report_command_reads_the_runs_of_v1_0_as_an_earlier_golden_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The command must give the report the split's tasks, or no earlier golden set is read.
    out = tmp_path / "report.md"
    assert (
        cli.main(
            ["report", "--split", "test", "--runs-dir", str(ROOT / "results" / "runs"), "--output", str(out)]
        )
        == 0
    )
    capsys.readouterr()
    text = out.read_text(encoding="utf-8")
    # Plain numbers only: a failing assert must not print the report, which names test tasks.
    read = text.count("| earlier golden set |")
    has_determinism = "## Determinism" in text
    assert read == 6
    assert has_determinism
