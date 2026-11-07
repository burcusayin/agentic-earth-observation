"""Tests for the report: which runs count, and what each task came to in them. Run folders are written with
the runner's own writers into a temporary folder, so the report reads exactly what a run leaves behind."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from agentic_earth_observation.evals import report
from agentic_earth_observation.evals.records import (
    AttemptOutcome,
    AttemptRecord,
    CodeVersion,
    Hashes,
    RunRecord,
    TaskCounts,
    append_attempt,
    read_attempts,
    read_run_record,
    write_run_record,
)
from agentic_earth_observation.evals.report import (
    ListedRun,
    ReportError,
    check_same_code,
    collect_runs,
    task_results,
)
from agentic_earth_observation.evals.run_eval import (
    FROZEN_EVALUATION_CODE,
    FROZEN_EVALUATION_RULES,
    FROZEN_GAZETTEER,
)
from agentic_earth_observation.evals.scoring import TurnScore
from tests.test_eval_records import ALL_RIGHT, PASSED, STARTED, git, run_record, turn

GOLDEN = "9" * 64  # the golden set the report is made from: any hash, the same in the records
COMMIT = "0123456789abcdef0123456789abcdef01234567"
TASKS = ["dev-a", "dev-b", "dev-c"]
FAILED = TurnScore(
    route=True, tools=True, arguments=ALL_RIGHT, first_arguments=ALL_RIGHT, grounded=True, outcome=False
)


def hashes(**changes: str) -> Hashes:
    fields = {
        "retrieval_prompt": "1" * 64,
        "decide_prompt": "2" * 64,
        "tool_descriptions": "3" * 64,
        "golden": GOLDEN,
        "evaluation_code": FROZEN_EVALUATION_CODE,
        "gazetteer": FROZEN_GAZETTEER,
        "evaluation_rules": FROZEN_EVALUATION_RULES,
    }
    return Hashes.model_validate(fields | changes)


def record(run_id: str, **changes: Any) -> RunRecord:  # noqa: ANN401 - any field of the record
    fields: dict[str, Any] = {
        "run_id": run_id,
        "status": "complete",
        "split": "dev",
        "task_order": TASKS,
        "hashes": hashes(),
        "code": CodeVersion(commit=COMMIT, dirty=False),
        "tasks": TaskCounts(passed=3),
    }
    fields |= changes
    if fields["status"] != "running":  # the record requires an end time once the run ended
        fields.setdefault("finished_at", STARTED + timedelta(hours=1))
    return run_record(**fields)


def attempt(
    run_id: str, task_id: str, number: int, outcome: AttemptOutcome, passed: bool | None
) -> AttemptRecord:
    if outcome == "scored":
        turns = [turn(score=PASSED if passed else FAILED)]
        reason = None
    else:
        turns = [turn(score=None, oracle=None)]
        reason = "turn 1: A2A ReadTimeout" if outcome == "infrastructure" else "turn 1 raised ValueError: bug"
    return AttemptRecord(
        run_id=run_id,
        task_id=task_id,
        attempt=number,
        outcome=outcome,
        passed=passed,
        reason=reason,
        turns=turns,
    )


Ending = tuple[AttemptOutcome, bool | None]  # an attempt's outcome, and whether its task passed
PASS: Ending = ("scored", True)
FAIL: Ending = ("scored", False)
ERROR: Ending = ("errored", False)
INFRA: Ending = ("infrastructure", None)


# task id -> its attempts, each as (outcome, passed), in order
Attempts = Mapping[str, Sequence[Ending]]


def write_run(
    runs: Path,
    run_id: str,
    per_task: Attempts,
    **changes: Any,  # noqa: ANN401 - any field of the record
) -> Path:
    """A run's folder: its record, and for each task its attempts in order."""
    folder = runs / run_id
    write_run_record(folder, record(run_id, **changes))
    for task_id, attempts in per_task.items():
        for number, (outcome, passed) in enumerate(attempts, start=1):
            append_attempt(folder, attempt(run_id, task_id, number, outcome, passed))
    return folder


ALL_PASS = {task: [PASS] for task in TASKS}


def collect(runs: Path, **options: Any) -> list[ListedRun]:  # noqa: ANN401 - collect_runs' options
    return collect_runs(runs, "dev", TASKS, golden=GOLDEN, **options)


def states(listed: Sequence[ListedRun]) -> list[tuple[str, str]]:
    return [(run.folder, run.state) for run in listed]


class TestWhichRunsCount:
    """Every run is listed, with why it is not in the numbers if not (docs/decisions.md 2026-11-05)."""

    def test_a_complete_clean_consistent_run_is_reported_with_its_results(self, tmp_path: Path) -> None:
        write_run(tmp_path, "r1", ALL_PASS)
        (run,) = collect(tmp_path)
        assert (run.folder, run.state, run.reason) == ("r1", "reported", None)
        assert run.results == {"dev-a": "passed", "dev-b": "passed", "dev-c": "passed"}
        assert run.record is not None

    def test_a_folder_without_a_record_did_not_start(self, tmp_path: Path) -> None:
        (tmp_path / "r1").mkdir()
        (run,) = collect(tmp_path)
        assert (run.state, run.record, run.results) == ("did_not_start", None, {})

    def test_an_unreadable_record_is_listed_without_this_machines_paths(self, tmp_path: Path) -> None:
        folder = tmp_path / "r1"
        folder.mkdir()
        (folder / "run.json").write_text('{"run_id": "r1"}', encoding="utf-8")
        (run,) = collect(tmp_path)
        assert run.state == "unreadable"
        assert run.reason is not None
        assert run.reason.startswith("run.json is not a run record this version reads:")
        assert str(tmp_path) not in run.reason

    @pytest.mark.parametrize(
        ("content", "listed"),
        [
            ('{"run_id": "r1", "split": "test"}', False),  # names the other split: not this report's
            ('{"run_id": "r1", "split": "dev"}', True),
            ('{"run_id": "r1"}', True),  # names no split: it may be this report's
            ("not json", True),
            ('["a", "list"]', True),
        ],
    )
    def test_an_unreadable_record_of_the_other_split_is_not_listed(
        self, tmp_path: Path, content: str, listed: bool
    ) -> None:
        folder = tmp_path / "r1"
        folder.mkdir()
        (folder / "run.json").write_text(content, encoding="utf-8")
        assert states(collect(tmp_path)) == ([("r1", "unreadable")] if listed else [])

    def test_a_record_that_cannot_be_read_as_a_file_says_so(self, tmp_path: Path) -> None:
        (tmp_path / "r1" / "run.json").mkdir(parents=True)
        (run,) = collect(tmp_path)
        assert (run.state, run.reason) == ("unreadable", "run.json cannot be read (Is a directory)")

    def test_attempts_that_cannot_be_read_are_listed_without_local_paths(self, tmp_path: Path) -> None:
        folder = write_run(tmp_path, "r1", ALL_PASS)
        with (folder / "attempts.jsonl").open("a", encoding="utf-8") as out:
            out.write('{"not": "an attempt"}\n')
        (run,) = collect(tmp_path)
        assert run.state == "unreadable"
        assert run.reason is not None
        assert run.reason.startswith(
            "attempts.jsonl cannot be read: r1/attempts.jsonl:4: not an attempt record"
        )
        assert str(tmp_path) not in run.reason

    def test_a_run_still_marked_running_crashed_or_is_still_going(self, tmp_path: Path) -> None:
        write_run(tmp_path, "r1", {}, status="running", tasks=TaskCounts())
        (run,) = collect(tmp_path)
        assert (run.state, run.reason) == ("running", "it crashed, or is still running")

    def test_an_aborted_run_is_listed_with_its_reason_and_counts(self, tmp_path: Path) -> None:
        reason = "3 tasks errored"
        write_run(tmp_path, "r1", {}, status="aborted", aborted_reason=reason, tasks=TaskCounts(errored=3))
        (run,) = collect(tmp_path)
        assert (run.state, run.reason) == ("aborted", reason)
        assert run.record is not None
        assert run.record.tasks == TaskCounts(errored=3)

    def test_a_run_from_a_dirty_checkout_is_listed_not_reported(self, tmp_path: Path) -> None:
        write_run(tmp_path, "r1", ALL_PASS, code=CodeVersion(commit=COMMIT, dirty=True))
        (run,) = collect(tmp_path)
        assert (run.state, run.reason) == ("dirty", "the checkout had changes that were not committed")

    def test_a_dirty_run_is_reported_when_scratch_runs_are_allowed(self, tmp_path: Path) -> None:
        write_run(tmp_path, "r1", ALL_PASS, code=CodeVersion(commit=COMMIT, dirty=True))
        (run,) = collect(tmp_path, allow_dirty=True)
        assert run.state == "reported"

    @pytest.mark.parametrize(
        ("field", "named"),
        [
            ("golden", "the golden set"),
            ("evaluation_code", "the evaluation code"),
            ("gazetteer", "the gazetteer"),
            ("evaluation_rules", "the labelling rules"),
        ],
    )
    def test_a_run_on_other_frozen_files_is_listed_not_reported(
        self, tmp_path: Path, field: str, named: str
    ) -> None:
        write_run(tmp_path, "r1", ALL_PASS, hashes=hashes(**{field: "0" * 64}))
        (run,) = collect(tmp_path)
        assert run.state == "other_golden_set"
        assert run.reason == f"{named} differs from what this report reads"

    @pytest.mark.parametrize(
        "order",
        [["dev-a", "dev-b"], ["dev-a", "dev-b", "dev-c", "dev-d"], ["dev-a", "dev-b", "dev-x"]],
        ids=["a task missing", "a task more", "another task"],
    )
    def test_a_run_over_other_tasks_is_listed_not_reported(self, tmp_path: Path, order: list[str]) -> None:
        write_run(tmp_path, "r1", {task: [PASS] for task in order}, task_order=order)
        (run,) = collect(tmp_path)
        assert (run.state, run.reason) == ("other_golden_set", "its tasks are not the tasks of the split")

    def test_a_run_whose_record_and_attempts_disagree_is_inconsistent(self, tmp_path: Path) -> None:
        write_run(tmp_path, "r1", ALL_PASS, tasks=TaskCounts(passed=2, failed=1))
        (run,) = collect(tmp_path)
        assert run.state == "inconsistent"
        assert run.reason == (
            "run.json counts 2 passed, 1 failed, 0 errored, 0 lost; "
            "attempts.jsonl gives 3 passed, 0 failed, 0 errored, 0 lost"
        )

    def test_a_run_whose_attempts_do_not_add_up_is_inconsistent(self, tmp_path: Path) -> None:
        write_run(tmp_path, "r1", {"dev-a": [PASS], "dev-b": [PASS]}, tasks=TaskCounts(passed=2))
        (run,) = collect(tmp_path)
        assert (run.state, run.reason, run.results) == ("inconsistent", "dev-c has no attempt", {})

    def test_a_run_of_the_other_split_is_not_listed(self, tmp_path: Path) -> None:
        write_run(tmp_path, "r1", ALL_PASS, split="test")
        assert collect(tmp_path) == []

    def test_runs_are_listed_in_the_order_they_started_and_files_are_ignored(self, tmp_path: Path) -> None:
        write_run(tmp_path, "20261107T200000Z-m-r2", ALL_PASS)
        write_run(tmp_path, "20261107T190000Z-m-r1", ALL_PASS)
        (tmp_path / "tool_calls.jsonl").write_text("", encoding="utf-8")  # left by `make serve-mcp`
        assert states(collect(tmp_path)) == [
            ("20261107T190000Z-m-r1", "reported"),
            ("20261107T200000Z-m-r2", "reported"),
        ]

    def test_no_folder_of_runs_is_no_run(self, tmp_path: Path) -> None:
        assert collect(tmp_path / "runs") == []

    def test_a_file_where_the_folder_of_runs_should_be_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "runs").write_text("", encoding="utf-8")
        with pytest.raises(ReportError, match="runs is not a folder"):
            collect(tmp_path / "runs")


class TestTaskResults:
    """A task's result is its last attempt's: passed, failed, errored, or lost after every attempt."""

    def results(
        self,
        tmp_path: Path,
        per_task: Attempts,
        **changes: Any,  # noqa: ANN401 - fields of the record
    ) -> dict[str, str]:
        folder = write_run(tmp_path, "r1", per_task, **changes)
        return dict(task_results(read_run_record(folder), read_attempts(folder)))

    def test_each_kind_of_ending(self, tmp_path: Path) -> None:
        found = self.results(
            tmp_path,
            {"dev-a": [INFRA, PASS], "dev-b": [ERROR], "dev-c": [INFRA, INFRA, INFRA]},
            tasks=TaskCounts(passed=1, errored=1, lost=1),
        )
        assert found == {"dev-a": "passed", "dev-b": "errored", "dev-c": "lost"}

    def test_a_failed_task(self, tmp_path: Path) -> None:
        found = self.results(
            tmp_path,
            {"dev-a": [FAIL], "dev-b": [PASS], "dev-c": [PASS]},
            tasks=TaskCounts(passed=2, failed=1),
        )
        assert found["dev-a"] == "failed"

    @pytest.mark.parametrize(
        ("per_task", "said"),
        [
            ({"dev-a": [PASS], "dev-b": [PASS]}, "dev-c has no attempt"),
            ({**ALL_PASS, "dev-z": [PASS]}, "an attempt of dev-z, which is not one of the run's tasks"),
            ({"dev-a": [INFRA], "dev-b": [PASS], "dev-c": [PASS]}, "dev-a ended after 1 of 3 attempts"),
            (
                {"dev-a": [PASS, PASS], "dev-b": [PASS], "dev-c": [PASS]},
                "dev-a has an attempt after its last",
            ),
        ],
        ids=["a task without attempts", "an unknown task", "a task cut short", "an attempt too many"],
    )
    def test_a_record_that_does_not_add_up_is_refused(
        self, tmp_path: Path, per_task: Attempts, said: str
    ) -> None:
        with pytest.raises(ValueError, match=said):
            self.results(tmp_path, per_task)

    def test_attempts_must_be_numbered_in_order(self, tmp_path: Path) -> None:
        folder = write_run(tmp_path, "r1", {"dev-b": [PASS], "dev-c": [PASS]})
        append_attempt(folder, attempt("r1", "dev-a", 2, *PASS))
        with pytest.raises(ValueError, match="dev-a: attempt 2 where attempt 1 was due"):
            task_results(read_run_record(folder), read_attempts(folder))

    def test_an_attempt_of_another_run_is_refused(self, tmp_path: Path) -> None:
        folder = write_run(tmp_path, "r1", {"dev-b": [PASS], "dev-c": [PASS]})
        append_attempt(folder, attempt("r0", "dev-a", 1, *PASS))
        with pytest.raises(ValueError, match="an attempt of run r0"):
            task_results(read_run_record(folder), read_attempts(folder))


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


def reported(run_id: str, commit: str, **hash_changes: str) -> ListedRun:
    return ListedRun(
        folder=run_id,
        state="reported",
        record=record(run_id, code=CodeVersion(commit=commit, dirty=False), hashes=hashes(**hash_changes)),
        results=dict.fromkeys(TASKS, "passed"),
    )


class TestSameCode:
    """Runs compared with each other must come from the same code: commits may differ only in results/."""

    def test_commits_that_differ_only_in_the_run_records_are_the_same_code(self, repo: Path) -> None:
        first = git(repo, "rev-parse", "HEAD").strip()
        (repo / "results" / "runs").mkdir(parents=True)
        (repo / "results" / "runs" / "run.json").write_text("{}", encoding="utf-8")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "records")
        check_same_code([reported("r1", first), reported("r2", git(repo, "rev-parse", "HEAD").strip())], repo)

    def test_commits_that_differ_in_code_are_refused(self, repo: Path) -> None:
        first = git(repo, "rev-parse", "HEAD").strip()
        (repo / "src.py").write_text("x = 2\n", encoding="utf-8")
        git(repo, "commit", "-q", "-am", "change")
        second = git(repo, "rev-parse", "HEAD").strip()
        with pytest.raises(
            ReportError, match=f"r2 ran on {second[:12]}, whose code differs from {first[:12]}"
        ):
            check_same_code([reported("r1", first), reported("r2", second)], repo)

    def test_a_commit_this_repository_does_not_have_is_refused(self, repo: Path) -> None:
        first = git(repo, "rev-parse", "HEAD").strip()
        with pytest.raises(ReportError, match="r2 ran on fedcba987654, which this repository does not have"):
            check_same_code(
                [reported("r1", first), reported("r2", "fedcba9876543210" * 2 + "01234567")], repo
            )

    @pytest.mark.parametrize("field", ["retrieval_prompt", "decide_prompt", "tool_descriptions"])
    def test_runs_on_other_prompts_or_tools_are_refused(self, repo: Path, field: str) -> None:
        first = git(repo, "rev-parse", "HEAD").strip()
        with pytest.raises(ReportError, match=f"r2 differs from r1 in {field}"):
            check_same_code([reported("r1", first), reported("r2", first, **{field: "0" * 64})], repo)

    def test_runs_on_one_commit_need_no_git_comparison(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_git(root: Path, *arguments: str) -> None:
            raise AssertionError(f"git {arguments} was not needed")

        monkeypatch.setattr(report, "_git", no_git)
        first = "fedcba9876543210" * 2 + "01234567"
        check_same_code([reported("r1", first), reported("r2", first), reported("r3", first)], repo)

    def test_without_git_the_report_says_so(self, repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(ReportError, match="git is not installed"):
            check_same_code([reported("r1", "a" * 40), reported("r2", "b" * 40)], repo)

    @pytest.mark.parametrize("trouble", [OSError("no"), subprocess.TimeoutExpired("git", 30)])
    def test_git_that_cannot_run_is_reported(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch, trouble: Exception
    ) -> None:
        def broken(*args: object, **kwargs: object) -> None:
            raise trouble

        monkeypatch.setattr(subprocess, "run", broken)
        said = f"git cat-file could not run in repo: {type(trouble).__name__}"
        with pytest.raises(ReportError, match=said):
            check_same_code([reported("r1", "a" * 40), reported("r2", "b" * 40)], repo)

    def test_a_comparison_git_cannot_make_is_reported_with_its_words(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = report._git

        def diff_fails(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
            if arguments[0] == "diff":
                return subprocess.CompletedProcess(["git"], 128, "", "fatal: bad object\n")
            return real(root, *arguments)

        monkeypatch.setattr(report, "_git", diff_fails)
        first = git(repo, "rev-parse", "HEAD").strip()
        (repo / "src.py").write_text("x = 3\n", encoding="utf-8")
        git(repo, "commit", "-q", "-am", "change")
        second = git(repo, "rev-parse", "HEAD").strip()
        said = f"git cannot compare {first[:12]} and {second[:12]} in repo: fatal: bad object"
        with pytest.raises(ReportError, match=said):
            check_same_code([reported("r1", first), reported("r2", second)], repo)

    def test_one_run_or_none_needs_no_comparison(self, repo: Path) -> None:
        check_same_code([], repo)
        check_same_code([reported("r1", "fedcba9876543210" * 2 + "01234567")], repo)
