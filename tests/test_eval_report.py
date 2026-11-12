"""Tests for the report: which runs count, and what each task came to in them. Run folders are written with
the runner's own writers into a temporary folder, so the report reads exactly what a run leaves behind."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any, get_args

import pytest

from agentic_earth_observation.domain import Product
from agentic_earth_observation.evals import report
from agentic_earth_observation.evals.golden import Category, Expectation, ExpectedSearch, Route, Task, Turn
from agentic_earth_observation.evals.records import (
    AttemptOutcome,
    AttemptRecord,
    CodeVersion,
    Hashes,
    ModelBuild,
    OracleCheck,
    RunRecord,
    TaskCounts,
    TurnRecord,
    append_attempt,
    read_attempts,
    read_run_record,
    write_run_record,
)
from agentic_earth_observation.evals.report import (
    CategoryShare,
    Comparison,
    ListedRun,
    Mean,
    ModelSummary,
    Rate,
    Report,
    ReportError,
    RunSeconds,
    build_report,
    collect_runs,
    mark_comparable,
    render_markdown,
    task_results,
)
from agentic_earth_observation.evals.run_eval import (
    FROZEN_EVALUATION_CODE,
    FROZEN_EVALUATION_RULES,
    FROZEN_GAZETTEER,
)
from agentic_earth_observation.evals.scoring import ArgumentScore, TurnScore
from agentic_earth_observation.orchestrator.models import TurnOutput
from agentic_earth_observation.retrieval_agent.models import ModelFailure, Usage
from tests.fakes import SCENE
from tests.test_eval_records import ALL_RIGHT, PASSED, STARTED, git, run_record, turn

GOLDEN = "9" * 64  # the golden set the report is made from: any hash, the same in the records
COMMIT = "0123456789abcdef0123456789abcdef01234567"
TASKS = ["dev-a", "dev-b", "dev-c"]
ONE_TURN = dict.fromkeys(TASKS, 1)  # each task and its number of turns
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


def model_build(model: str, digest: str = "digest-of-the-model") -> ModelBuild:
    return ModelBuild(
        provider="ollama", model=model, digest=digest, server_version="0.40.1", options={"seed": 42}
    )


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
    return collect_runs(runs, "dev", ONE_TURN, golden=GOLDEN, **options)


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
        assert run.reason.startswith("run.json is not a run record this version reads: ")
        assert "problem(s)" not in run.reason
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

    def test_attempts_that_cannot_be_read_are_listed_by_line_without_their_text(self, tmp_path: Path) -> None:
        # The line could quote a task or an answer: the reason names where, never what.
        folder = write_run(tmp_path, "r1", ALL_PASS)
        with (folder / "attempts.jsonl").open("a", encoding="utf-8") as out:
            out.write('{"run_id": "r1", "answer": "SECRETPLACE"}\n')
        (run,) = collect(tmp_path)
        assert run.state == "unreadable"
        assert run.reason is not None
        assert run.reason.startswith("line 4 of attempts.jsonl is not an attempt record this version reads: ")
        assert "SECRETPLACE" not in run.reason
        assert str(tmp_path) not in run.reason
        assert len(run.reason) < 200

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
            ("evaluation_rules", "docs/evaluation.md"),
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
        return dict(task_results(read_run_record(folder), read_attempts(folder), ONE_TURN))

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
            task_results(read_run_record(folder), read_attempts(folder), ONE_TURN)

    def test_an_attempt_of_another_run_is_refused(self, tmp_path: Path) -> None:
        folder = write_run(tmp_path, "r1", {"dev-b": [PASS], "dev-c": [PASS]})
        append_attempt(folder, attempt("r0", "dev-a", 1, *PASS))
        with pytest.raises(ValueError, match="an attempt of run r0"):
            task_results(read_run_record(folder), read_attempts(folder), ONE_TURN)


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


def reported(run_id: str, commit: str, **changes: Any) -> ListedRun:  # noqa: ANN401 - fields of the record
    return ListedRun(
        folder=run_id,
        state="reported",
        record=record(run_id, **({"code": CodeVersion(commit=commit, dirty=False)} | changes)),
        results=dict.fromkeys(TASKS, "passed"),
    )


def marked(runs: Sequence[ListedRun], root: Path) -> list[tuple[str, str, str | None]]:
    return [(run.folder, run.state, run.reason) for run in mark_comparable(runs, root)]


class TestComparableSetup:
    """The latest counted run sets the setup; an earlier run on another one is listed, not counted, with what
    differs, so that after a fix both are listed (decision of 2026-11-05)."""

    def head(self, repo: Path) -> str:
        return git(repo, "rev-parse", "HEAD").strip()

    def test_commits_that_differ_only_in_the_run_records_are_one_setup(self, repo: Path) -> None:
        first = self.head(repo)
        (repo / "results" / "runs").mkdir(parents=True)
        (repo / "results" / "runs" / "run.json").write_text("{}", encoding="utf-8")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "records")
        runs = [reported("r1", first), reported("r2", self.head(repo))]
        assert marked(runs, repo) == [("r1", "reported", None), ("r2", "reported", None)]

    def test_a_run_on_other_code_is_listed_with_what_differs(self, repo: Path) -> None:
        first = self.head(repo)
        (repo / "src.py").write_text("x = 2\n", encoding="utf-8")
        git(repo, "commit", "-q", "-am", "change")
        found = mark_comparable([reported("r1", first), reported("r2", self.head(repo))], repo)
        assert [(run.state, run.reason) for run in found] == [
            ("other_setup", "its code differs from that of r2 outside results/"),
            ("reported", None),
        ]
        assert (found[0].results, found[0].attempts) == ({}, [])  # nothing of it reaches the numbers

    @pytest.mark.parametrize(
        ("changes", "said"),
        [
            ({"hashes": hashes(retrieval_prompt="0" * 64)}, "its retrieval prompt differs from that of r2"),
            ({"hashes": hashes(tool_descriptions="0" * 64)}, "its tool descriptions differ from those of r2"),
            (
                {
                    "settings": {
                        "turn_deadline_s": 1.0,
                        "max_model_calls": 6,
                        "mcp_url": "http://127.0.0.1:8001/mcp",
                    }
                },
                "its settings differ from those of r2: turn_deadline_s",
            ),
            (
                {"model": model_build("qwen3.5:9b").model_copy(update={"options": {"seed": 7}})},
                "its model options differ from those of r2: seed",
            ),
            ({"shuffle_seed": 99}, "its task order's seed differs from that of r2"),
            ({"reference_date": date(2025, 12, 14)}, "its reference date differs from that of r2"),
            (
                {"model": model_build("qwen3.5:9b").model_copy(update={"server_version": "0.41.0"})},
                "it ran on another Ollama version than r2",
            ),
        ],
    )
    def test_a_run_on_another_setup_is_listed_with_what_differs(
        self, repo: Path, changes: dict[str, Any], said: str
    ) -> None:
        commit = self.head(repo)
        latest = reported("r2", commit, model=model_build("qwen3.5:9b"))
        earlier = reported("r1", commit, **({"model": model_build("qwen3.5:9b")} | changes))
        assert marked([earlier, latest], repo) == [("r1", "other_setup", said), ("r2", "reported", None)]

    @pytest.mark.parametrize(
        ("earlier_dirty", "said"),
        [
            (False, "it ran from a clean checkout, r2 from one with uncommitted changes"),
            (True, "it ran from a checkout with uncommitted changes, r2 from a clean one"),
        ],
    )
    def test_a_run_from_a_checkout_with_changes_and_one_without_are_two_setups(
        self, repo: Path, earlier_dirty: bool, said: str
    ) -> None:
        commit = self.head(repo)
        earlier = reported("r1", commit, code=CodeVersion(commit=commit, dirty=earlier_dirty))
        latest = reported("r2", commit, code=CodeVersion(commit=commit, dirty=not earlier_dirty))
        assert marked([earlier, latest], repo)[0] == ("r1", "other_setup", said)

    def test_another_build_of_the_same_model_is_another_setup_but_another_model_is_not(
        self, repo: Path
    ) -> None:
        commit = self.head(repo)
        runs = [
            reported("r1", commit, model=model_build("m1", "d1")),
            reported("r2", commit, model=model_build("m1", "d9")),
            reported("r3", commit, model=model_build("m2", "d2")),
        ]
        assert marked(runs, repo) == [
            ("r1", "other_setup", "it ran another build of m1 than r2"),
            ("r2", "reported", None),
            ("r3", "reported", None),
        ]

    def test_an_earlier_run_on_a_commit_this_repository_does_not_have_is_listed(self, repo: Path) -> None:
        unknown = "fedcba9876543210" * 2 + "01234567"
        assert marked([reported("r1", unknown), reported("r2", self.head(repo))], repo)[0] == (
            "r1",
            "other_setup",
            "its commit fedcba987654 is not in this repository",
        )

    def test_the_latest_run_on_a_commit_this_repository_does_not_have_is_refused(self, repo: Path) -> None:
        unknown = "fedcba9876543210" * 2 + "01234567"
        with pytest.raises(ReportError, match="r2 ran on fedcba987654, which this repository does not have"):
            mark_comparable([reported("r1", self.head(repo)), reported("r2", unknown)], repo)

    def test_runs_that_do_not_count_are_left_as_they_are(self, repo: Path) -> None:
        aborted = ListedRun(folder="r0", state="aborted", reason="3 tasks errored")
        runs = [aborted, reported("r1", self.head(repo)), reported("r2", self.head(repo), shuffle_seed=5)]
        assert marked(runs, repo)[0] == ("r0", "aborted", "3 tasks errored")

    def test_runs_on_one_commit_need_no_git_comparison(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        commit = self.head(repo)
        real = report._git

        def only_the_commit_check(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
            if arguments[0] != "cat-file":
                raise AssertionError(f"git {arguments} was not needed")
            return real(root, *arguments)

        monkeypatch.setattr(report, "_git", only_the_commit_check)
        runs = [reported("r1", commit), reported("r2", commit), reported("r3", commit)]
        assert [state for _, state, _ in marked(runs, repo)] == ["reported"] * 3

    def test_without_git_the_report_says_so(self, repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(ReportError, match="git is not installed"):
            mark_comparable([reported("r1", "a" * 40), reported("r2", "b" * 40)], repo)

    @pytest.mark.parametrize("trouble", [OSError("no"), subprocess.TimeoutExpired("git", 30)])
    def test_git_that_cannot_run_is_reported(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch, trouble: Exception
    ) -> None:
        def broken(*args: object, **kwargs: object) -> None:
            raise trouble

        monkeypatch.setattr(subprocess, "run", broken)
        with pytest.raises(
            ReportError, match=f"git cat-file could not run in repo: {type(trouble).__name__}"
        ):
            mark_comparable([reported("r1", "a" * 40), reported("r2", "b" * 40)], repo)

    def test_a_comparison_git_cannot_make_is_reported_with_its_words(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = report._git

        def diff_fails(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
            if arguments[0] == "diff":
                return subprocess.CompletedProcess(["git"], 128, "", "fatal: bad object\n")
            return real(root, *arguments)

        monkeypatch.setattr(report, "_git", diff_fails)
        first = self.head(repo)
        (repo / "src.py").write_text("x = 3\n", encoding="utf-8")
        git(repo, "commit", "-q", "-am", "change")
        second = self.head(repo)
        said = f"git cannot compare {first[:12]} and {second[:12]} in repo: fatal: bad object"
        with pytest.raises(ReportError, match=said):
            mark_comparable([reported("r1", first), reported("r2", second)], repo)

    def test_one_run_or_none_needs_no_comparison(self, repo: Path) -> None:
        unknown = "fedcba9876543210" * 2 + "01234567"
        assert mark_comparable([], repo) == []
        assert marked([reported("r1", unknown)], repo) == [("r1", "reported", None)]


# ---------------------------------------------------------------------------------------------------------
# The numbers: two models, three runs each, over three made-up tasks. Every expected value below is worked
# out by hand from this table (P passed, F failed, E errored, L lost after three attempts):
#
#   task             m1: run 1  run 2       run 3    m2: run 1  run 2  run 3
#   dev-a (search)       P*     I then P    P            F#     F#     F#
#   dev-b (search)       F~     F~          P            P      P      P
#   dev-c (decline)      P      L           E            P      P      P
#
#   * first search call wrong, last right (self-corrected); ~ last call wrong (dates); # declined a task that
#   expected a search, with a parse error (a model failure). I: one attempt lost to the infrastructure.
# ---------------------------------------------------------------------------------------------------------

WRONG_DATES = ALL_RIGHT.model_copy(update={"dates": False})
NO_SEARCH = ALL_RIGHT.model_copy(
    update={
        "product": False,
        "bbox": False,
        "dates": False,
        "cloud": False,
        "limit": False,
        "no_extra": False,
    }
)


def search_task(task_id: str, category: Category) -> Task:
    search = ExpectedSearch(
        product=Product.SENTINEL_2_L2A,
        place="Milan",
        start=date(2025, 3, 1),
        end=date(2025, 3, 31),
        cloud=(0, 20),
    )
    expect = Expectation(
        route=Route.DELEGATE, required_tools=["resolve_place", "search_scenes"], search=search
    )
    return Task(id=task_id, split="dev", category=category, turns=[Turn(user="(a question)", expect=expect)])


def follow_up_task(task_id: str) -> Task:
    """A two-turn task: a search, then a search that changes one thing."""
    (first,) = search_task(task_id, Category.SINGLE_SEARCH).turns
    return Task(id=task_id, split="dev", category=Category.FOLLOW_UP, turns=[first, first])


def decline_task(task_id: str) -> Task:
    expect = Expectation(route=Route.DECLINE, forbidden_tools=["search_scenes"])
    return Task(
        id=task_id,
        split="dev",
        category=Category.OUT_OF_SCOPE,
        turns=[Turn(user="(a question)", expect=expect)],
    )


GOLDEN_TASKS = [
    search_task("dev-a", Category.SINGLE_SEARCH),
    search_task("dev-b", Category.ROBUSTNESS),
    decline_task("dev-c"),
]


def declined(failure: ModelFailure | None = None) -> TurnOutput:
    return TurnOutput(
        trace_id="0a1b2c3d-t01-a1-u1",
        route=Route.DECLINE,
        status="declined",
        answer="(the reply)",
        failure=failure,
    )


def scored_turn(
    *,
    seconds: float,
    passed: bool = True,
    arguments: ArgumentScore | None = ALL_RIGHT,
    first: ArgumentScore | None = None,
    precision: float | None = 1.0,
    recall: float | None = None,
    output: TurnOutput | None = None,
) -> TurnRecord:
    score = TurnScore(
        route=passed or arguments is not NO_SEARCH,
        tools=True,
        arguments=arguments,
        first_arguments=first if first is not None else arguments,
        grounded=True,
        outcome=passed,
    )
    oracle = (
        OracleCheck(ids=[SCENE.id], precision=precision, recall=recall) if precision is not None else None
    )
    changes: dict[str, Any] = {"seconds": seconds, "score": score, "oracle": oracle}
    if output is not None:
        changes |= {"output": output, "calls": []}
    return turn(**changes)


def ended(
    run_id: str,
    task_id: str,
    number: int,
    outcome: AttemptOutcome,
    passed: bool | None,
    the_turn: TurnRecord | None = None,
) -> AttemptRecord:
    if outcome == "scored":
        assert the_turn is not None
        assert the_turn.score is not None
        assert the_turn.score.passed is passed, "the scenario's turn must agree"
        return AttemptRecord(
            run_id=run_id, task_id=task_id, attempt=number, outcome="scored", passed=passed, turns=[the_turn]
        )
    return attempt(run_id, task_id, number, outcome, passed)


def write_attempts(runs: Path, run_id: str, attempts: Sequence[AttemptRecord], **changes: Any) -> None:  # noqa: ANN401 - any field of the record
    folder = runs / run_id
    write_run_record(folder, record(run_id, **changes))
    for one in attempts:
        append_attempt(folder, one)


def write_scenario(runs: Path, *, m1_digests: Sequence[str] = ("d1", "d1", "d1")) -> None:
    a_first = scored_turn(seconds=10, first=WRONG_DATES, recall=0.8)
    m1 = {
        "a1": [
            ended("a1", "dev-a", 1, "scored", True, a_first),
            ended(
                "a1",
                "dev-b",
                1,
                "scored",
                False,
                scored_turn(seconds=20, passed=False, arguments=WRONG_DATES, precision=0.5),
            ),
            ended(
                "a1",
                "dev-c",
                1,
                "scored",
                True,
                scored_turn(seconds=5, arguments=None, precision=None, output=declined()),
            ),
        ],
        "a2": [
            ended("a2", "dev-a", 1, "infrastructure", None),
            ended("a2", "dev-a", 2, "scored", True, scored_turn(seconds=11)),
            ended(
                "a2",
                "dev-b",
                1,
                "scored",
                False,
                scored_turn(seconds=21, passed=False, arguments=WRONG_DATES),
            ),
            *(ended("a2", "dev-c", n, "infrastructure", None) for n in (1, 2, 3)),
        ],
        "a3": [
            ended("a3", "dev-a", 1, "scored", True, scored_turn(seconds=12)),
            ended("a3", "dev-b", 1, "scored", True, scored_turn(seconds=22)),
            ended("a3", "dev-c", 1, "errored", False),
        ],
    }
    counts = {
        "a1": TaskCounts(passed=2, failed=1),
        "a2": TaskCounts(passed=1, failed=1, lost=1),
        "a3": TaskCounts(passed=2, errored=1),
    }
    for index, (run_id, attempts) in enumerate(m1.items()):
        write_attempts(
            runs,
            run_id,
            attempts,
            tasks=counts[run_id],
            run_index=index + 1,
            model=model_build("m1", m1_digests[index]),
            unattributed_calls={"a1": 2, "a3": 1}.get(run_id, 0),
        )
    for index, run_id in enumerate(("b1", "b2", "b3")):
        attempts = [
            ended(
                run_id,
                "dev-a",
                1,
                "scored",
                False,
                scored_turn(
                    seconds=3,
                    passed=False,
                    arguments=NO_SEARCH,
                    precision=None,
                    output=declined("parse_error"),
                ),
            ),
            ended(run_id, "dev-b", 1, "scored", True, scored_turn(seconds=15)),
            ended(
                run_id,
                "dev-c",
                1,
                "scored",
                True,
                scored_turn(seconds=4, arguments=None, precision=None, output=declined()),
            ),
        ]
        write_attempts(
            runs,
            run_id,
            attempts,
            tasks=TaskCounts(passed=2, failed=1),
            run_index=index + 1,
            model=model_build("m2", "d2"),
        )


def build(runs: Path, **options: Any) -> Report:  # noqa: ANN401 - build_report's options
    listed = collect_runs(
        runs,
        "dev",
        {task.id: len(task.turns) for task in GOLDEN_TASKS},
        golden=GOLDEN,
        allow_dirty=options.pop("allow_dirty", False),
    )
    return build_report("dev", listed, GOLDEN_TASKS, **options)


@pytest.fixture
def scenario(tmp_path: Path) -> Report:
    write_scenario(tmp_path)
    return build(tmp_path)


def model(found: Report, name: str) -> ModelSummary:
    (summary,) = [m for m in found.models if m.model == name]
    return summary


class TestHeadline:
    """Counts with 95% bootstrap intervals over whole tasks (docs/evaluation.md), lost tasks both ways."""

    def test_lost_tasks_left_out(self, scenario: Report) -> None:
        m1 = model(scenario, "m1")
        assert m1.lost_excluded is not None
        assert m1.lost_excluded.estimate == pytest.approx(
            1 + 1 / 3 + 1 / 2
        )  # dev-c: 1 of its 2 runs with a result
        assert m1.with_result == 3

    def test_lost_tasks_counted_as_failed(self, scenario: Report) -> None:
        m1 = model(scenario, "m1")
        assert m1.lost_failed.estimate == pytest.approx(1 + 1 / 3 + 1 / 3)
        assert m1.tasks == 3

    def test_the_intervals_hold_the_estimate_and_stay_within_the_tasks(self, scenario: Report) -> None:
        for summary in scenario.models:
            for interval, top in (
                (summary.lost_excluded, summary.with_result),
                (summary.lost_failed, summary.tasks),
            ):
                assert interval is not None
                assert 0 <= interval.low <= interval.estimate <= interval.high <= top

    def test_a_model_whose_runs_agree_still_has_an_interval_since_its_tasks_differ(
        self, scenario: Report
    ) -> None:
        m2 = model(scenario, "m2")
        assert m2.lost_failed.estimate == pytest.approx(2.0)
        assert m2.lost_failed.low < 2.0 < m2.lost_failed.high  # resampling the tasks changes the count

    def test_stability(self, scenario: Report) -> None:
        assert model(scenario, "m1").stability == pytest.approx(1 / 3)  # dev-a only
        assert model(scenario, "m2").stability == pytest.approx(1.0)

    def test_each_task_with_its_results_in_run_order(self, scenario: Report) -> None:
        assert model(scenario, "m1").per_task == {
            "dev-a": ["passed", "passed", "passed"],
            "dev-b": ["failed", "failed", "passed"],
            "dev-c": ["passed", "lost", "errored"],
        }

    def test_each_category_descriptively(self, scenario: Report) -> None:
        shares = model(scenario, "m1").per_category
        assert list(shares) == ["single_search", "robustness", "out_of_scope"]  # in the golden set's order
        assert shares["single_search"] == CategoryShare(passed=1.0, with_result=1, tasks=1)
        assert shares["robustness"].passed == pytest.approx(1 / 3)
        assert (shares["robustness"].with_result, shares["robustness"].tasks) == (1, 1)
        assert shares["out_of_scope"] == CategoryShare(passed=0.5, with_result=1, tasks=1)

    def test_the_runs_of_a_model_are_named_with_their_counts(self, scenario: Report) -> None:
        m1 = model(scenario, "m1")
        assert m1.runs == ["a1", "a2", "a3"]
        assert m1.run_counts == [
            TaskCounts(passed=2, failed=1),
            TaskCounts(passed=1, failed=1, lost=1),
            TaskCounts(passed=2, errored=1),
        ]
        assert (m1.digest, m1.server_versions) == ("d1", ["0.40.1"])


class TestSecondary:
    def test_the_first_search_call_and_the_last(self, scenario: Report) -> None:
        m1 = model(scenario, "m1")
        assert m1.last_call == Rate(part=4, whole=6)  # dev-a x3, dev-b in run 3
        assert m1.first_call == Rate(part=3, whole=6)  # dev-a run 1 corrected itself; dev-b wrong twice

    def test_over_refusal_and_over_clarification_on_turns_that_expected_a_search(
        self, scenario: Report
    ) -> None:
        assert model(scenario, "m2").over_refusal == Rate(part=3, whole=6)
        assert model(scenario, "m2").over_clarification == Rate(part=0, whole=6)
        assert model(scenario, "m1").over_refusal == Rate(part=0, whole=6)

    def test_scene_precision_and_recall(self, scenario: Report) -> None:
        m1 = model(scenario, "m1")
        assert m1.precision is not None
        assert m1.precision.value == pytest.approx(5.5 / 6)
        assert m1.precision.turns == 6
        assert m1.recall == Mean(value=0.8, turns=1)
        assert model(scenario, "m2").recall is None  # no task asked for a number of scenes

    def test_failures_and_what_the_infrastructure_cost(self, scenario: Report) -> None:
        m1 = model(scenario, "m1")
        assert m1.model_failures == {}
        assert model(scenario, "m2").model_failures == {"parse_error": 3}
        assert m1.grounding_failures == 0
        assert m1.errored == ["dev-c (run 3)"]
        assert (m1.infrastructure_attempts, m1.lost, m1.unattributed_calls) == (4, 1, 3)  # of every run

    def test_the_cost_of_a_task_over_its_scored_attempts(self, scenario: Report) -> None:
        cost = model(scenario, "m1").cost
        assert cost is not None
        assert cost.tasks == 7  # every scored last attempt: dev-a x3, dev-b x3, dev-c in run 1
        assert cost.median_seconds == 12.0  # of 5, 10, 11, 12, 20, 21, 22
        assert cost.p90_seconds == pytest.approx(21.4)
        assert cost.median_input_tokens == 400.0  # 100 + 300 on a delegated turn; a decline uses none here
        assert cost.median_output_tokens == 40.0


class TestComparison:
    def test_the_sign_test_on_the_tasks_where_the_majorities_differ(self, scenario: Report) -> None:
        (pair,) = scenario.comparisons
        assert pair == Comparison(
            first="m1",
            second="m2",
            first_only=1,
            second_only=1,
            both_passed=0,
            both_failed=0,
            left_out=1,
            p_value=1.0,
        )  # dev-a m1 only, dev-b m2 only, dev-c no majority for m1 (P against E)

    def test_one_model_has_nothing_to_compare(self, tmp_path: Path) -> None:
        write_run(tmp_path, "r1", ALL_PASS)
        found = collect_runs(tmp_path, "dev", ONE_TURN, golden=GOLDEN)
        tasks = [search_task(task_id, Category.SINGLE_SEARCH) for task_id in TASKS]
        assert build_report("dev", found, tasks).comparisons == []


class TestRefusals:
    def test_runs_of_one_model_on_an_older_build_are_listed_not_counted(self, tmp_path: Path) -> None:
        write_scenario(tmp_path, m1_digests=("d1", "d1", "d9"))
        found = build(tmp_path)
        assert [(run.folder, run.state) for run in found.listed][:3] == [
            ("a1", "other_setup"),
            ("a2", "other_setup"),
            ("a3", "reported"),
        ]
        assert found.listed[0].reason == "it ran another build of m1 than a3"
        assert model(found, "m1").runs == ["a3"]

    def test_no_run_that_counts_gives_a_report_of_the_runs_alone(self, tmp_path: Path) -> None:
        write_run(tmp_path, "r1", {}, status="running", tasks=TaskCounts())
        found = build_report("dev", collect_runs(tmp_path, "dev", ONE_TURN, golden=GOLDEN), GOLDEN_TASKS)
        assert (found.models, found.comparisons, [run.state for run in found.listed]) == ([], [], ["running"])


class TestMarkdown:
    def test_the_report_names_what_it_is_made_from_and_every_run(self, scenario: Report) -> None:
        text = render_markdown(scenario)
        assert text.startswith("# Evaluation report: dev split\n")
        assert "Made from 6 runs that count, of 6 listed, on commit 0123456789ab.\n" in text
        assert "| Recall, the mean over turns | 0.80 over 1 turn |" in text
        assert "| single_search | 1.00 of 1 (of 1 task) |" in text
        assert "1 comparison;" in text
        assert "Runs from a checkout with changes" not in text
        for run_id in ("a1", "a2", "a3", "b1", "b2", "b3"):
            assert f"| {run_id} |" in text
        assert "| m1 | 1.8 of 3 (" in text
        assert "| m1 | m2 | 1 | 1 | 1 | 1 | 0 | 0 |" in text
        assert "dev-c (run 3)" in text
        assert "(a question)" not in text  # the report never prints a task's text

    def test_runs_that_do_not_count_are_listed_with_why(self, tmp_path: Path) -> None:
        write_scenario(tmp_path)
        (tmp_path / "x0").mkdir()
        text = render_markdown(build(tmp_path))
        assert "| x0 | did not start | it has no run.json |" in text

    def test_a_report_with_scratch_runs_says_it_is_not_for_publication(self, tmp_path: Path) -> None:
        write_run(tmp_path, "r1", ALL_PASS, code=CodeVersion(commit=COMMIT, dirty=True))
        tasks = [search_task(task_id, Category.SINGLE_SEARCH) for task_id in TASKS]
        found = build_report(
            "dev", collect_runs(tmp_path, "dev", ONE_TURN, golden=GOLDEN, allow_dirty=True), tasks
        )
        assert found.scratch
        assert (
            "Runs from a checkout with changes are in these numbers: not for publication."
            in render_markdown(found)
        )

    def test_one_run_is_said_as_one(self, tmp_path: Path) -> None:
        write_run(tmp_path, "r1", ALL_PASS)
        tasks = [search_task(task_id, Category.SINGLE_SEARCH) for task_id in TASKS]
        text = render_markdown(
            build_report("dev", collect_runs(tmp_path, "dev", ONE_TURN, golden=GOLDEN), tasks)
        )
        assert "Made from 1 run that counts, of 1 listed, on commit 0123456789ab.\n" in text
        assert "| one run only | 1 |" in text  # the same result in every run says nothing about one run

    def test_runs_on_commits_that_differ_only_in_results_say_so(self, repo: Path) -> None:
        first = git(repo, "rev-parse", "HEAD").strip()
        (repo / "results").mkdir()
        (repo / "results" / "note").write_text("a run record\n", encoding="utf-8")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "records")
        second = git(repo, "rev-parse", "HEAD").strip()
        runs = repo / "runs"
        write_run(runs, "r1", ALL_PASS, code=CodeVersion(commit=first, dirty=False))
        write_run(runs, "r2", ALL_PASS, code=CodeVersion(commit=second, dirty=False))
        tasks = [search_task(task_id, Category.SINGLE_SEARCH) for task_id in TASKS]
        found = build_report("dev", collect_runs(runs, "dev", ONE_TURN, golden=GOLDEN), tasks, root=repo)
        on = ", ".join(sorted([first[:12], second[:12]]))
        assert (
            f"Made from 2 runs that count, of 2 listed, on commits {on}, which differ only in `results/`.\n"
            in render_markdown(found)
        )

    def test_with_no_run_that_counts_the_report_says_so(self, tmp_path: Path) -> None:
        text = render_markdown(build_report("dev", [], GOLDEN_TASKS))
        assert "No run counts yet." in text


class TestEdges:
    def test_a_run_that_counts_without_its_record_cannot_be_made(self) -> None:
        with pytest.raises(ValueError, match="a run that counts carries its record"):
            ListedRun(folder="r1", state="reported")

    def test_a_model_with_every_task_lost_has_no_cost_and_nothing_left_out(self, tmp_path: Path) -> None:
        write_run(tmp_path, "r1", {task: [INFRA, INFRA, INFRA] for task in TASKS}, tasks=TaskCounts(lost=3))
        tasks = [search_task(task_id, Category.SINGLE_SEARCH) for task_id in TASKS]
        found = build_report("dev", collect_runs(tmp_path, "dev", ONE_TURN, golden=GOLDEN), tasks)
        (summary,) = found.models
        assert (summary.cost, summary.lost_excluded, summary.with_result) == (None, None, 0)
        assert summary.lost_failed.estimate == 0.0
        assert (summary.lost, summary.infrastructure_attempts) == (3, 9)
        assert summary.last_call == Rate(part=0, whole=0)
        text = render_markdown(found)
        assert "| qwen3.5:9b | none | 0.0 of 3 (" in text
        assert "Seconds per task" not in text
        assert "| Last search call with every argument right | none to count |" in text

    def test_two_models_that_agree_are_counted_as_such(self, tmp_path: Path) -> None:
        write_run(tmp_path, "r1", ALL_PASS, model=model_build("m1"))
        write_run(
            tmp_path,
            "r2",
            {"dev-a": [FAIL], "dev-b": [PASS], "dev-c": [PASS]},
            model=model_build("m2"),
            tasks=TaskCounts(passed=2, failed=1),
        )
        tasks = [search_task(task_id, Category.SINGLE_SEARCH) for task_id in TASKS]
        (pair,) = build_report(
            "dev", collect_runs(tmp_path, "dev", ONE_TURN, golden=GOLDEN), tasks
        ).comparisons
        assert (pair.first_only, pair.second_only, pair.both_passed, pair.both_failed, pair.left_out) == (
            1,
            0,
            2,
            0,
            0,
        )
        assert pair.p_value == 1.0  # one task where they differ is no evidence


class TestWhatTheScenarioDoesNotCover:
    def tasks(self) -> list[Task]:
        return [search_task(task_id, Category.SINGLE_SEARCH) for task_id in TASKS]

    def test_a_turn_that_cited_nothing_has_no_precision_to_average(self, tmp_path: Path) -> None:
        cited_nothing = scored_turn(seconds=1).model_copy(update={"oracle": OracleCheck(ids=[SCENE.id])})
        write_attempts(
            tmp_path,
            "r1",
            [
                ended("r1", "dev-a", 1, "scored", True, cited_nothing),
                ended("r1", "dev-b", 1, "scored", True, scored_turn(seconds=1)),
                ended("r1", "dev-c", 1, "scored", True, scored_turn(seconds=1)),
            ],
        )
        (summary,) = build_report(
            "dev", collect_runs(tmp_path, "dev", ONE_TURN, golden=GOLDEN), self.tasks()
        ).models
        assert summary.precision == Mean(value=1.0, turns=2)  # the share of nothing is not 0, it is undefined

    def test_the_seconds_of_a_task_are_those_of_all_its_turns(self, tmp_path: Path) -> None:
        two_turns = AttemptRecord(
            run_id="r1",
            task_id="dev-a",
            attempt=1,
            outcome="scored",
            passed=True,
            turns=[scored_turn(seconds=10), scored_turn(seconds=7)],
        )
        write_attempts(
            tmp_path,
            "r1",
            [
                two_turns,
                ended("r1", "dev-b", 1, "scored", True, scored_turn(seconds=1)),
                ended("r1", "dev-c", 1, "scored", True, scored_turn(seconds=2)),
            ],
        )
        turns = {"dev-a": 2, "dev-b": 1, "dev-c": 1}
        tasks = [follow_up_task("dev-a"), *self.tasks()[1:]]
        (summary,) = build_report("dev", collect_runs(tmp_path, "dev", turns, golden=GOLDEN), tasks).models
        assert summary.cost is not None
        # dev-a takes 10 + 7: of 1, 2, 17 the median is 2 and the 90th percentile 2 + 0.8 * (17 - 2) = 14.
        assert summary.cost.median_seconds == 2.0
        assert summary.cost.p90_seconds == pytest.approx(14.0)  # 8.4 if only a task's first turn counted
        assert summary.cost.median_input_tokens == 400.0  # dev-a's two turns use 800: median of 800, 400, 400
        assert summary.cost.tasks == 3

    def test_the_tokens_of_a_task_add_its_decision_and_its_retrieval(self, tmp_path: Path) -> None:
        write_run(tmp_path, "r1", ALL_PASS)
        (summary,) = build_report(
            "dev", collect_runs(tmp_path, "dev", ONE_TURN, golden=GOLDEN), self.tasks()
        ).models
        assert summary.cost is not None
        assert (summary.cost.median_input_tokens, summary.cost.median_output_tokens) == (400.0, 40.0)

    def test_runs_on_other_prompts_are_listed_not_counted_together(self, tmp_path: Path) -> None:
        write_run(tmp_path, "r1", ALL_PASS)
        write_run(tmp_path, "r2", ALL_PASS, hashes=hashes(retrieval_prompt="0" * 64))
        found = build_report("dev", collect_runs(tmp_path, "dev", ONE_TURN, golden=GOLDEN), self.tasks())
        assert [(run.state, run.reason) for run in found.listed] == [
            ("other_setup", "its retrieval prompt differs from that of r2"),
            ("reported", None),
        ]
        assert found.models[0].runs == ["r2"]

    def test_a_reason_cannot_break_the_table(self, tmp_path: Path) -> None:
        reason = "turn 1 raised ValueError: a | b\nsecond line"
        write_run(tmp_path, "r1", {}, status="aborted", aborted_reason=reason, tasks=TaskCounts())
        text = render_markdown(
            build_report("dev", collect_runs(tmp_path, "dev", ONE_TURN, golden=GOLDEN), self.tasks())
        )
        assert "| r1 | aborted | turn 1 raised ValueError: a / b second line | qwen3.5:9b |" in text


class TestReviewFindings:
    """What the review of 2026-11-08 found, each checked."""

    def tasks(self) -> list[Task]:
        return [search_task(task_id, Category.SINGLE_SEARCH) for task_id in TASKS]

    def report(self, runs: Path, **options: Any) -> Report:  # noqa: ANN401 - collect_runs' options
        return build_report(
            "dev", collect_runs(runs, "dev", ONE_TURN, golden=GOLDEN, **options), self.tasks()
        )

    def test_a_link_to_a_run_does_not_count_it_twice(self, tmp_path: Path) -> None:
        folder = write_run(tmp_path, "r1", ALL_PASS)
        (tmp_path / "r2").symlink_to(folder)
        assert states(collect(tmp_path)) == [("r1", "reported"), ("r2", "inconsistent")]
        assert collect(tmp_path)[1].reason == "a link to another folder, not a run's own"

    def test_a_copy_of_a_run_under_another_name_does_not_count(self, tmp_path: Path) -> None:
        folder = write_run(tmp_path, "r1", ALL_PASS)
        shutil.copytree(folder, tmp_path / "r1-copy")
        assert [(run.folder, run.state, run.reason) for run in collect(tmp_path)] == [
            ("r1", "reported", None),
            ("r1-copy", "inconsistent", "its record is of run r1"),
        ]

    def test_hidden_folders_are_not_runs(self, tmp_path: Path) -> None:
        write_run(tmp_path, "r1", ALL_PASS)
        (tmp_path / ".git").mkdir()
        assert states(collect(tmp_path)) == [("r1", "reported")]

    def test_a_record_naming_a_split_that_does_not_exist_is_listed(self, tmp_path: Path) -> None:
        folder = tmp_path / "r1"
        folder.mkdir()
        (folder / "run.json").write_text('{"split": "Dev"}', encoding="utf-8")
        assert states(collect(tmp_path)) == [("r1", "unreadable")]

    @pytest.mark.parametrize(
        ("name", "said"), [("run.json", "run.json"), ("attempts.jsonl", "attempts.jsonl")]
    )
    def test_a_record_that_is_not_utf8_text_is_unreadable_not_a_crash(
        self, tmp_path: Path, name: str, said: str
    ) -> None:
        folder = write_run(tmp_path, "r1", ALL_PASS)
        (folder / name).write_bytes(b"\xff\xfe\x00")
        (run,) = collect(tmp_path)
        assert (run.state, run.reason) == ("unreadable", f"{said} is not UTF-8 text")

    def test_attempts_that_cannot_be_read_as_a_file_say_so(self, tmp_path: Path) -> None:
        folder = tmp_path / "r1"
        write_run_record(folder, record("r1"))
        (folder / "attempts.jsonl").mkdir()
        (run,) = collect(tmp_path)
        assert (run.state, run.reason) == ("unreadable", "attempts.jsonl cannot be read (Is a directory)")

    @pytest.mark.parametrize(
        ("the_attempt", "said"),
        [
            (
                AttemptRecord(
                    run_id="r1",
                    task_id="dev-a",
                    attempt=1,
                    outcome="scored",
                    passed=True,
                    turns=[turn(score=FAILED)],
                ),
                "dev-a passed, though a turn of it failed",
            ),
            (
                AttemptRecord(
                    run_id="r1", task_id="dev-a", attempt=1, outcome="scored", passed=False, turns=[turn()]
                ),
                "dev-a failed, though every turn of it passed",
            ),
        ],
    )
    def test_a_verdict_its_turns_contradict_is_inconsistent(
        self, tmp_path: Path, the_attempt: AttemptRecord, said: str
    ) -> None:
        counts = TaskCounts(passed=3) if the_attempt.passed else TaskCounts(passed=2, failed=1)
        write_attempts(
            tmp_path,
            "r1",
            [the_attempt, *(attempt("r1", t, 1, *PASS) for t in ("dev-b", "dev-c"))],
            tasks=counts,
        )
        (run,) = collect(tmp_path)
        assert (run.state, run.reason) == ("inconsistent", said)

    def test_a_passed_follow_up_with_one_of_its_two_turns_is_inconsistent(self, tmp_path: Path) -> None:
        write_run(tmp_path, "r1", ALL_PASS)
        found = collect_runs(tmp_path, "dev", {"dev-a": 2, "dev-b": 1, "dev-c": 1}, golden=GOLDEN)
        assert [(run.state, run.reason) for run in found] == [
            ("inconsistent", "dev-a passed with 1 of its 2 turns")
        ]

    def test_a_failed_attempt_that_went_on_after_its_failed_turn_is_inconsistent(
        self, tmp_path: Path
    ) -> None:
        went_on = AttemptRecord(
            run_id="r1",
            task_id="dev-a",
            attempt=1,
            outcome="scored",
            passed=False,
            turns=[turn(score=FAILED), turn()],
        )
        write_attempts(
            tmp_path,
            "r1",
            [went_on, *(attempt("r1", t, 1, *PASS) for t in ("dev-b", "dev-c"))],
            tasks=TaskCounts(passed=2, failed=1),
        )
        found = collect_runs(tmp_path, "dev", {"dev-a": 2, "dev-b": 1, "dev-c": 1}, golden=GOLDEN)
        assert [(run.state, run.reason) for run in found] == [
            ("inconsistent", "dev-a failed but did not stop at its first failed turn")
        ]

    def test_an_errored_task_is_named_by_its_column_whatever_the_run_index(self, tmp_path: Path) -> None:
        errored = {"dev-a": [PASS], "dev-b": [PASS], "dev-c": [ERROR]}
        write_run(tmp_path, "r1", ALL_PASS, run_index=1)
        write_run(
            tmp_path, "r2", errored, run_index=1, tasks=TaskCounts(passed=2, errored=1)
        )  # a second evaluation
        (summary,) = self.report(tmp_path).models
        assert summary.errored == ["dev-c (run 2)"]

    def test_failed_catalogue_queries_are_counted(self, tmp_path: Path) -> None:
        failed_query = scored_turn(seconds=1).model_copy(
            update={"oracle": OracleCheck(failed="CatalogError: down")}
        )
        write_attempts(
            tmp_path,
            "r1",
            [
                ended("r1", "dev-a", 1, "scored", True, failed_query),
                *(attempt("r1", t, 1, *PASS) for t in ("dev-b", "dev-c")),
            ],
        )
        (summary,) = self.report(tmp_path).models
        assert summary.oracle_failures == 1
        assert summary.precision == Mean(value=1.0, turns=2)

    @pytest.mark.parametrize(
        ("options", "said"),
        [
            (
                {"seed": 42, "temperature": 0.0},
                "- The runs were at temperature 0: at 0 a model's runs are nearly identical",
            ),
            ({"seed": 42, "temperature": 0.7}, "- The runs were at temperature 0.7.\n"),
            ({"seed": 42}, "- The runs were at temperature not recorded.\n"),
        ],
    )
    def test_the_note_on_temperature_says_what_the_runs_used(
        self, tmp_path: Path, options: dict[str, Any], said: str
    ) -> None:
        write_run(tmp_path, "r1", ALL_PASS, model=model_build("m1").model_copy(update={"options": options}))
        assert said in render_markdown(self.report(tmp_path))

    def test_a_grounding_failure_is_not_called_impossible(self, tmp_path: Path) -> None:
        ungrounded = TurnScore(
            route=True,
            tools=True,
            arguments=ALL_RIGHT,
            first_arguments=ALL_RIGHT,
            grounded=False,
            outcome=True,
        )
        write_attempts(
            tmp_path,
            "r1",
            [
                ended("r1", "dev-a", 1, "scored", False, turn(score=ungrounded)),
                *(attempt("r1", t, 1, *PASS) for t in ("dev-b", "dev-c")),
            ],
            tasks=TaskCounts(passed=2, failed=1),
        )
        text = render_markdown(self.report(tmp_path))
        assert "so grounding is a property of the design" not in text
        assert "- 1 answer cited a scene no tool returned, which the templates should prevent" in text

    def test_the_noise_figures_of_the_rules_are_quoted_only_for_the_test_split(
        self, scenario: Report
    ) -> None:
        assert "with 3 tasks the intervals are wide" in render_markdown(scenario)
        assert "can be noise" not in render_markdown(scenario)
        # The figure is for the 50 tasks of golden-test-v2; the 18 of v1 have another one in the rules.
        assert report.TEST_SPLIT_TASKS == 50
        quoted = "less than about 16 to 20 points (8 to 10 tasks) can be noise"
        assert f"{quoted} (docs/evaluation.md)" in "\n".join(report._comparison_table([], 50))
        rules = Path(__file__).resolve().parents[1] / "docs" / "evaluation.md"
        assert quoted in rules.read_text(encoding="utf-8")  # the report quotes the rules, word for word
        assert "with 18 tasks the intervals are wide" in "\n".join(report._comparison_table([], 18))

    def test_a_model_with_other_than_three_runs_says_so(self, tmp_path: Path) -> None:
        write_run(tmp_path, "r1", ALL_PASS)
        assert "1 run, where the rules call for 3: r1." in render_markdown(self.report(tmp_path))

    def test_the_stopped_attempts_are_explained(self, scenario: Report) -> None:
        assert "- An attempt stops at its first failed turn" in render_markdown(scenario)

    def test_names_cannot_break_a_table_or_a_heading(self, tmp_path: Path) -> None:
        write_run(tmp_path, "r1", ALL_PASS, model=model_build("a|b\nc"))
        text = render_markdown(self.report(tmp_path))
        assert "| r1 | counted |  | a/b c |" in text
        assert "## a/b c\n" in text
        assert "a|b" not in text


class TestPValues:
    @pytest.mark.parametrize(
        ("value", "shown"),
        [
            (1.0, "1"),
            (0.21875, "0.22"),
            (0.03125, "0.031"),
            (0.0004, "<0.001"),
            (0.625, "0.63"),  # an exact tie rounds up, as read by hand, not to the even digit
            (0.125, "0.13"),
            (0.0625, "0.063"),
            (0.1, "0.1"),
        ],
    )
    def test_small_values_are_never_shown_as_zero(self, value: float, shown: str) -> None:
        assert report._p(value) == shown


class TestWarmUp:
    def test_each_runs_warm_up_is_shown_and_a_run_without_one_says_so(self, tmp_path: Path) -> None:
        write_run(tmp_path, "r1", ALL_PASS)  # a record made before the warm-up existed
        write_run(tmp_path, "r2", ALL_PASS, warm_up_seconds=2.04)
        tasks = [search_task(task_id, Category.SINGLE_SEARCH) for task_id in TASKS]
        found = build_report("dev", collect_runs(tmp_path, "dev", ONE_TURN, golden=GOLDEN), tasks)
        assert found.models[0].warm_up_seconds == [None, 2.04]
        assert (
            "| Warm-up after each run's fresh load of the model, seconds | not done, 2.0 |"
            in render_markdown(found)
        )


class TestEveryState:
    def test_every_state_a_run_can_be_in_has_its_words(self) -> None:
        assert set(report.STATE_WORDS) == set(get_args(report.RunState))

    def test_a_report_with_a_run_on_another_setup_renders(self, tmp_path: Path) -> None:
        write_run(tmp_path, "r1", ALL_PASS, shuffle_seed=7)
        write_run(tmp_path, "r2", ALL_PASS)
        tasks = [search_task(task_id, Category.SINGLE_SEARCH) for task_id in TASKS]
        text = render_markdown(
            build_report("dev", collect_runs(tmp_path, "dev", ONE_TURN, golden=GOLDEN), tasks)
        )
        assert "| r1 | other setup | its task order's seed differs from that of r2 |" in text
        assert "Made from 1 run that counts, of 2 listed" in text


def timed_turn(seconds: float, decision_model_s: float, retrieval_model_s: float) -> TurnRecord:
    """A passed turn whose output says how long the model took in the decision and in the retrieval."""
    base = turn()
    assert base.output is not None
    assert base.output.retrieval is not None
    output = base.output.model_copy(
        update={
            "decision_usage": Usage(model_seconds=decision_model_s),
            "retrieval": base.output.retrieval.model_copy(
                update={"usage": Usage(model_seconds=retrieval_model_s)}
            ),
        }
    )
    return turn(seconds=seconds, output=output)


class TestTimePerRun:
    """Each run's time over its scored attempts, so that a drift from run to run can be read (2026-11-08)."""

    def tasks(self) -> list[Task]:
        return [search_task(task_id, Category.SINGLE_SEARCH) for task_id in TASKS]

    def test_each_runs_seconds_are_those_of_its_scored_attempts(self, tmp_path: Path) -> None:
        # r1: three scored tasks of 10, 20 and 30 s, the model taking 1 + 8, 2 + 16 and 3 + 24 s of them.
        # r2: dev-a lost to the infrastructure (its attempts left out), dev-b and dev-c of 5 and 7 s.
        write_attempts(
            tmp_path,
            "r1",
            [
                ended("r1", "dev-a", 1, "scored", True, timed_turn(10, 1, 8)),
                ended("r1", "dev-b", 1, "scored", True, timed_turn(20, 2, 16)),
                ended("r1", "dev-c", 1, "scored", True, timed_turn(30, 3, 24)),
            ],
        )
        write_attempts(
            tmp_path,
            "r2",
            [
                *(attempt("r2", "dev-a", n, *INFRA) for n in (1, 2, 3)),
                ended("r2", "dev-b", 1, "scored", True, timed_turn(5, 1, 3)),
                ended("r2", "dev-c", 1, "scored", True, timed_turn(7, 1, 5)),
            ],
            tasks=TaskCounts(passed=2, lost=1),
        )
        (summary,) = build_report(
            "dev", collect_runs(tmp_path, "dev", ONE_TURN, golden=GOLDEN), self.tasks()
        ).models
        assert summary.run_seconds == [
            RunSeconds(tasks=3, seconds=60.0, model_seconds=54.0),
            RunSeconds(tasks=2, seconds=12.0, model_seconds=10.0),
        ]
        text = render_markdown(
            build_report("dev", collect_runs(tmp_path, "dev", ONE_TURN, golden=GOLDEN), self.tasks())
        )
        assert "| Seconds in turns per run, over its scored tasks | 60.0 (3 tasks), 12.0 (2 tasks) |" in text
        assert "| Model seconds per run, over its scored tasks | 54.0, 10.0 |" in text

    def test_a_run_with_nothing_scored_says_so(self, tmp_path: Path) -> None:
        write_run(tmp_path, "r1", {task: [INFRA, INFRA, INFRA] for task in TASKS}, tasks=TaskCounts(lost=3))
        (summary,) = build_report(
            "dev", collect_runs(tmp_path, "dev", ONE_TURN, golden=GOLDEN), self.tasks()
        ).models
        assert summary.run_seconds == [RunSeconds(tasks=0, seconds=0.0, model_seconds=0.0)]
        text = render_markdown(
            build_report("dev", collect_runs(tmp_path, "dev", ONE_TURN, golden=GOLDEN), self.tasks())
        )
        assert "| Seconds in turns per run, over its scored tasks | 0.0 (0 tasks) |" in text

    def test_a_turn_that_raised_used_no_model_time_on_record(self) -> None:
        raised = turn(output=None, error="RetrievalAgentError: failed", score=None, oracle=None, calls=[])
        assert report._usages(raised) == []
        assert report._usages(timed_turn(5, 1, 3)) == [Usage(model_seconds=1), Usage(model_seconds=3)]
