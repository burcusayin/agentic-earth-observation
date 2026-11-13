"""The report on a split's runs (docs/evaluation.md, "Runs and statistics" and "Comparing two setups").

This first part reads the run folders: which runs count, and what each task came to in them. Every run is
listed, and one that does not count says why: reports cover every run, failed ones included (decision of
2026-11-05). Only a run that ended `complete`, from a clean checkout, on the frozen files and the split's
tasks, with a record that adds up, in a folder of its own name, and on the same setup as the latest such run,
is in the headline numbers.

The runs of earlier setups keep their results, for the comparison of two setups (decision of 2026-11-12). So
do the runs of an earlier frozen golden set that `EARLIER_GOLDEN_SETS` names: they are compared with the first
setup on this golden set, on the tasks they share, and used for nothing else.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import shutil
import subprocess
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agentic_earth_observation.config import ROOT
from agentic_earth_observation.evals.golden import SPLITS, Task
from agentic_earth_observation.evals.groups import added_in_v2
from agentic_earth_observation.evals.records import (
    ATTEMPTS_FILE,
    MAX_ATTEMPTS,
    OUTPUT_FOLDER,
    RUN_FILE,
    AttemptRecord,
    RecordError,
    RunRecord,
    TaskCounts,
    TurnRecord,
    read_attempts,
    read_run_record,
)
from agentic_earth_observation.evals.run_eval import (
    FROZEN_EVALUATION_CODE,
    FROZEN_EVALUATION_RULES,
    FROZEN_GAZETTEER,
    Split,
)
from agentic_earth_observation.evals.scoring import ArgumentScore
from agentic_earth_observation.evals.setups import (
    Actions,
    Change,
    Determinism,
    PeriodChange,
    TurnKey,
    change,
    determinism,
    period_change,
    turn_actions,
)
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
from agentic_earth_observation.retrieval_agent.models import Usage

RunState = Literal[
    "did_not_start",
    "unreadable",
    "running",
    "aborted",
    "dirty",
    "other_golden_set",
    "earlier_golden_set",
    "inconsistent",
    "other_setup",
    "reported",
]
COUNTED: tuple[Result, ...] = ("passed", "failed", "errored", "lost")  # the order counts are written in
RUNS_PER_MODEL = 3  # docs/evaluation.md: "Every model runs every task 3 times"
AgentsCode = Literal["same", "differs", "unknown"]
SYSTEM_CODE = "src/agentic_earth_observation/"  # the code the agents run, but for:
EVALUATION_CODE = "src/agentic_earth_observation/evals/"
# The frozen test split (golden-test-v2), which the noise figure quoted from docs/evaluation.md is for.
TEST_SPLIT_TASKS = 50


class EarlierGoldenSet(BaseModel):
    """A frozen golden set whose runs the report still reads, for the session-noise comparison only.

    `golden` and `evaluation_rules` are the hashes its runs record. `tasks` is, for each split, the digest
    (`tasks_digest`) of the tasks its runs ran, as those tasks read in the golden set the report is made from:
    a run on it is used only when its tasks have not changed since.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tag: str
    golden: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_rules: str = Field(pattern=r"^[0-9a-f]{64}$")
    tasks: dict[Split, str]


# The golden set of v1.0's runs. Its 18 test tasks are the first 18 of golden-test-v2, byte for byte
# (tests/test_golden.py); the digests were taken from the tasks at tag golden-test-v1.
EARLIER_GOLDEN_SETS = (
    EarlierGoldenSet(
        tag="golden-test-v1",
        golden="743d5e2abb5db223099b2b6f37c73fbdc503b80c745d4a456e999003b5ed9bc7",
        evaluation_rules="9f3ae9ba0643eebc8082a4b6dc549224fc5674a85af3fa8d75f18ca3f10abb5b",
        tasks={
            "dev": "87f2ad93d770ff55687fc95284afa07b4929e5a97c6afed2f8b646e9f6fbb7b2",
            "test": "e5ff07fec44deb3bbc899696fec34495e0c2745d593846314fea5a8b7e16e014",
        },
    ),
)


def tasks_digest(tasks: Sequence[Task]) -> str:
    """SHA-256 over tasks as they read, in the order of their ids: equal digests, equal tasks. Each task is
    written as JSON with sorted keys by the standard library, so that the order in which pydantic lays out a
    task's fields does not matter."""
    digest = hashlib.sha256()
    for task in sorted(tasks, key=lambda task: task.id):
        text = json.dumps(task.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        digest.update(text.encode("utf-8") + b"\n")
    return digest.hexdigest()


class ReportError(Exception):
    """The report cannot be made as asked. The message says what to do."""


class ListedRun(BaseModel):
    """A run as the report lists it: its folder's name, whether it counts, and why not."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    folder: str
    state: RunState
    reason: str | None = None
    record: RunRecord | None = None
    # task id -> what it came to, and the attempts: for a run that counts, a run of an earlier setup and a run
    # of an earlier golden set the report still reads (the comparison of setups uses them); otherwise empty
    results: dict[str, Result] = {}
    attempts: list[AttemptRecord] = []

    @model_validator(mode="after")
    def _counted_runs_carry_their_record(self) -> Self:
        if self.state == "reported" and self.record is None:
            raise ValueError("a run that counts carries its record")
        return self


def collect_runs(
    runs_dir: Path,
    split: Split,
    task_turns: Mapping[str, int],
    *,
    golden: str,
    allow_dirty: bool = False,
    tasks: Sequence[Task] = (),
) -> list[ListedRun]:
    """Every run of `split` under `runs_dir`, in the order they started (their folder names sort so).

    `task_turns` names the split's tasks and how many turns each has, and `golden` is the hash of the golden
    set the report reads: a run on another golden set is listed, not counted. A run on one of
    `EARLIER_GOLDEN_SETS` keeps its results when `tasks`, the split's tasks, show that the tasks it ran are
    unchanged. `allow_dirty` counts runs from a checkout with changes, for scratch use. A folder without a
    readable record cannot always say its split, so it is listed unless it names the other split. Hidden
    folders (`.git`, ...) are not runs.
    """
    if not runs_dir.exists():
        return []
    if not runs_dir.is_dir():
        raise ReportError(f"{runs_dir.name} is not a folder: give the folder that holds the runs' folders")
    listed = []
    for folder in sorted(
        path for path in runs_dir.iterdir() if path.is_dir() and not path.name.startswith(".")
    ):
        run = _listed(folder, split, task_turns, golden, allow_dirty, tasks)
        if run is not None:
            listed.append(run)
    return listed


def _listed(
    folder: Path,
    split: Split,
    task_turns: Mapping[str, int],
    golden: str,
    allow_dirty: bool,
    tasks: Sequence[Task],
) -> ListedRun | None:
    """How the report lists one folder, or None for a run of the other split."""
    if folder.is_symlink():
        return ListedRun(
            folder=folder.name, state="inconsistent", reason="a link to another folder, not a run's own"
        )
    if not (folder / RUN_FILE).exists():
        return ListedRun(folder=folder.name, state="did_not_start", reason=f"it has no {RUN_FILE}")
    try:
        record = read_run_record(folder)
    except RecordError as err:
        named = _named_split(folder)
        if named in SPLITS and named != split:
            return None
        return ListedRun(folder=folder.name, state="unreadable", reason=_unreadable(err))
    if record.split != split:
        return None

    def listed(state: RunState, reason: str | None) -> ListedRun:
        return ListedRun(folder=folder.name, state=state, reason=reason, record=record)

    if record.run_id != folder.name:  # a copy, or a folder renamed by hand: it would count twice
        return listed("inconsistent", f"its record is of run {record.run_id}")
    if record.status == "running":
        return listed("running", "it crashed, or is still running")
    if record.status == "aborted":
        return listed("aborted", record.aborted_reason)
    if record.code.dirty and not allow_dirty:
        return listed("dirty", "the checkout had changes that were not committed")
    state: RunState = "reported"
    reason = None
    if changed := _other_frozen_files(record, golden):
        earlier, why_not = _earlier_golden_set(record, split, tasks)
        if earlier is None:
            verb = "differs" if len(changed) == 1 else "differ"
            said = f"{' and '.join(changed)} {verb} from what this report reads"
            return listed("other_golden_set", f"{said}: {why_not}" if why_not else said)
        state = "earlier_golden_set"
        reason = (
            f"it ran on {earlier.tag}, on {len(record.task_order)} of these tasks: it is read only for the "
            "comparison of setups"
        )
    elif set(record.task_order) != set(task_turns):  # a record that names a task twice does not read at all
        return listed("other_golden_set", "its tasks are not the tasks of the split")
    try:
        attempts = read_attempts(folder)
        results = task_results(record, attempts, task_turns)
    except RecordError as err:
        return listed("unreadable", _attempts_unreadable(err))
    except ValueError as err:
        return listed("inconsistent", str(err))
    tally = Counter(results.values())
    found = TaskCounts(**{what: tally[what] for what in COUNTED})
    if found != record.tasks:
        return listed(
            "inconsistent",
            f"{RUN_FILE} counts {_counts(record.tasks)}; {ATTEMPTS_FILE} gives {_counts(found)}",
        )
    return ListedRun(
        folder=folder.name, state=state, reason=reason, record=record, results=results, attempts=attempts
    )


def _earlier_golden_set(
    record: RunRecord, split: Split, tasks: Sequence[Task]
) -> tuple[EarlierGoldenSet | None, str | None]:
    """The earlier golden set the run ran on, if the report still reads it: one of `EARLIER_GOLDEN_SETS`, on
    today's evaluation code and gazetteer, whose tasks in this split read as they did. Else None, and, for a
    run on one of them, why it is not read."""
    hashes = (record.hashes.golden, record.hashes.evaluation_rules)
    earlier = next((one for one in EARLIER_GOLDEN_SETS if (one.golden, one.evaluation_rules) == hashes), None)
    if earlier is None:
        return None, None
    by_id = {task.id: task for task in tasks}
    if (record.hashes.evaluation_code, record.hashes.gazetteer) != (FROZEN_EVALUATION_CODE, FROZEN_GAZETTEER):
        why = "on other evaluation code or another gazetteer than this report reads"
    elif not tasks:
        why = "and the report was not given the split's tasks to check them against"
    elif not set(record.task_order) <= set(by_id):
        why = "and some of its tasks are not tasks of this split"
    elif split not in earlier.tasks:
        why = f"which names no tasks of the {split} split"
    elif tasks_digest([by_id[task_id] for task_id in record.task_order]) != earlier.tasks[split]:
        why = "and its tasks have changed since"
    else:
        return earlier, None
    return None, f"it ran on {earlier.tag}, {why}"


def _other_frozen_files(record: RunRecord, golden: str) -> list[str]:
    """The frozen files the run used that are not the ones this report reads."""
    expected = {
        "the golden set": (record.hashes.golden, golden),
        "the evaluation code": (record.hashes.evaluation_code, FROZEN_EVALUATION_CODE),
        "the gazetteer": (record.hashes.gazetteer, FROZEN_GAZETTEER),
        "docs/evaluation.md": (record.hashes.evaluation_rules, FROZEN_EVALUATION_RULES),
    }
    return [name for name, (used, read) in expected.items() if used != read]


def task_results(
    record: RunRecord, attempts: Sequence[AttemptRecord], task_turns: Mapping[str, int]
) -> dict[str, Result]:
    """What each task of a run came to, by its last attempt. ValueError when the attempts do not add up to a
    finished run: a task without attempts, an attempt of another run or task, attempts out of order, an
    attempt after the last, a task that ended before its last attempt without being scored, or a scored
    attempt whose turns say otherwise than its verdict (`task_turns`: how many turns each task has)."""
    tried: dict[str, list[AttemptRecord]] = {}
    for attempt in attempts:
        if attempt.run_id != record.run_id:
            raise ValueError(f"an attempt of run {attempt.run_id} in the folder of run {record.run_id}")
        if attempt.task_id not in record.task_order:
            raise ValueError(f"an attempt of {attempt.task_id}, which is not one of the run's tasks")
        earlier = tried.setdefault(attempt.task_id, [])
        if earlier and earlier[-1].outcome != "infrastructure":
            raise ValueError(f"{attempt.task_id} has an attempt after its last")
        if attempt.attempt != len(earlier) + 1:
            raise ValueError(
                f"{attempt.task_id}: attempt {attempt.attempt} where attempt {len(earlier) + 1} was due"
            )
        earlier.append(attempt)
    results: dict[str, Result] = {}
    for task_id in record.task_order:
        if task_id not in tried:
            raise ValueError(f"{task_id} has no attempt")
        last = tried[task_id][-1]
        if last.outcome == "scored":
            _check_verdict(last, task_turns[task_id])
            results[task_id] = "passed" if last.passed else "failed"
        elif last.outcome == "errored":
            results[task_id] = "errored"
        elif last.attempt == MAX_ATTEMPTS:
            results[task_id] = "lost"
        else:
            raise ValueError(f"{task_id} ended after {last.attempt} of {MAX_ATTEMPTS} attempts")
    return results


def _check_verdict(attempt: AttemptRecord, turns: int) -> None:
    """A scored attempt passed when all its task's turns ran and passed, with no model failure; it failed at
    its first failed turn, where the attempt stops (decision of 2026-11-07)."""
    task = attempt.task_id
    ok = [
        turn.score is not None
        and turn.score.passed
        and turn.output is not None
        and turn.output.failure is None
        for turn in attempt.turns
    ]
    if attempt.passed:
        if not all(ok):
            raise ValueError(f"{task} passed, though a turn of it failed")
        if len(ok) != turns:
            raise ValueError(f"{task} passed with {len(ok)} of its {turns} turns")
        return
    if all(ok):
        raise ValueError(f"{task} failed, though every turn of it passed")
    if len(ok) > turns or ok.index(False) != len(ok) - 1:
        raise ValueError(f"{task} failed but did not stop at its first failed turn")


def mark_comparable(listed: Sequence[ListedRun], root: Path = ROOT) -> list[ListedRun]:
    """`listed`, with every counted run that is not on the latest counted run's setup marked `other_setup`,
    and what differs. After a fix the runs are repeated and both are listed (decision of 2026-11-05): the
    latest, which count, and the earlier ones, which do not. Those keep their results, for the comparison of
    setups (decision of 2026-11-12).

    The setup: the code (commits may differ only in `results/`, where run records are committed between runs),
    the prompts and tool descriptions, the settings, the model options, the task order's seed, the reference
    date, the Ollama version, and for runs of the same model its build. Runs of other models are compared with
    the latest run too, except for the build."""
    # A counted run always has its record (ListedRun checks it); the filter only tells the type checker.
    counted = [(run, run.record) for run in listed if run.state == "reported" and run.record is not None]
    if len(counted) < 2:
        return list(listed)
    latest, latest_record = counted[-1]
    latest_of_model = {record.model.model: run for run, record in counted}  # the last one of each model stays
    marked = []
    for run in listed:
        if run.state != "reported" or run is latest or run.record is None:
            marked.append(run)
            continue
        same_model = latest_of_model[run.record.model.model]
        why = _setup_differs(run.record, latest.folder, latest_record, same_model, root)
        marked.append(run.model_copy(update={"state": "other_setup", "reason": why}) if why else run)
    return marked


def group_setups(runs: Sequence[ListedRun], root: Path = ROOT) -> list[list[ListedRun]]:
    """The runs of one model, in the order they ran, cut into its setups (decision of 2026-11-12): a run that
    differs from the first run of the setup before it starts a new one. A setup is what `mark_comparable`
    says, the model's build included, on one golden set.

    Setups are cut where they change, so a setup that comes back after another is a new one: the runs of a
    phase come from one command at one commit, and a commit between them, even outside the code, cuts them.
    A run on a commit this repository does not have, or after one, starts a setup of its own, since what
    differs cannot be known. ValueError for runs of several models, or a run without its record."""
    groups: list[list[ListedRun]] = []
    for run in runs:
        if run.record is None:
            raise ValueError(f"{run.folder} has no record to compare")
        if groups and _same_setup(run.record, groups[-1][0], root):
            groups[-1].append(run)
        else:
            groups.append([run])
    if len(_models(runs)) > 1:
        raise ValueError(f"setups are cut for one model at a time, not for {', '.join(_models(runs))}")
    return groups


def _same_setup(record: RunRecord, first: ListedRun, root: Path) -> bool:
    """Whether a run is on the setup whose first run is `first`, of the same model: compared as the later of
    the two, so that git's messages name the commits in the order they ran."""
    earlier = first.record
    assert earlier is not None  # noqa: S101 - group_setups checked it
    if record.hashes.golden != earlier.hashes.golden:
        return False
    commits = {earlier.code.commit, record.code.commit}
    if len(commits) > 1 and not all(_has_commit(root, commit) for commit in commits):
        return False
    later = ListedRun(folder="", state="reported", record=record)
    return _setup_differs(earlier, "", record, later, root) is None


def _setup_differs(
    record: RunRecord, latest_folder: str, latest: RunRecord, same_model: ListedRun, root: Path
) -> str | None:
    """How a run's setup differs from that of `latest`, or None. ReportError when the two commits differ and
    this repository does not have that of `latest`; the comparison of setups checks both commits first."""
    for field, words in (
        ("retrieval_prompt", "retrieval prompt differs from that"),
        ("decide_prompt", "decision prompt differs from that"),
        ("tool_descriptions", "tool descriptions differ from those"),
    ):
        if getattr(record.hashes, field) != getattr(latest.hashes, field):
            return f"its {words} of {latest_folder}"
    for what, mine, theirs in (
        ("settings", record.settings, latest.settings),
        ("model options", record.model.options, latest.model.options),
    ):
        if changed := sorted(key for key in mine.keys() | theirs.keys() if mine.get(key) != theirs.get(key)):
            return f"its {what} differ from those of {latest_folder}: {', '.join(changed)}"
    if record.shuffle_seed != latest.shuffle_seed:
        return f"its task order's seed differs from that of {latest_folder}"
    if record.reference_date != latest.reference_date:
        return f"its reference date differs from that of {latest_folder}"
    if record.model.server_version != latest.model.server_version:
        return f"it ran on another Ollama version than {latest_folder}"
    if same_model.record is not None and record.model.digest != same_model.record.model.digest:
        return f"it ran another build of {record.model.model} than {same_model.folder}"
    if record.code.dirty != latest.code.dirty:  # uncommitted changes cannot be shown to match anything
        if record.code.dirty:
            return f"it ran from a checkout with uncommitted changes, {latest_folder} from a clean one"
        return f"it ran from a clean checkout, {latest_folder} from one with uncommitted changes"
    if record.code.commit != latest.code.commit:
        if not _has_commit(root, latest.code.commit):  # the setup cannot be known: report from another clone
            raise ReportError(
                f"{latest_folder} ran on {latest.code.commit[:12]}, which this repository does not have: "
                "fetch it, or report from a clone that has it"
            )
        if not _has_commit(root, record.code.commit):
            return f"its commit {record.code.commit[:12]} is not in this repository"
        done = _git(
            root,
            "diff",
            "--quiet",
            record.code.commit,
            latest.code.commit,
            "--",
            ".",
            f":(exclude){OUTPUT_FOLDER}",
        )
        if done.returncode == 1:
            return f"its code differs from that of {latest_folder} outside {OUTPUT_FOLDER}/"
        if done.returncode != 0:
            said = " ".join(done.stderr.split())[:200]
            raise ReportError(
                f"git cannot compare {record.code.commit[:12]} and {latest.code.commit[:12]} in "
                f"{root.name}: {said}"
            )
    return None


def _has_commit(root: Path, sha: str) -> bool:
    return _git(root, "cat-file", "-e", f"{sha}^{{commit}}").returncode == 0


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    git = shutil.which("git")
    if git is None:
        raise ReportError("git is not installed: the report checks that the runs it compares ran one code")
    try:
        return subprocess.run(  # noqa: S603 - git's full path and fixed arguments, no shell
            [git, "-C", str(root), *arguments], capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as err:
        raise ReportError(f"git {arguments[0]} could not run in {root.name}: {type(err).__name__}") from err


def _named_split(folder: Path) -> str | None:
    """The split a run.json that does not validate still names, if it parses as JSON and names one."""
    try:
        named = json.loads((folder / RUN_FILE).read_text(encoding="utf-8")).get("split")
    except (OSError, ValueError, AttributeError):  # unreadable, not JSON, or not a JSON object
        return None
    return named if isinstance(named, str) else None


def _unreadable(err: RecordError) -> str:
    """Why a run.json cannot be read, without the paths of this machine."""
    cause = err.__cause__
    if isinstance(cause, ValidationError):
        first = cause.errors()[0]
        where = ".".join(str(part) for part in first["loc"]) or "the top"
        return (
            f"{RUN_FILE} is not a run record this version reads: {_count(cause.error_count(), 'problem')}, "
            f"the first at {where}: {first['msg']}"
        )
    if isinstance(cause, UnicodeDecodeError):
        return f"{RUN_FILE} is not UTF-8 text"
    strerror = getattr(cause, "strerror", None)  # read_run_record wraps an OSError otherwise
    return f"{RUN_FILE} cannot be read" + (f" ({strerror})" if strerror else "")


def _attempts_unreadable(err: RecordError) -> str:
    """Why attempts.jsonl cannot be read: never the text of the line, which can quote a task or an answer."""
    cause = err.__cause__
    if isinstance(cause, ValidationError):
        first = cause.errors()[0]
        where = ".".join(str(part) for part in first["loc"]) or "the top"
        return (
            f"line {err.line} of {ATTEMPTS_FILE} is not an attempt record this version reads: "
            f"{_count(cause.error_count(), 'problem')}, the first at {where}: {first['msg']}"
        )
    if isinstance(cause, UnicodeDecodeError):
        return f"{ATTEMPTS_FILE} is not UTF-8 text"
    strerror = getattr(cause, "strerror", None)
    return f"{ATTEMPTS_FILE} cannot be read" + (f" ({strerror})" if strerror else "")


def _counts(counts: TaskCounts) -> str:
    return ", ".join(f"{getattr(counts, what)} {what}" for what in COUNTED)


# ---------------------------------------------------------------------------------------------------------
# The numbers (docs/evaluation.md, "Scoring" and "Runs and statistics"; decisions of 2026-11-02 to 2026-11-07)
# ---------------------------------------------------------------------------------------------------------


class Rate(BaseModel):
    """`part` of `whole`, as counted; no share is made up when `whole` is 0."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    part: int = Field(ge=0)
    whole: int = Field(ge=0)

    def __str__(self) -> str:
        return (
            f"{self.part} of {self.whole} ({self.part / self.whole:.0%})" if self.whole else "none to count"
        )


class Mean(BaseModel):
    """A mean over the turns that have the value."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: float
    turns: int = Field(ge=1)


class CategoryShare(BaseModel):
    """Passed tasks of one category, lost runs left out: descriptive only, too few tasks for an interval."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: float
    with_result: int
    tasks: int


class Cost(BaseModel):
    """What a task took, over the scored last attempts: the runner's clock, the model server's, and tokens."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tasks: int
    median_seconds: float
    p90_seconds: float
    median_model_seconds: float
    median_load_seconds: float
    median_input_tokens: float
    median_output_tokens: float


class RunSeconds(BaseModel):
    """One run's time over its scored attempts: the runner's clock and the model server's."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tasks: int
    seconds: float
    model_seconds: float


class ModelSummary(BaseModel):
    """Everything the report says about one model, over its runs that count."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    digest: str | None
    server_versions: list[str]
    runs: list[str]
    run_counts: list[TaskCounts]
    warm_up_seconds: list[float | None]  # each run's warm-up after its fresh load; None before it existed
    run_seconds: list[RunSeconds]  # each run's time, so that a drift from run to run shows
    tasks: int
    with_result: int  # tasks with a result in at least one run
    lost_excluded: Interval | None  # passed tasks, lost runs left out; None when no task has a result
    lost_failed: Interval  # passed tasks, lost runs counted as failed
    stability: float
    per_task: dict[str, list[Result]]
    per_category: dict[str, CategoryShare]
    last_call: Rate  # search turns whose last search call had every argument right
    first_call: Rate  # ... and whose first one had: the difference is self-correction
    over_clarification: Rate  # search turns that ended asking instead
    over_refusal: Rate  # search turns that ended declining instead
    precision: (
        Mean | None
    )  # the mean over turns of each turn's precision: a turn citing many scenes weighs one
    recall: Mean | None
    oracle_failures: int  # turns whose catalogue query failed: left out of precision and recall
    temperature: list[float | None]  # as the model options record it; None where they do not
    grounding_failures: int
    model_failures: dict[str, int]
    errored: list[str]  # "task (run n)", n the run's column: a bug of ours, counted as failed
    infrastructure_attempts: int
    lost: int
    unattributed_calls: int
    cost: Cost | None


class Comparison(BaseModel):
    """Two models on the tasks both have a majority result for: an exact sign test on those they split."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    first: str
    second: str
    first_only: int
    second_only: int
    both_passed: int
    both_failed: int
    left_out: int  # tasks where either model has no majority result
    p_value: float


class SetupSummary(BaseModel):
    """One setup of a model: its runs, in the order they ran, numbered from the model's first setup."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    number: int = Field(ge=1)
    golden: str | None  # the tag of an earlier golden set; None for the one the report reads
    runs: list[str]
    differs: str | None  # how it differs from the model's setup before it


class SetupChange(BaseModel):
    """One model's tasks in two consecutive setups on the golden set the report reads."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    before: int  # setup numbers
    after: int
    change: bool  # what the agents are given or run differs; otherwise only their environment does
    first: bool  # the model's first change on this golden set: the only one whose new tasks had not been run
    rows: list[Change]  # all tasks, then, for the first change, the tasks of v1 and those added in v2
    periods: PeriodChange | None


class SessionNoise(BaseModel):
    """One model's runs on an earlier golden set against its first setup on this one, on shared tasks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    tag: str
    earlier: int  # setup numbers
    baseline: int | None
    refused: str | None = None  # why the two are not compared
    differences: list[str] = []  # what differs besides the golden set and the task order
    agents: AgentsCode = "same"  # whether the code the agents run is the same on both sides
    change: Change | None = None


class DeterminismRow(BaseModel):
    """The determinism of one model's runs in one setup, or across the two sides of a session-noise pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    label: str
    measure: Determinism


class Report(BaseModel):
    """The report on one split: every run listed, the numbers of the runs that count, and the comparison of
    setups."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    split: Split
    listed: list[ListedRun]
    models: list[ModelSummary]
    comparisons: list[Comparison]
    scratch: bool  # runs from a checkout with changes are in the numbers
    setups: list[SetupSummary] = []
    changes: list[SetupChange] = []
    session_noise: list[SessionNoise] = []
    determinism: list[DeterminismRow] = []


def build_report(
    split: Split, listed: Sequence[ListedRun], tasks: Sequence[Task], *, root: Path = ROOT
) -> Report:
    """The numbers of the runs that count, once the runs on another setup than the latest are marked so
    (`mark_comparable`)."""
    listed = mark_comparable(listed, root)
    # A counted run always has its record (ListedRun checks it); the filter only tells the type checker.
    counted = [(run, run.record) for run in listed if run.state == "reported" and run.record is not None]
    by_model: dict[str, list[tuple[ListedRun, RunRecord]]] = {}
    for run, record in counted:
        by_model.setdefault(record.model.model, []).append((run, record))
    models = [_model_summary(name, runs, tasks) for name, runs in by_model.items()]
    comparisons = [
        _compare(first, second, [task.id for task in tasks])
        for index, first in enumerate(models)
        for second in models[index + 1 :]
    ]
    compared = _compare_setups(listed, tasks, root)
    used = {folder for setup in compared["setups"] for folder in setup.runs}
    scratch = any(
        run.record is not None and run.record.code.dirty and (run.state == "reported" or run.folder in used)
        for run in listed
    )
    return Report(
        split=split, listed=list(listed), models=models, comparisons=comparisons, scratch=scratch, **compared
    )


def _compare_setups(listed: Sequence[ListedRun], tasks: Sequence[Task], root: Path) -> dict[str, Any]:
    """For each model: its setups, numbered in the order they ran; the comparison of its consecutive setups on
    this golden set; the session noise of its last setup on an earlier golden set against its first on this
    one; and the determinism of each (docs/evaluation.md, "Comparing two setups"). The fields of `Report`
    they fill."""
    current = [run for run in listed if run.state in ("reported", "other_setup") and run.record is not None]
    earlier = [run for run in listed if run.state == "earlier_golden_set" and run.record is not None]
    order = [task.id for task in tasks]
    found: dict[str, list[Any]] = {"setups": [], "changes": [], "session_noise": [], "determinism": []}
    for model in _models([*earlier, *current]):
        before, now = group_setups(_of(earlier, model), root), group_setups(_of(current, model), root)
        ordered = sorted([*before, *now], key=_started)
        number = {id(group): index for index, group in enumerate(ordered, start=1)}
        for index, group in enumerate(ordered):
            found["setups"].append(
                SetupSummary(
                    model=model,
                    number=number[id(group)],
                    golden=_tag_of(group) if group in before else None,
                    runs=[run.folder for run in group],
                    differs=_differs_from(ordered[index - 1], group, root) if index else None,
                )
            )
            if (measure := _determinism(group, order)) is not None:
                found["determinism"].append(
                    DeterminismRow(model=model, label=f"setup {number[id(group)]}", measure=measure)
                )
        changed_before = False
        for first, second in itertools.pairwise(now):
            pair = _setup_change(
                model,
                (number[id(first)], first),
                (number[id(second)], second),
                not changed_before,
                order,
                root,
            )
            changed_before = changed_before or pair.change
            found["changes"].append(pair)
        if before:
            baseline = (number[id(now[0])], now[0]) if now else None
            noise = _session_noise(model, (number[id(before[-1])], before[-1]), baseline, order, root)
            found["session_noise"].append(noise)
            if noise.change is not None and now:
                both = [*before[-1], *now[0]]
                shared = [task_id for task_id in order if all(task_id in run.results for run in both)]
                found["determinism"].append(
                    DeterminismRow(
                        model=model,
                        label=(
                            f"setups {noise.earlier} and {noise.baseline}, "
                            f"on the {len(shared)} tasks of {noise.tag}"
                        ),
                        measure=determinism([turn_actions(run.attempts) for run in both], shared),
                    )
                )
    return found


def _started(group: Sequence[ListedRun]) -> tuple[datetime, str]:
    """When a setup's first run started; its folder's name, which starts with that time, breaks a tie."""
    record = group[0].record
    assert record is not None  # noqa: S101 - see _tag_of
    return record.started_at, group[0].folder


def _models(runs: Sequence[ListedRun]) -> list[str]:
    """The models of the runs, in the order they first ran."""
    return list(dict.fromkeys(run.record.model.model for run in runs if run.record is not None))


def _of(group: Sequence[ListedRun], model: str) -> list[ListedRun]:
    return [run for run in group if run.record is not None and run.record.model.model == model]


def _tag_of(group: Sequence[ListedRun]) -> str:
    record = group[0].record
    assert record is not None  # noqa: S101 - group_setups takes only runs with a record
    hashes = (record.hashes.golden, record.hashes.evaluation_rules)
    return next(
        earlier.tag for earlier in EARLIER_GOLDEN_SETS if (earlier.golden, earlier.evaluation_rules) == hashes
    )


def _differs_from(before: Sequence[ListedRun], after: Sequence[ListedRun], root: Path) -> str:
    """How the first run of a model's setup differs from the first run of its setup before, said of the
    setup. Something does: `group_setups` cut there."""
    first, then = before[0], after[0]
    assert first.record is not None  # noqa: S101 - see _tag_of
    assert then.record is not None  # noqa: S101 - see _tag_of
    if then.record.hashes.golden != first.record.hashes.golden:
        return "another golden set"
    commits = {first.record.code.commit, then.record.code.commit}
    if len(commits) > 1 and not _has_commit(root, first.record.code.commit):
        return f"the commit of {first.folder} is not in this repository, so the two cannot be compared"
    why = _setup_differs(then.record, first.folder, first.record, first, root)
    if why is None:  # pragma: no cover - group_setups cut here, so something differs
        raise ValueError(f"{then.folder} starts a setup, though nothing differs from {first.folder}")
    return why


def _per_task(runs: Sequence[ListedRun]) -> dict[str, list[Result]]:
    """Each task's results over the runs that ran it."""
    per_task: dict[str, list[Result]] = {}
    for run in runs:
        for task_id, result in run.results.items():
            per_task.setdefault(task_id, []).append(result)
    return per_task


def _shared(order: Sequence[str], *sides: Mapping[str, Sequence[Result]]) -> list[str]:
    """The tasks every side ran, in the order of the split."""
    return [task_id for task_id in order if all(task_id in side for side in sides)]


def _setup_change(
    model: str,
    before: tuple[int, Sequence[ListedRun]],
    after: tuple[int, Sequence[ListedRun]],
    may_be_first: bool,
    order: Sequence[str],
    root: Path,
) -> SetupChange:
    """One model's two consecutive setups on this golden set: every run in them is the model's. A pair is a
    change when what the agents are given or run differs; otherwise only their environment does, and the
    pair shows noise. The model's first change is the one whose new tasks had not been run."""
    (number_before, runs_before), (number_after, runs_after) = before, after
    mine, theirs = runs_before[0].record, runs_after[0].record
    assert mine is not None  # noqa: S101 - see _tag_of
    assert theirs is not None  # noqa: S101 - see _tag_of
    is_change = _chosen_change(mine, theirs, root)
    first = may_be_first and is_change
    results_before, results_after = _per_task(runs_before), _per_task(runs_after)
    shared = _shared(order, results_before, results_after)
    rows = [change(f"All {len(shared)} tasks", shared, results_before, results_after)]
    new = [task_id for task_id in shared if added_in_v2(task_id)]
    seen = [task_id for task_id in shared if not added_in_v2(task_id)]
    if first and new and seen:
        rows += [
            change(
                f"The {len(seen)} of v1, seen when the change was chosen", seen, results_before, results_after
            ),
            change(f"The {len(new)} added in v2", new, results_before, results_after),
        ]
    return SetupChange(
        model=model,
        before=number_before,
        after=number_after,
        change=is_change,
        first=first,
        rows=rows,
        periods=period_change(shared, results_before, results_after),
    )


def _chosen_change(mine: RunRecord, theirs: RunRecord, root: Path) -> bool:
    """Whether what the agents are given or run differs between two runs: the prompts, the tool descriptions,
    the settings, the model options, or the code the agents run. When that code cannot be known (a checkout
    with changes, a commit this repository does not have), it may have: a change, to be safe. The Ollama
    version, the model's build, the libraries, the task order's seed, the reference date and the code outside
    the agents are their environment."""
    if (
        (mine.hashes.retrieval_prompt, mine.hashes.decide_prompt, mine.hashes.tool_descriptions)
        != (theirs.hashes.retrieval_prompt, theirs.hashes.decide_prompt, theirs.hashes.tool_descriptions)
        or mine.settings != theirs.settings
        or mine.model.options != theirs.model.options
        or mine.code.dirty
        or theirs.code.dirty
    ):
        return True
    if mine.code.commit == theirs.code.commit:
        return False
    return _code_difference(mine.code.commit, theirs.code.commit, root)[1] != "same"


def _session_noise(
    model: str,
    earlier: tuple[int, Sequence[ListedRun]],
    baseline: tuple[int, Sequence[ListedRun]] | None,
    order: Sequence[str],
    root: Path,
) -> SessionNoise:
    """A model's last setup on an earlier golden set against its first on this one. Refused when what the
    model was asked, or the model itself, differs: the prompts, the tool descriptions, the model options or
    its build. Everything else that differs is listed."""
    number, runs_before = earlier
    pair = SessionNoise(model=model, tag=_tag_of(runs_before), earlier=number, baseline=None)
    if baseline is None:
        return pair.model_copy(update={"refused": f"{model} has no setup on this golden set yet"})
    baseline_number, runs_after = baseline
    pair = pair.model_copy(update={"baseline": baseline_number})
    mine, theirs = runs_before[0].record, runs_after[0].record
    assert mine is not None  # noqa: S101 - see _tag_of
    assert theirs is not None  # noqa: S101 - see _tag_of
    for differs, words in (
        (mine.hashes.retrieval_prompt != theirs.hashes.retrieval_prompt, "the retrieval prompt differs"),
        (mine.hashes.decide_prompt != theirs.hashes.decide_prompt, "the decision prompt differs"),
        (mine.hashes.tool_descriptions != theirs.hashes.tool_descriptions, "the tool descriptions differ"),
        (mine.model.options != theirs.model.options, "the model options differ"),
        (mine.model.digest != theirs.model.digest, "the model's build differs"),
    ):
        if differs:
            return pair.model_copy(
                update={"refused": f"{words}, and the comparison is for the same model, asked the same way"}
            )
    results_before, results_after = _per_task(runs_before), _per_task(runs_after)
    shared = _shared(order, results_before, results_after)
    differences, agents = _what_else_differs(mine, theirs, root)
    return pair.model_copy(
        update={
            "differences": differences,
            "agents": agents,
            "change": change(f"The {len(shared)} tasks of {pair.tag}", shared, results_before, results_after),
        }
    )


def _what_else_differs(mine: RunRecord, theirs: RunRecord, root: Path) -> tuple[list[str], AgentsCode]:
    """What the run records say differs between the runs of a session-noise pair, besides the golden set and
    what refuses the comparison; and whether the code the agents run is the same, differs, or cannot be
    known."""
    found = []
    for what, first, second in (
        ("settings", mine.settings, theirs.settings),
        ("libraries", mine.libraries, theirs.libraries),
    ):
        if keys := sorted(key for key in first.keys() | second.keys() if first.get(key) != second.get(key)):
            found.append(f"{what}: {', '.join(keys)}")
    for differs, said in (
        (mine.shuffle_seed != theirs.shuffle_seed, "the task order's seed"),
        (mine.reference_date != theirs.reference_date, "the reference date"),
        (
            mine.model.server_version != theirs.model.server_version,
            f"Ollama {mine.model.server_version} and {theirs.model.server_version}",
        ),
        (mine.model.provider != theirs.model.provider, "the model's provider"),
        (mine.platform != theirs.platform, "the platform"),
        (mine.code.dirty or theirs.code.dirty, "a checkout with uncommitted changes"),
    ):
        if differs:
            found.append(said)
    agents: AgentsCode = "unknown" if mine.code.dirty or theirs.code.dirty else "same"
    if mine.code.commit != theirs.code.commit:
        said, code = _code_difference(mine.code.commit, theirs.code.commit, root)
        found.append(said)
        agents = "unknown" if "unknown" in (agents, code) else code
    return found, agents


def _code_difference(first: str, second: str, root: Path) -> tuple[str, AgentsCode]:
    """The files that differ between two commits outside `results/`, those of the code the agents run (under
    `src/`, outside `evals/`) named; and whether that code is the same, differs, or cannot be known."""
    if not (_has_commit(root, first) and _has_commit(root, second)):
        return f"the code: commit {first[:12]} or {second[:12]} is not in this repository", "unknown"
    done = _git(root, "diff", "--name-only", first, second, "--", ".", f":(exclude){OUTPUT_FOLDER}")
    if done.returncode != 0:
        said = " ".join(done.stderr.split())[:200]
        raise ReportError(f"git cannot compare {first[:12]} and {second[:12]} in {root.name}: {said}")
    files = done.stdout.split()
    system = [name for name in files if name.startswith(SYSTEM_CODE) and not name.startswith(EVALUATION_CODE)]
    named = f": {', '.join(system)}" if system else ""
    said = (
        f"the code: {_count(len(files), 'file')} outside {OUTPUT_FOLDER}/, "
        f"{len(system)} of them in the code the agents run{named}"
    )
    return said, "differs" if system else "same"


def _determinism(runs: Sequence[ListedRun], order: Sequence[str]) -> Determinism | None:
    """The determinism of a setup's runs over the tasks in `order`; None for a single run."""
    if len(runs) < 2:
        return None
    actions: list[dict[TurnKey, Actions]] = [turn_actions(run.attempts) for run in runs]
    return determinism(actions, order)


def _model_summary(
    name: str, counted: Sequence[tuple[ListedRun, RunRecord]], tasks: Sequence[Task]
) -> ModelSummary:
    runs = [run for run, _ in counted]
    records = [record for _, record in counted]  # one build of the model: mark_comparable saw to it
    per_task = {task.id: [run.results[task.id] for run in runs] for task in tasks}
    results = list(per_task.values())
    with_result = [runs_of for runs_of in results if task_share(runs_of, lost="excluded") is not None]
    finals = [turn for run in runs for turn in _scored_turns(run)]
    search = [turn for turn in finals if turn.score is not None and turn.score.arguments is not None]
    return ModelSummary(
        model=name,
        digest=records[0].model.digest,
        server_versions=sorted({record.model.server_version or "unknown" for record in records}),
        runs=[run.folder for run in runs],
        run_counts=[record.tasks for record in records],
        warm_up_seconds=[record.warm_up_seconds for record in records],
        run_seconds=[_run_seconds(_scored_last_attempts(run)) for run in runs],
        tasks=len(tasks),
        with_result=len(with_result),
        lost_excluded=bootstrap_interval(with_result, _passed_left_out) if with_result else None,
        lost_failed=bootstrap_interval(results, _passed_counted_failed),
        stability=stability(results),
        per_task=per_task,
        per_category=_per_category(tasks, per_task),
        last_call=_rate(search, lambda turn: _all_right(turn.score.arguments if turn.score else None)),
        first_call=_rate(search, lambda turn: _all_right(turn.score.first_arguments if turn.score else None)),
        over_clarification=_rate(
            search, lambda turn: turn.output is not None and turn.output.status == "clarified"
        ),
        over_refusal=_rate(search, lambda turn: turn.output is not None and turn.output.status == "declined"),
        precision=_mean([turn.oracle.precision for turn in finals if turn.oracle is not None]),
        recall=_mean([turn.oracle.recall for turn in finals if turn.oracle is not None]),
        oracle_failures=sum(
            1 for turn in finals if turn.oracle is not None and turn.oracle.failed is not None
        ),
        temperature=_temperatures(records),
        grounding_failures=sum(1 for turn in finals if turn.score is not None and not turn.score.grounded),
        model_failures=dict(
            sorted(
                Counter(turn.output.failure for turn in finals if turn.output and turn.output.failure).items()
            )
        ),
        errored=[
            f"{task.id} (run {position})"  # the column of the per-task table
            for position, run in enumerate(runs, start=1)
            for task in tasks
            if run.results[task.id] == "errored"
        ],
        infrastructure_attempts=sum(
            1 for run in runs for one in run.attempts if one.outcome == "infrastructure"
        ),
        lost=sum(1 for runs_of in results for result in runs_of if result == "lost"),
        unattributed_calls=sum(record.unattributed_calls for record in records),
        cost=_cost([attempt for run in runs for attempt in _scored_last_attempts(run)]),
    )


def _temperatures(records: Sequence[RunRecord]) -> list[float | None]:
    """The temperatures the runs' model options record, None for a run that records none."""
    found: set[float | None] = set()
    for record in records:
        value = record.model.options.get("temperature")
        found.add(float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None)
    return sorted(found, key=lambda value: (value is None, value or 0.0))


def _temperature_words(values: Sequence[float | None]) -> str:
    return ", ".join("not recorded" if value is None else f"{value:g}" for value in values)


def _passed_left_out(tasks: Sequence[Sequence[Result]]) -> float:
    return sum(task_share(runs, lost="excluded") or 0.0 for runs in tasks)


def _passed_counted_failed(tasks: Sequence[Sequence[Result]]) -> float:
    return sum(task_share(runs, lost="failed") or 0.0 for runs in tasks)


def _per_category(tasks: Sequence[Task], per_task: dict[str, list[Result]]) -> dict[str, CategoryShare]:
    categories: dict[str, list[Task]] = {}
    for task in tasks:
        categories.setdefault(str(task.category), []).append(task)
    shares = {}
    for category, members in categories.items():
        found = [
            share for task in members if (share := task_share(per_task[task.id], lost="excluded")) is not None
        ]
        shares[category] = CategoryShare(passed=sum(found), with_result=len(found), tasks=len(members))
    return shares


def _scored_last_attempts(run: ListedRun) -> list[AttemptRecord]:
    """The run's scored attempts: what the numbers of a turn are taken from. A scored attempt is always its
    task's last, since only an attempt lost to the infrastructure is tried again (task_results checks it)."""
    return [attempt for attempt in run.attempts if attempt.outcome == "scored"]


def _scored_turns(run: ListedRun) -> list[TurnRecord]:
    return [turn for attempt in _scored_last_attempts(run) for turn in attempt.turns]


def _all_right(arguments: ArgumentScore | None) -> bool:
    return arguments is not None and arguments.all


def _rate(turns: Sequence[TurnRecord], counts: Callable[[TurnRecord], bool]) -> Rate:
    return Rate(part=sum(1 for turn in turns if counts(turn)), whole=len(turns))


def _mean(values: Sequence[float | None]) -> Mean | None:
    present = [value for value in values if value is not None]
    return Mean(value=sum(present) / len(present), turns=len(present)) if present else None


def _usages(turn: TurnRecord) -> list[Usage]:
    """The model's usage in a turn: the decision's, and the retrieval agent's when it was called."""
    if turn.output is None:
        return []
    retrieval = [turn.output.retrieval.usage] if turn.output.retrieval is not None else []
    return [turn.output.decision_usage, *retrieval]


def _run_seconds(attempts: Sequence[AttemptRecord]) -> RunSeconds:
    turns = [turn for attempt in attempts for turn in attempt.turns]
    return RunSeconds(
        tasks=len(attempts),
        seconds=sum(turn.seconds for turn in turns),
        model_seconds=sum(usage.model_seconds for turn in turns for usage in _usages(turn)),
    )


def _cost(attempts: Sequence[AttemptRecord]) -> Cost | None:
    if not attempts:
        return None
    columns: dict[str, list[float]] = {"seconds": [], "model": [], "load": [], "input": [], "output": []}
    for attempt in attempts:
        usages = [usage for turn in attempt.turns for usage in _usages(turn)]
        columns["seconds"].append(sum(turn.seconds for turn in attempt.turns))
        columns["model"].append(sum(usage.model_seconds for usage in usages))
        columns["load"].append(sum(usage.load_seconds for usage in usages))
        columns["input"].append(sum(usage.input_tokens for usage in usages))
        columns["output"].append(sum(usage.output_tokens for usage in usages))
    ordered = {name: sorted(values) for name, values in columns.items()}
    return Cost(
        tasks=len(attempts),
        median_seconds=quantile(ordered["seconds"], 0.5),
        p90_seconds=quantile(ordered["seconds"], 0.9),
        median_model_seconds=quantile(ordered["model"], 0.5),
        median_load_seconds=quantile(ordered["load"], 0.5),
        median_input_tokens=quantile(ordered["input"], 0.5),
        median_output_tokens=quantile(ordered["output"], 0.5),
    )


def _compare(first: ModelSummary, second: ModelSummary, task_ids: Sequence[str]) -> Comparison:
    tally: Counter[str] = Counter()
    for task_id in task_ids:
        a, b = majority(first.per_task[task_id]), majority(second.per_task[task_id])
        if a is None or b is None:
            tally["left_out"] += 1
        elif a == b:
            tally[f"both_{a}"] += 1
        else:
            tally["first_only" if a == "passed" else "second_only"] += 1
    return Comparison(
        first=first.model,
        second=second.model,
        first_only=tally["first_only"],
        second_only=tally["second_only"],
        both_passed=tally["both_passed"],
        both_failed=tally["both_failed"],
        left_out=tally["left_out"],
        p_value=exact_sign_test(tally["first_only"], tally["second_only"]),
    )


# ---------------------------------------------------------------------------------------------------------
# The markdown
# ---------------------------------------------------------------------------------------------------------

STATE_WORDS: dict[RunState, str] = {
    "did_not_start": "did not start",
    "unreadable": "unreadable",
    "running": "crashed or still running",
    "aborted": "aborted",
    "dirty": "dirty checkout",
    "other_golden_set": "other golden set",
    "earlier_golden_set": "earlier golden set",
    "inconsistent": "inconsistent",
    "other_setup": "other setup",
    "reported": "counted",
}
RESULT_WORDS: dict[Result, str] = {
    "passed": "passed",
    "failed": "failed",
    "errored": "errored",
    "lost": "lost",
}


def render_markdown(report: Report) -> str:
    """The report as markdown. It names task ids and categories, never a task's text."""
    lines = [f"# Evaluation report: {report.split} split", ""]
    if report.scratch:
        lines += ["> Runs from a checkout with changes are in these numbers: not for publication.", ""]
    counted = [run.record for run in report.listed if run.state == "reported" and run.record is not None]
    if counted:
        first = counted[0]
        commits = sorted({record.code.commit[:12] for record in counted})
        on = (
            f"on commit {commits[0]}"
            if len(commits) == 1
            else f"on commits {', '.join(commits)}, which differ only in `{OUTPUT_FOLDER}/`"
        )
        lines += [
            f"Made from {_count(len(counted), 'run')} that {'counts' if len(counted) == 1 else 'count'}, of "
            f"{len(report.listed)} listed, {on}.",
            f"Reference date {first.reference_date}. Frozen files: golden set {first.hashes.golden[:12]}, "
            f"evaluation code {first.hashes.evaluation_code[:12]}, gazetteer {first.hashes.gazetteer[:12]}, "
            f"docs/evaluation.md {first.hashes.evaluation_rules[:12]}.",
            "",
        ]
    else:
        lines += ["No run counts yet.", ""]
    lines += _runs_table(report.listed)
    if report.models:
        lines += _headline_table(report.models)
        if report.comparisons:
            lines += _comparison_table(report.comparisons, report.models[0].tasks)
    if report.changes or report.session_noise:
        lines += _setups_section(report)
    if report.determinism:
        lines += _determinism_table(report.determinism)
    if report.models:
        for summary in report.models:
            lines += _model_section(summary)
        lines += _notes(report)
    return "\n".join(lines).rstrip("\n") + "\n"


def _setups_section(report: Report) -> list[str]:
    counted = (
        " The numbers above count the runs marked counted in the table of every run: those on the setup of "
        "the latest run, also when a run on another setup came between them."
        if report.models
        else ""
    )
    lines = [
        "## Between setups",
        "",
        "A setup is a set of a model's runs, one after another, with the same code, prompts, tool "
        f"descriptions and model options.{counted} A task's result in a setup is the "
        "majority of the model's runs there, among the runs with a result; a task without one is left out, "
        "and counted. The exact sign test is on the tasks gained against those lost (docs/evaluation.md, "
        '"Comparing two setups").',
        "",
        "| Model | Setup | Golden set | Runs | What differs from the model's setup before |",
        "|---|---|---|---|---|",
    ]
    for setup in report.setups:
        lines.append(
            f"| {_cell(setup.model)} | {setup.number} | {_cell(setup.golden or 'this one')} | "
            f"{_cell(', '.join(setup.runs))} | {_cell(setup.differs or '')} |"
        )
    lines.append("")
    for pair in report.changes:
        lines += _change_section(pair)
    for noise in report.session_noise:
        lines += _noise_section(noise)
    return lines


CHANGE_HEADER = [
    "| Tasks | Passed in both | Failed in both | Gained | Lost | Left out (no majority) | p |",
    "|---|---|---|---|---|---|---|",
]


def _change_rows(rows: Sequence[Change]) -> list[str]:
    lines = [*CHANGE_HEADER]
    for row in rows:
        lines.append(
            f"| {_cell(row.scope)} | {row.both_passed} | {row.both_failed} | {len(row.gained)} | "
            f"{len(row.lost)} | {row.left_out} | {_p(row.p_value)} |"
        )
    lines.append("")
    whole = rows[0]
    lines += [
        f"- Gained: {_cell(', '.join(whole.gained)) or 'none'}.",
        f"- Lost: {_cell(', '.join(whole.lost)) or 'none'}.",
        "",
    ]
    return lines


def _change_section(pair: SetupChange) -> list[str]:
    lines = [f"### {_cell(pair.model)}: setup {pair.before}, then setup {pair.after}", ""]
    if not pair.change:
        lines += [
            "Nothing the agents are given or run differs between these two setups, only their environment "
            "(see the table of setups), so this shows noise, not the effect of a change.",
            "",
        ]
    lines += _change_rows(pair.rows)
    if pair.change and not pair.first:
        lines += [
            "For a change after the model's first on this golden set, every task had been run when it was "
            "chosen, so the split into seen and new tasks says nothing and is not shown.",
            "",
        ]
    if pair.periods is not None:
        lines += _periods_table(pair.periods)
    return lines


def _periods_table(periods: PeriodChange) -> list[str]:
    lines = [
        "| Relative period | Tasks | Passed before | Passed after | Period before | Period after |",
        "|---|---|---|---|---|---|",
    ]
    for result in periods.periods:
        lines.append(
            f"| {result.period} | {result.tasks} | {result.passed_before} | {result.passed_after} | "
            f"{result.before} | {result.after} |"
        )
    left_out = len(periods.periods) - periods.compared
    lines += [
        "",
        f"A period passes when all its tasks pass; one with a task left out is left out "
        f"({_count(left_out, 'period')} here). Sign test over the {_count(periods.compared, 'period')} "
        f"with a result on both sides: {periods.gained} gained, {periods.lost} lost, "
        f"p = {_p(periods.p_value)}. With {_count(periods.compared, 'period')} it cannot go below "
        f"{periods.floor:g}, so nothing is called significant on its strength.",
        "",
    ]
    return lines


def _noise_section(noise: SessionNoise) -> list[str]:
    against = f"setup {noise.baseline}" if noise.baseline is not None else "a setup on this golden set"
    lines = [
        f"### Session noise, {_cell(noise.model)}: setup {noise.earlier} ({noise.tag}) against {against}",
        "",
    ]
    if noise.refused is not None or noise.change is None:
        return [*lines, f"Not compared: {_cell(noise.refused or 'nothing to compare')}.", ""]
    differences = "; ".join(noise.differences) or "nothing else that the run records name"
    same = {
        "same": "A difference here is not the effect of a change to them, to the model or to the code the "
        "agents run, which are the same on both sides.",
        "differs": "The code the agents run differs too, so a difference here may also come from it.",
        "unknown": "Whether the code the agents run is the same cannot be told, so a difference here may "
        "also come from it.",
    }[noise.agents]
    lines += [
        f"The tasks of {noise.tag} in both setups, with the same prompts, tool descriptions, model options "
        f"and build. {same} It shows what the session, the task order among more tasks and the live "
        f"catalogue can move. What else differs: {_cell(differences)}.",
        "",
    ]
    return [*lines, *_change_rows([noise.change])]


def _determinism_table(rows: Sequence[DeterminismRow]) -> list[str]:
    lines = [
        "## Determinism",
        "",
        "Over the turns that every run reached in a scored attempt: the share where every run made the same "
        "tool calls (tools, arguments, order), and the share where every run also cited the same scenes.",
        "",
        "| Model | Runs of | Runs | Turns every run reached | Same tool calls | Also the same scenes |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        measure = row.measure
        lines.append(
            f"| {_cell(row.model)} | {_cell(row.label)} | {measure.runs} | {measure.turns} | "
            f"{Rate(part=measure.same_calls, whole=measure.turns)} | "
            f"{Rate(part=measure.same_scenes, whole=measure.turns)} |"
        )
    return [*lines, ""]


def _runs_table(listed: Sequence[ListedRun]) -> list[str]:
    lines = [
        "## Every run",
        "",
        "| Run | State | Why it does not count | Model | Passed | Failed | Errored | Lost | "
        "Unattributed calls |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for run in listed:
        record = run.record
        counts = record.tasks if record is not None else None
        cells = [
            _cell(run.folder),
            STATE_WORDS[run.state],
            _cell(run.reason or ""),
            _cell(record.model.model) if record is not None else "",
            *([str(getattr(counts, what)) for what in COUNTED] if counts is not None else [""] * 4),
            str(record.unattributed_calls) if record is not None else "",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return [*lines, ""]


def _headline_table(models: Sequence[ModelSummary]) -> list[str]:
    lines = [
        "## Passed tasks",
        "",
        "For each task, the share of its runs that passed, summed over the tasks: with lost runs left out "
        "(out of the tasks with a result) and counted as failed (out of all tasks). A run is lost when every "
        "attempt fell to the infrastructure. The 95% interval resamples whole tasks.",
        "",
        "| Model | Lost tasks left out | Lost tasks counted as failed | Same result in every run | Runs |",
        "|---|---|---|---|---|",
    ]
    for summary in models:
        left_out = (
            _interval(summary.lost_excluded, summary.with_result)
            if summary.lost_excluded is not None
            else "none"
        )
        lines.append(
            f"| {_cell(summary.model)} | {left_out} | {_interval(summary.lost_failed, summary.tasks)} | "
            f"{_stability(summary)} | {len(summary.runs)} |"
        )
    return [*lines, ""]


def _comparison_table(comparisons: Sequence[Comparison], tasks: int) -> list[str]:
    noise = (
        "differences of less than about 16 to 20 points (8 to 10 tasks) can be noise (docs/evaluation.md)"
        if tasks == TEST_SPLIT_TASKS
        else f"with {_count(tasks, 'task')} the intervals are wide, so read any difference with care"
    )
    lines = [
        "## Between models",
        "",
        "A task's result for a model is the majority of its runs; a task without one is left out. The exact "
        f"sign test uses the tasks only one of the two passed. {_count(len(comparisons), 'comparison')}; "
        f"{noise}.",
        "",
        "| First | Second | Only the first passed | Only the second passed | Left out (no majority) | p | "
        "Both passed | Both failed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for pair in comparisons:
        lines.append(
            f"| {_cell(pair.first)} | {_cell(pair.second)} | {pair.first_only} | {pair.second_only} | "
            f"{pair.left_out} | {_p(pair.p_value)} | {pair.both_passed} | {pair.both_failed} |"
        )
    return [*lines, ""]


def _model_section(summary: ModelSummary) -> list[str]:
    runs = len(summary.runs)
    lines = [
        f"## {_cell(summary.model)}",
        "",
        f"Build {_cell(summary.digest or 'unknown')}, Ollama {_cell(', '.join(summary.server_versions))}, "
        f"temperature {_temperature_words(summary.temperature)}; {_count(runs, 'run')}"
        + ("" if runs == RUNS_PER_MODEL else f", where the rules call for {RUNS_PER_MODEL}")
        + f": {_cell(', '.join(summary.runs))}.",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Last search call with every argument right | {summary.last_call} |",
        f"| First search call with every argument right | {summary.first_call} |",
        f"| Asked instead of searching | {summary.over_clarification} |",
        f"| Declined instead of searching | {summary.over_refusal} |",
        f"| Scene precision, the mean over turns | {_mean_text(summary.precision)} |",
        f"| Recall, the mean over turns | {_mean_text(summary.recall)} |",
        f"| Catalogue queries for them that failed (turns left out) | {summary.oracle_failures} |",
        f"| Answers citing a scene no tool returned | {summary.grounding_failures} |",
        "| Model failures | "
        + (", ".join(f"{kind} {count}" for kind, count in summary.model_failures.items()) or "none")
        + " |",
        "| Errored tasks (a bug of ours, counted as failed) | "
        f"{_cell(', '.join(summary.errored)) or 'none'} |",
        f"| Attempts lost to the infrastructure (retried) | {summary.infrastructure_attempts} |",
        f"| Tasks lost in every attempt | {summary.lost} |",
        f"| Tool calls under no trace id of the run | {summary.unattributed_calls} |",
        "| Warm-up after each run's fresh load of the model, seconds | "
        f"{_warm_ups(summary.warm_up_seconds)} |",
        "| Seconds in turns per run, over its scored tasks | "
        + ", ".join(f"{run.seconds:.1f} ({_count(run.tasks, 'task')})" for run in summary.run_seconds)
        + " |",
        "| Model seconds per run, over its scored tasks | "
        + ", ".join(f"{run.model_seconds:.1f}" for run in summary.run_seconds)
        + " |",
    ]
    if summary.cost is not None:
        cost = summary.cost
        lines += [
            f"| Seconds per task, median and 90th percentile | {cost.median_seconds:.1f} and "
            f"{cost.p90_seconds:.1f} |",
            f"| Model seconds and load seconds per task, median | {cost.median_model_seconds:.1f} and "
            f"{cost.median_load_seconds:.2f} |",
            f"| Input and output tokens per task, median | {cost.median_input_tokens:.0f} and "
            f"{cost.median_output_tokens:.0f} |",
        ]
    lines += ["", "| Category | Passed, lost runs left out |", "|---|---|"]
    for category, share in summary.per_category.items():
        lines.append(
            f"| {_cell(category)} | {share.passed:.2f} of {share.with_result} "
            f"(of {_count(share.tasks, 'task')}) |"
        )
    runs_header = " | ".join(f"Run {index}" for index in range(1, len(summary.runs) + 1))
    lines += ["", f"| Task | {runs_header} |", "|---|" + "---|" * len(summary.runs)]
    for task_id, results in summary.per_task.items():
        lines.append(f"| {_cell(task_id)} | " + " | ".join(RESULT_WORDS[result] for result in results) + " |")
    return [*lines, ""]


def _notes(report: Report) -> list[str]:
    tasks = report.models[0].tasks
    temperatures = sorted(
        {value for summary in report.models for value in summary.temperature},
        key=lambda value: (value is None, value or 0.0),
    )
    grounding = sum(summary.grounding_failures for summary in report.models)
    return [
        "## Reading these numbers",
        "",
        f"- With {_count(tasks, 'task')}, one task is {100 / tasks:.1f} points. Per-category numbers are "
        "descriptive only.",
        f"- The runs were at temperature {_temperature_words(temperatures)}"
        + (
            ": at 0 a model's runs are nearly identical, so they show stability, not more evidence."
            if temperatures == [0.0]
            else "."
        ),
        "- Infrastructure failures are retried and excluded; errored tasks are a bug of ours and count as "
        "failed.",
        "- An attempt stops at its first failed turn, so the measures per turn and the cost of a task cover "
        "the turns that ran: a follow-up failed at its first turn adds one turn and a shorter task.",
        (
            "- Answers are built from templates and cite only scene ids the tools returned, so grounding is "
            "a property of the design: no answer cited another."
            if grounding == 0
            else f"- {_count(grounding, 'answer')} cited a scene no tool returned, which the templates "
            "should prevent: a bug to look into before reading the rest."
        ),
        "- Rules and limitations: `docs/evaluation.md` and `docs/decisions.md`.",
        "",
    ]


def _warm_ups(seconds: Sequence[float | None]) -> str:
    return ", ".join("not done" if value is None else f"{value:.1f}" for value in seconds)


def _p(value: float) -> str:
    """A p-value with two significant digits, never rounded to 0, and a tie rounded up (0.625 is 0.63):
    Python's own formatting rounds an exact tie to the even digit, which is not how a p-value is read."""
    if value < 0.001:
        return "<0.001"
    exact = Decimal(str(value))
    rounded = exact.quantize(Decimal(10) ** (exact.adjusted() - 1), rounding=ROUND_HALF_UP)
    return format(rounded.normalize(), "f")


def _stability(summary: ModelSummary) -> str:
    """The share of tasks with the same result in every run: nothing to say of a single run."""
    return f"{summary.stability:.0%}" if len(summary.runs) > 1 else "one run only"


def _interval(interval: Interval, out_of: int) -> str:
    return f"{interval.estimate:.1f} of {out_of} (95% interval {interval.low:.1f} to {interval.high:.1f})"


def _mean_text(mean: Mean | None) -> str:
    return f"{mean.value:.2f} over {_count(mean.turns, 'turn')}" if mean is not None else "none to count"


def _count(number: int, noun: str) -> str:
    """'1 run', '2 runs'."""
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _cell(text: str) -> str:
    """Text for one table cell: on one line, and without a column separator."""
    return " ".join(text.split()).replace("|", "/")
