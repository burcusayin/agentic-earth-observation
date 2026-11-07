"""The report on a split's runs (docs/evaluation.md, "Runs and statistics").

This first part reads the run folders: which runs count, and what each task came to in them. Every run is
listed, and one that does not count says why: reports cover every run, failed ones included (decision of
2026-11-05). Only a run that ended `complete`, from a clean checkout, on the frozen files and the split's
tasks, with a record that adds up, is in the numbers.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agentic_earth_observation.config import ROOT
from agentic_earth_observation.evals.golden import Task
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

RunState = Literal[
    "did_not_start",
    "unreadable",
    "running",
    "aborted",
    "dirty",
    "other_golden_set",
    "inconsistent",
    "reported",
]
COUNTED: tuple[Result, ...] = ("passed", "failed", "errored", "lost")  # the order counts are written in


class ReportError(Exception):
    """The report cannot be made as asked. The message says what to do."""


class ListedRun(BaseModel):
    """A run as the report lists it: its folder's name, whether it counts, and why not."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    folder: str
    state: RunState
    reason: str | None = None
    record: RunRecord | None = None
    results: dict[str, Result] = {}  # task id -> what it came to; only for a reported run
    attempts: list[AttemptRecord] = []  # only for a reported run

    @model_validator(mode="after")
    def _counted_runs_carry_their_record(self) -> Self:
        if self.state == "reported" and self.record is None:
            raise ValueError("a run that counts carries its record")
        return self


def collect_runs(
    runs_dir: Path, split: Split, task_ids: Sequence[str], *, golden: str, allow_dirty: bool = False
) -> list[ListedRun]:
    """Every run of `split` under `runs_dir`, in the order they started (their folder names sort so).

    `task_ids` are the split's tasks and `golden` the hash of the golden set the report reads: a run on
    another golden set is listed, not counted. `allow_dirty` counts runs from a checkout with changes, for
    scratch use.
    A folder without a record cannot say its split, so it is listed whatever the split.
    """
    if not runs_dir.exists():
        return []
    if not runs_dir.is_dir():
        raise ReportError(f"{runs_dir.name} is not a folder: give the folder that holds the runs' folders")
    listed = []
    for folder in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        run = _listed(folder, split, task_ids, golden, allow_dirty)
        if run is not None:
            listed.append(run)
    return listed


def _listed(
    folder: Path, split: Split, task_ids: Sequence[str], golden: str, allow_dirty: bool
) -> ListedRun | None:
    """How the report lists one folder, or None for a run of the other split."""
    if not (folder / RUN_FILE).exists():
        return ListedRun(folder=folder.name, state="did_not_start", reason=f"it has no {RUN_FILE}")
    try:
        record = read_run_record(folder)
    except RecordError as err:
        if _named_split(folder) not in (split, None):
            return None
        return ListedRun(folder=folder.name, state="unreadable", reason=_unreadable(err))
    if record.split != split:
        return None

    def listed(state: RunState, reason: str | None) -> ListedRun:
        return ListedRun(folder=folder.name, state=state, reason=reason, record=record)

    if record.status == "running":
        return listed("running", "it crashed, or is still running")
    if record.status == "aborted":
        return listed("aborted", record.aborted_reason)
    if record.code.dirty and not allow_dirty:
        return listed("dirty", "the checkout had changes that were not committed")
    if changed := _other_frozen_files(record, golden):
        verb = "differs" if len(changed) == 1 else "differ"
        return listed("other_golden_set", f"{' and '.join(changed)} {verb} from what this report reads")
    if set(record.task_order) != set(task_ids):  # a record that names a task twice does not read at all
        return listed("other_golden_set", "its tasks are not the tasks of the split")
    try:
        attempts = read_attempts(folder)
        results = task_results(record, attempts)
    except RecordError as err:
        return listed("unreadable", f"{ATTEMPTS_FILE} cannot be read: {_without_folder(str(err), folder)}")
    except ValueError as err:
        return listed("inconsistent", str(err))
    tally = Counter(results.values())
    found = TaskCounts(**{what: tally[what] for what in COUNTED})
    if found != record.tasks:
        return listed(
            "inconsistent",
            f"{RUN_FILE} counts {_counts(record.tasks)}; {ATTEMPTS_FILE} gives {_counts(found)}",
        )
    return ListedRun(folder=folder.name, state="reported", record=record, results=results, attempts=attempts)


def _other_frozen_files(record: RunRecord, golden: str) -> list[str]:
    """The frozen files the run used that are not the ones this report reads."""
    expected = {
        "the golden set": (record.hashes.golden, golden),
        "the evaluation code": (record.hashes.evaluation_code, FROZEN_EVALUATION_CODE),
        "the gazetteer": (record.hashes.gazetteer, FROZEN_GAZETTEER),
        "the labelling rules": (record.hashes.evaluation_rules, FROZEN_EVALUATION_RULES),
    }
    return [name for name, (used, read) in expected.items() if used != read]


def task_results(record: RunRecord, attempts: Sequence[AttemptRecord]) -> dict[str, Result]:
    """What each task of a run came to, by its last attempt. ValueError when the attempts do not add up to a
    finished run: a task without attempts, an attempt of another run or task, attempts out of order, an
    attempt after the last, or a task that ended before its last attempt without being scored."""
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
            results[task_id] = "passed" if last.passed else "failed"
        elif last.outcome == "errored":
            results[task_id] = "errored"
        elif last.attempt == MAX_ATTEMPTS:
            results[task_id] = "lost"
        else:
            raise ValueError(f"{task_id} ended after {last.attempt} of {MAX_ATTEMPTS} attempts")
    return results


def check_same_code(runs: Sequence[ListedRun], root: Path = ROOT) -> None:
    """Refuse runs that did not run the same code. Run records are committed between invocations, which moves
    HEAD, so commits may differ in `results/` only; the prompts and the tool descriptions must be the same."""
    recorded = [(run.folder, run.record) for run in runs if run.record is not None]
    if len(recorded) < 2:
        return
    first_folder, first = recorded[0]
    for folder, record in recorded[1:]:
        for field in ("retrieval_prompt", "decide_prompt", "tool_descriptions"):
            if getattr(record.hashes, field) != getattr(first.hashes, field):
                raise ReportError(
                    f"{folder} differs from {first_folder} in {field}: runs compared must use the same "
                    "prompts and tools"
                )
        if record.code.commit != first.code.commit:
            _same_code(first_folder, first.code.commit, folder, record.code.commit, root)


def _same_code(first_folder: str, first: str, folder: str, commit: str, root: Path) -> None:
    for name, sha in ((first_folder, first), (folder, commit)):
        if _git(root, "cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
            raise ReportError(
                f"{name} ran on {sha[:12]}, which this repository does not have: fetch it, or report from a "
                "clone that has it"
            )
    done = _git(root, "diff", "--quiet", first, commit, "--", ".", f":(exclude){OUTPUT_FOLDER}")
    if done.returncode == 1:
        raise ReportError(
            f"{folder} ran on {commit[:12]}, whose code differs from {first[:12]} outside {OUTPUT_FOLDER}/: "
            "runs compared must come from the same code"
        )
    if done.returncode != 0:
        said = " ".join(done.stderr.split())[:200]
        raise ReportError(f"git cannot compare {first[:12]} and {commit[:12]} in {root.name}: {said}")


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
    strerror = getattr(cause, "strerror", None)  # read_run_record wraps an OSError otherwise
    return f"{RUN_FILE} cannot be read" + (f" ({strerror})" if strerror else "")


def _without_folder(message: str, folder: Path) -> str:
    return message.replace(str(folder), folder.name)


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


class ModelSummary(BaseModel):
    """Everything the report says about one model, over its runs that count."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    digest: str | None
    server_versions: list[str]
    runs: list[str]
    run_counts: list[TaskCounts]
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
    precision: Mean | None
    recall: Mean | None
    grounding_failures: int
    model_failures: dict[str, int]
    errored: list[str]  # "task (run n)": a bug of ours, counted as failed
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


class Report(BaseModel):
    """The report on one split: every run listed, and the numbers of the runs that count."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    split: Split
    listed: list[ListedRun]
    models: list[ModelSummary]
    comparisons: list[Comparison]
    scratch: bool  # runs from a checkout with changes are in the numbers


def build_report(
    split: Split, listed: Sequence[ListedRun], tasks: Sequence[Task], *, root: Path = ROOT
) -> Report:
    """The numbers of the runs that count. ReportError when the runs compared did not run the same code, or a
    model ran as more than one build."""
    # A counted run always has its record (ListedRun checks it); the filter only tells the type checker.
    counted = [(run, run.record) for run in listed if run.state == "reported" and run.record is not None]
    check_same_code([run for run, _ in counted], root)
    by_model: dict[str, list[tuple[ListedRun, RunRecord]]] = {}
    for run, record in counted:
        by_model.setdefault(record.model.model, []).append((run, record))
    models = [_model_summary(name, runs, tasks) for name, runs in by_model.items()]
    comparisons = [
        _compare(first, second, [task.id for task in tasks])
        for index, first in enumerate(models)
        for second in models[index + 1 :]
    ]
    scratch = any(record.code.dirty for _, record in counted)
    return Report(split=split, listed=list(listed), models=models, comparisons=comparisons, scratch=scratch)


def _model_summary(
    name: str, counted: Sequence[tuple[ListedRun, RunRecord]], tasks: Sequence[Task]
) -> ModelSummary:
    runs = [run for run, _ in counted]
    records = [record for _, record in counted]
    builds: dict[str | None, str] = {}
    for run, record in zip(runs, records, strict=True):
        builds.setdefault(record.model.digest, run.folder)
    if len(builds) > 1:
        named = ", ".join(f"{digest} in {folder}" for digest, folder in builds.items())
        raise ReportError(
            f"{name} ran as more than one build: {named}. Report the runs of one build together"
        )
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
        grounding_failures=sum(1 for turn in finals if turn.score is not None and not turn.score.grounded),
        model_failures=dict(
            sorted(
                Counter(turn.output.failure for turn in finals if turn.output and turn.output.failure).items()
            )
        ),
        errored=[
            f"{task.id} (run {record.run_index})"
            for run, record in zip(runs, records, strict=True)
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


def _cost(attempts: Sequence[AttemptRecord]) -> Cost | None:
    if not attempts:
        return None
    columns: dict[str, list[float]] = {"seconds": [], "model": [], "load": [], "input": [], "output": []}
    for attempt in attempts:
        usages = [
            usage
            for turn in attempt.turns
            if turn.output is not None
            for usage in (
                turn.output.decision_usage,
                *([turn.output.retrieval.usage] if turn.output.retrieval else []),
            )
        ]
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
    "inconsistent": "inconsistent",
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
            f"labelling rules {first.hashes.evaluation_rules[:12]}.",
            "",
        ]
    else:
        lines += ["No run counts yet.", ""]
    lines += _runs_table(report.listed)
    if report.models:
        lines += _headline_table(report.models)
        if report.comparisons:
            lines += _comparison_table(report.comparisons)
        for summary in report.models:
            lines += _model_section(summary)
        lines += _notes(report)
    return "\n".join(lines).rstrip("\n") + "\n"


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
            run.folder,
            STATE_WORDS[run.state],
            _cell(run.reason or ""),
            record.model.model if record is not None else "",
            *([str(getattr(counts, what)) for what in COUNTED] if counts is not None else [""] * 4),
            str(record.unattributed_calls) if record is not None else "",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return [*lines, ""]


def _headline_table(models: Sequence[ModelSummary]) -> list[str]:
    lines = [
        "## Passed tasks",
        "",
        "The mean over a model's runs, with a 95% bootstrap interval that resamples whole tasks. A lost task "
        "fell to the infrastructure in every attempt: it is shown both ways.",
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
            f"| {summary.model} | {left_out} | {_interval(summary.lost_failed, summary.tasks)} | "
            f"{_stability(summary)} | {len(summary.runs)} |"
        )
    return [*lines, ""]


def _comparison_table(comparisons: Sequence[Comparison]) -> list[str]:
    lines = [
        "## Between models",
        "",
        "A task's result for a model is the majority of its runs. The exact sign test uses the tasks only "
        f"one of the two passed. {_count(len(comparisons), 'comparison')}; differences of fewer than about "
        "5 tasks can be noise.",
        "",
        "| First | Second | Only the first passed | Only the second passed | Left out (no majority) | p | "
        "Both passed | Both failed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for pair in comparisons:
        lines.append(
            f"| {pair.first} | {pair.second} | {pair.first_only} | {pair.second_only} | {pair.left_out} | "
            f"{pair.p_value:.2f} | {pair.both_passed} | {pair.both_failed} |"
        )
    return [*lines, ""]


def _model_section(summary: ModelSummary) -> list[str]:
    lines = [
        f"## {summary.model}",
        "",
        f"Build {summary.digest or 'unknown'}, Ollama {', '.join(summary.server_versions)}; runs "
        f"{', '.join(summary.runs)}.",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Last search call with every argument right | {summary.last_call} |",
        f"| First search call with every argument right | {summary.first_call} |",
        f"| Asked instead of searching | {summary.over_clarification} |",
        f"| Declined instead of searching | {summary.over_refusal} |",
        f"| Scene precision | {_mean_text(summary.precision)} |",
        f"| Recall | {_mean_text(summary.recall)} |",
        f"| Answers citing a scene no tool returned | {summary.grounding_failures} |",
        "| Model failures | "
        + (", ".join(f"{kind} {count}" for kind, count in summary.model_failures.items()) or "none")
        + " |",
        f"| Errored tasks (a bug of ours, counted as failed) | {', '.join(summary.errored) or 'none'} |",
        f"| Attempts lost to the infrastructure (retried) | {summary.infrastructure_attempts} |",
        f"| Tasks lost in every attempt | {summary.lost} |",
        f"| Tool calls under no trace id of the run | {summary.unattributed_calls} |",
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
            f"| {category} | {share.passed:.1f} of {share.with_result} (of {_count(share.tasks, 'task')}) |"
        )
    runs_header = " | ".join(f"Run {index}" for index in range(1, len(summary.runs) + 1))
    lines += ["", f"| Task | {runs_header} |", "|---|" + "---|" * len(summary.runs)]
    for task_id, results in summary.per_task.items():
        lines.append(f"| {task_id} | " + " | ".join(RESULT_WORDS[result] for result in results) + " |")
    return [*lines, ""]


def _notes(report: Report) -> list[str]:
    tasks = report.models[0].tasks
    return [
        "## Reading these numbers",
        "",
        f"- With {tasks} tasks, one task is {100 / tasks:.1f} points. Per-category numbers are descriptive "
        "only.",
        "- A model's runs at temperature 0 are nearly identical: they show stability, not more evidence.",
        "- Infrastructure failures are retried and excluded; errored tasks are a bug of ours and count as "
        "failed.",
        "- Answers are built from templates and cite only scene ids the tools returned, so grounding is a "
        "property of the design.",
        "- Rules and limitations: `docs/evaluation.md` and `docs/decisions.md`.",
        "",
    ]


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
