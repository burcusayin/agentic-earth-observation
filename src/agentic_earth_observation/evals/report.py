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
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from agentic_earth_observation.config import ROOT
from agentic_earth_observation.evals.records import (
    ATTEMPTS_FILE,
    MAX_ATTEMPTS,
    OUTPUT_FOLDER,
    RUN_FILE,
    AttemptRecord,
    RecordError,
    RunRecord,
    TaskCounts,
    read_attempts,
    read_run_record,
)
from agentic_earth_observation.evals.run_eval import (
    FROZEN_EVALUATION_CODE,
    FROZEN_EVALUATION_RULES,
    FROZEN_GAZETTEER,
    Split,
)
from agentic_earth_observation.evals.stats import Result

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
        results = task_results(record, read_attempts(folder))
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
    return ListedRun(folder=folder.name, state="reported", record=record, results=results)


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
            f"{RUN_FILE} is not a run record this version reads: {cause.error_count()} problem(s), the first "
            f"at {where}: {first['msg']}"
        )
    strerror = getattr(cause, "strerror", None)  # read_run_record wraps an OSError otherwise
    return f"{RUN_FILE} cannot be read" + (f" ({strerror})" if strerror else "")


def _without_folder(message: str, folder: Path) -> str:
    return message.replace(str(folder), folder.name)


def _counts(counts: TaskCounts) -> str:
    return ", ".join(f"{getattr(counts, what)} {what}" for what in COUNTED)
