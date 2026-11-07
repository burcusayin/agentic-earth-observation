"""What an evaluation run writes down, and how it is read back.

A run is one model's pass over one split of the golden set. Its folder holds:
- run.json: one RunRecord, which says exactly what ran: the commit, the model build, the hashes of the
  prompts, the tool descriptions, the golden set and the evaluation code, the settings and both seeds;
- attempts.jsonl: one AttemptRecord per attempt of a task, written as each attempt ends, so that a run cut
  short keeps what it finished;
- tool_calls.jsonl: the MCP server's own record of every tool call of the run.

docs/evaluation.md ("Runs and statistics") lists what every run record stores. How an attempt is counted is in
docs/decisions.md (2026-11-02 and 2026-11-05).
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit, urlunsplit

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, model_validator

from agentic_earth_observation.catalog.gazetteer import DEFAULT_PATH as GAZETTEER_PATH
from agentic_earth_observation.catalog_mcp.tool_log import TRACE_ID, ToolCallRecord
from agentic_earth_observation.config import ROOT, Settings
from agentic_earth_observation.evals.golden import evaluation_code_hash, golden_hash
from agentic_earth_observation.evals.scoring import TurnScore
from agentic_earth_observation.orchestrator import prompts as decide_prompts
from agentic_earth_observation.orchestrator.models import TurnOutput
from agentic_earth_observation.retrieval_agent import prompts as retrieval_prompts

RUN_FILE = "run.json"
ATTEMPTS_FILE = "attempts.jsonl"
TOOL_LOG_FILE = "tool_calls.jsonl"
MAX_ATTEMPTS = 3  # the first try, and two more after infrastructure failures (decision of 2026-11-02)
OUTPUT_FOLDER = "results"  # what runs write; it does not make the checkout "dirty"
RAW_RUNS = "raw"  # results/raw: scratch runs, not tracked
REPORTED_RUNS = "runs"  # results/runs: runs for the report, from a clean checkout only
EVALUATION_RULES = ROOT / "docs" / "evaluation.md"  # the labelling rules and the scoring, frozen

RunStatus = Literal["running", "complete", "aborted"]
# scored: the turns ran and were scored. infrastructure: a failure outside the model's control; the task is
# tried again and the attempt is left out of the scores. errored: a bug in our own system; the task counts as
# failed and is listed apart.
AttemptOutcome = Literal["scored", "infrastructure", "errored"]
Value = str | int | float | bool | None
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
TraceId = Annotated[str, Field(pattern=rf"^{TRACE_ID.pattern}$")]

_RECORD = ConfigDict(frozen=True, extra="forbid")

# The settings a run record keeps, and the ones it leaves out, each with the reason. A test checks that every
# setting is in exactly one of the two, so a new setting cannot be recorded, or dropped, without a decision.
RECORDED_SETTINGS = (
    "ollama_url",
    "stac_api_url",
    "mcp_url",
    "a2a_url",
    "stac_timeout_s",
    "stac_max_retries",
    "stac_deadline_s",
    "model_timeout_s",
    "decision_deadline_s",
    "mcp_timeout_s",
    "turn_deadline_s",
    "a2a_timeout_s",
    "harness_timeout_s",
    "max_model_calls",
    "num_predict",
)
NOT_RECORDED = {
    "mistral_api_key": "a secret",
    "tool_log": "a path on this machine; the run's folder holds the log",
    "reference_date": "the run record has the reference date the run used",
    "llm_provider": "in the run record's model build",
    "llm_model": "in the run record's model build",
}
LIBRARIES = (
    "a2a-sdk",
    "langchain-core",
    "langchain-mcp-adapters",
    "langchain-mistralai",
    "langchain-ollama",
    "langgraph",
    "mcp",
    "ollama",
    "pydantic",
    "pystac-client",
)


class ProvenanceError(Exception):
    """The commit of the checkout could not be read. The message says how to fix it."""


class RecordError(Exception):
    """A run's files are missing or cannot be read. The message names the file and, for a line, its number."""


class CodeVersion(BaseModel):
    """The commit a run started from. `dirty`: files outside results/ had changes or were not tracked."""

    model_config = _RECORD

    commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    dirty: bool


class ModelBuild(BaseModel):
    """The model under test, as exactly as its server names it."""

    model_config = _RECORD

    provider: Literal["ollama", "mistral"]
    model: str = Field(min_length=1)
    digest: str | None = None  # Ollama's digest of the model's files; an API model has none
    server_version: str | None = None  # the Ollama version
    options: dict[str, Value]  # what the chat model was built with: context, temperature, seed, ...


class Hashes(BaseModel):
    """SHA-256 of what a model reads and of what scores it. The record's commit covers the other code."""

    model_config = _RECORD

    retrieval_prompt: Sha256
    decide_prompt: Sha256
    tool_descriptions: Sha256  # the tools as the retrieval agent receives them
    golden: Sha256  # both splits
    evaluation_code: Sha256  # golden.py and scoring.py
    gazetteer: Sha256  # decides every expected box and the oracle's
    evaluation_rules: Sha256  # docs/evaluation.md


class TaskCounts(BaseModel):
    """What the tasks a run finished came to. `lost`: every attempt fell to the infrastructure, so the task is
    excluded from the scores, and counted here."""

    model_config = _RECORD

    passed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    errored: int = Field(default=0, ge=0)
    lost: int = Field(default=0, ge=0)


class RunRecord(BaseModel):
    """One run: what ran, on what, and how it ended."""

    model_config = _RECORD

    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")  # also the name of the run's folder
    status: RunStatus
    aborted_reason: str | None = None
    split: Literal["dev", "test"]
    run_index: int = Field(ge=1)  # which of the model's runs in this evaluation this is
    started_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    code: CodeVersion
    model: ModelBuild
    hashes: Hashes
    reference_date: date  # "today" for the agents
    shuffle_seed: int  # orders the tasks; the model's own seed is in model.options
    task_order: list[str] = Field(min_length=1)  # the task ids, in the order they ran
    # Calls in the run's tool log under none of the run's trace ids: they change no attempt, and the report
    # marks a run that has any (decision of 2026-11-06). A call that a turn past the harness timeout logs
    # after its attempt was read is under a known trace id, so it is in neither the attempt nor this count.
    unattributed_calls: int = Field(default=0, ge=0)
    # The tasks that ended. A task cut short by an abort has its attempts in attempts.jsonl and no count
    # here, so `lost` cannot be read off "the last attempt is infrastructure".
    tasks: TaskCounts = TaskCounts()
    settings: dict[str, Value]
    libraries: dict[str, str]
    platform: str

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if (self.status == "running") != (self.finished_at is None):
            raise ValueError("a running run has no end time, and a run that ended has one")
        if (self.status == "aborted") != (self.aborted_reason is not None):
            raise ValueError("an aborted run says why, and only an aborted run")
        if len(set(self.task_order)) < len(self.task_order):
            raise ValueError("task_order names a task twice")
        return self


class OracleCheck(BaseModel):
    """The direct catalogue query for a turn that expects a search (docs/evaluation.md, "Reported alongside").

    `ids` are the scenes the expected search returns. When the query fails, `failed` says why and the metrics
    are missing; the turn's score does not depend on them.
    """

    model_config = _RECORD

    ids: list[str] = []
    failed: str | None = None
    precision: float | None = Field(default=None, ge=0, le=1)
    # Recall is measured only when the question asks for a number of scenes.
    recall: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.failed is not None and (self.ids or self.precision is not None or self.recall is not None):
            raise ValueError("a failed query has no ids and no metrics")
        return self


class TurnRecord(BaseModel):
    """One turn of an attempt: what the system did, what the MCP server logged, and how it scored."""

    model_config = _RECORD

    trace_id: TraceId
    seconds: float = Field(ge=0)  # the runner's own clock around the turn
    output: TurnOutput | None = None  # None when the turn raised
    error: str | None = None  # the error a turn raised: its type and message
    calls: list[ToolCallRecord] = []  # as the MCP server logged them, in the order they were sent
    score: TurnScore | None = None  # only in a scored attempt
    oracle: OracleCheck | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if (self.output is None) != (self.error is not None):
            raise ValueError("a turn has its output, or the error that stopped it, never both or neither")
        if self.score is not None and self.output is None:
            raise ValueError("a turn without an output cannot be scored")
        return self


class AttemptRecord(BaseModel):
    """One attempt of a task: every turn it ran, and what the attempt counts as."""

    model_config = _RECORD

    run_id: str
    task_id: str
    attempt: int = Field(ge=1, le=MAX_ATTEMPTS)
    outcome: AttemptOutcome
    passed: bool | None  # scored: whether the task passed; errored: False; infrastructure: None
    reason: str | None = None  # why the attempt is not scored
    turns: list[TurnRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if any(turn.output is None for turn in self.turns[:-1]):
            raise ValueError(
                "an attempt stops at the turn that raised: only its last turn may lack an output"
            )
        scores = [turn.score is not None for turn in self.turns]
        if self.outcome == "scored":
            if self.passed is None or self.reason is not None:
                raise ValueError("a scored attempt passed or failed, and gives no reason")
            if not all(scores):
                raise ValueError("a scored attempt has a score for every turn")
            return self
        if not self.reason:
            raise ValueError(f"an attempt counted as {self.outcome} says why")
        if any(scores):
            raise ValueError(f"an attempt counted as {self.outcome} is not scored: its turns carry no score")
        if self.outcome == "infrastructure" and self.passed is not None:
            raise ValueError("an attempt lost to the infrastructure neither passed nor failed")
        if self.outcome == "errored" and self.passed is not False:
            raise ValueError("an errored attempt counts as failed")
        return self


def new_run_id(model: str, run_index: int, started: datetime) -> str:
    """A run's id and folder name, such as 20261105T213045Z-qwen3.5-9b-r1: it sorts by start time."""
    if started.tzinfo is None:
        raise ValueError("the start time needs a time zone")
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", model).strip("-")
    return f"{started.astimezone(UTC):%Y%m%dT%H%M%SZ}-{slug}-r{run_index}"


def code_version(root: Path = ROOT) -> CodeVersion:
    """The commit checked out at `root`, and whether the checkout differs from it.

    Dirty means a tracked file has changes, staged or not, or a file is neither tracked nor ignored. The
    results/ folder does not count: it holds what runs write, not what they run. `root` must be the top of a
    git checkout, so that a folder inside another repository is never described by that repository's commit.
    """
    top = _git(root, "rev-parse", "--show-toplevel").strip()
    if not top or not os.path.samefile(top, root):
        raise ProvenanceError(
            f"{root} is not the top of a git checkout (git says {top or 'nothing'}); "
            "run the evaluation from a clone of the repository"
        )
    commit = _git(root, "rev-parse", "HEAD").strip()
    # --untracked-files=all: a user's git configuration may hide untracked files from `git status`.
    changes = _git(
        root, "status", "--porcelain", "--untracked-files=all", "--", ".", f":(exclude){OUTPUT_FOLDER}"
    )
    return CodeVersion(commit=commit, dirty=bool(changes.strip()))


def uncommitted_runs(root: Path = ROOT) -> bool:
    """Whether results/runs holds anything git does not: a run for the report that is not on record yet."""
    folder = f"{OUTPUT_FOLDER}/{REPORTED_RUNS}"
    return bool(_git(root, "status", "--porcelain", "--untracked-files=all", "--", folder).strip())


def _git(root: Path, *arguments: str) -> str:
    git = shutil.which("git")
    if git is None:
        raise ProvenanceError("git is not installed: an evaluation run records the commit it ran from")
    try:
        done = subprocess.run(  # noqa: S603 - git's full path and fixed arguments, no shell
            [git, "-C", str(root), *arguments], capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as err:
        raise ProvenanceError(f"git {arguments[0]} could not run in {root}: {type(err).__name__}") from err
    if done.returncode != 0:
        said = " ".join(done.stderr.split())[:200]
        raise ProvenanceError(
            f"git {arguments[0]} failed in {root} ({said}); run the evaluation from a clone of the repository"
        )
    return done.stdout


def hashes(tool_descriptions: str) -> Hashes:
    """The hashes a run record stores. `tool_descriptions` is `tool_descriptions_hash` of the tools as the
    retrieval agent receives them from the running MCP server."""
    return Hashes(
        retrieval_prompt=retrieval_prompts.prompt_hash(),
        decide_prompt=decide_prompts.prompt_hash(),
        tool_descriptions=tool_descriptions,
        golden=golden_hash(),
        evaluation_code=evaluation_code_hash(),
        gazetteer=file_hash(GAZETTEER_PATH),
        evaluation_rules=file_hash(EVALUATION_RULES),
    )


def file_hash(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded_settings(settings: Settings) -> dict[str, Value]:
    """The settings a run record keeps: the named ones only, so that no secret or local path gets in."""
    values: dict[str, Value] = settings.model_dump(mode="json", include=set(RECORDED_SETTINGS))
    return {name: _without_credentials(values[name]) for name in RECORDED_SETTINGS}


def _without_credentials(value: Value) -> Value:
    """A URL as it may be published: a user name or password in it is left out."""
    if not isinstance(value, str) or "@" not in value:
        return value
    parts = urlsplit(value)
    return urlunsplit(parts._replace(netloc=parts.netloc.rsplit("@", 1)[1]))


def _for_the_record(text: str) -> str:
    """A record as it may be published: error messages from anywhere in the system can name a path of this
    machine, and a record never does."""
    for path, name in ((ROOT, "<repository>"), (Path.home(), "~")):
        if len(path.parts) > 1:  # a home folder of `/` would otherwise rewrite every path and URL
            text = text.replace(str(path), name)
    return text


def library_versions() -> dict[str, str]:
    """The installed version of each library the agents are built on, and of Python."""
    return {name: version(name) for name in LIBRARIES} | {"python": platform.python_version()}


def write_run_record(folder: Path, record: RunRecord) -> None:
    """Write run.json. An earlier version is replaced in one step, so a reader never sees half a file."""
    folder.mkdir(parents=True, exist_ok=True)
    draft = folder / f"{RUN_FILE}.tmp"
    with draft.open("w", encoding="utf-8") as out:
        out.write(_for_the_record(record.model_dump_json(indent=2)) + "\n")
        out.flush()
        os.fsync(out.fileno())
    draft.replace(folder / RUN_FILE)


def append_attempt(folder: Path, record: AttemptRecord) -> None:
    """Add one attempt to attempts.jsonl, and make sure it is on disk before the run goes on."""
    with (folder / ATTEMPTS_FILE).open("a", encoding="utf-8") as out:
        out.write(_for_the_record(record.model_dump_json()) + "\n")
        out.flush()
        os.fsync(out.fileno())


def read_run_record(folder: Path) -> RunRecord:
    """The run record of a run's folder."""
    path = folder / RUN_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as err:
        raise RecordError(f"cannot read {path} ({err.strerror}): is {folder} the folder of a run?") from err
    try:
        return RunRecord.model_validate_json(text)
    except ValidationError as err:
        raise RecordError(f"{path}: not a run record: {err}") from err


def read_attempts(folder: Path) -> list[AttemptRecord]:
    """Every attempt of a run, in the order they ended. A run that finished no attempt has no file: none.

    A last line without its newline is an attempt whose writing was cut short, by a crash: it did not finish.
    """
    path = folder / ATTEMPTS_FILE
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as err:
        raise RecordError(f"cannot read {path} ({err.strerror})") from err
    attempts = []
    # Split on "\n" only: str.splitlines() also splits on U+2028 and the like, which JSON leaves unescaped.
    for number, line in enumerate(text.split("\n")[:-1], start=1):
        if not line.strip():
            continue
        try:
            attempts.append(AttemptRecord.model_validate_json(line))
        except ValidationError as err:
            raise RecordError(f"{path}:{number}: not an attempt record: {err}") from err
    return attempts
