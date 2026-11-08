"""Tests for what an evaluation run writes down: the run record, the attempt records, and where the commit,
the hashes and the settings in them come from. No network and no model: git runs on folders made for the test.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from agentic_earth_observation.catalog_mcp.tool_log import ToolCallRecord
from agentic_earth_observation.config import ROOT, Settings
from agentic_earth_observation.domain import Product
from agentic_earth_observation.evals.golden import Route, evaluation_code_hash, golden_hash
from agentic_earth_observation.evals.records import (
    ATTEMPTS_FILE,
    LIBRARIES,
    MAX_ATTEMPTS,
    NOT_RECORDED,
    RECORDED_SETTINGS,
    RUN_FILE,
    AttemptRecord,
    CodeVersion,
    Hashes,
    ModelBuild,
    OracleCheck,
    ProvenanceError,
    RecordError,
    RunRecord,
    TurnRecord,
    append_attempt,
    code_version,
    hashes,
    library_versions,
    new_run_id,
    read_attempts,
    read_run_record,
    recorded_settings,
    uncommitted_runs,
    write_run_record,
)
from agentic_earth_observation.evals.scoring import ArgumentScore, TurnScore
from agentic_earth_observation.orchestrator import prompts as decide_prompts
from agentic_earth_observation.orchestrator.models import TurnOutput
from agentic_earth_observation.retrieval_agent import prompts as retrieval_prompts
from agentic_earth_observation.retrieval_agent.models import PreviousSearch, RetrievalResult, Usage
from tests.fakes import SCENE

RUN_ID = "20261105T213045Z-qwen3.5-9b-r1"
STARTED = datetime(2026, 11, 5, 21, 30, 45, tzinfo=UTC)
A_HASH = "ab" * 32
MILAN_MARCH = PreviousSearch(
    place="Milan",
    product=Product.SENTINEL_2_L2A,
    bbox=[9.0409, 45.3867, 9.2781, 45.5358],
    start_date=date(2025, 3, 1),
    end_date=date(2025, 3, 31),
    max_cloud_cover=20.0,
)
ALL_RIGHT = ArgumentScore(product=True, bbox=True, dates=True, cloud=True, limit=True, no_extra=True)
PASSED = TurnScore(
    route=True, tools=True, arguments=ALL_RIGHT, first_arguments=ALL_RIGHT, grounded=True, outcome=True
)


def run_record(**changes: Any) -> RunRecord:  # noqa: ANN401 - any field of the record
    fields: dict[str, Any] = {
        "run_id": RUN_ID,
        "status": "running",
        "split": "dev",
        "run_index": 1,
        "started_at": STARTED,
        "code": CodeVersion(commit="0123456789abcdef0123456789abcdef01234567", dirty=False),
        "model": ModelBuild(
            provider="ollama",
            model="qwen3.5:9b",
            digest="sha256-of-the-model",
            server_version="0.40.1",
            options={"num_ctx": 16384, "temperature": 0, "seed": 42, "reasoning": False, "keep_alive": "30m"},
        ),
        "hashes": Hashes(
            retrieval_prompt=A_HASH,
            decide_prompt=A_HASH,
            tool_descriptions=A_HASH,
            golden=A_HASH,
            evaluation_code=A_HASH,
            gazetteer=A_HASH,
            evaluation_rules=A_HASH,
        ),
        "reference_date": date(2025, 12, 15),
        "shuffle_seed": 42,
        "task_order": ["dev-b", "dev-a"],
        "settings": {"turn_deadline_s": 240.0, "max_model_calls": 6, "mcp_url": "http://127.0.0.1:8001/mcp"},
        "libraries": {"langgraph": "1.2.14"},
        "platform": "macOS-26-arm64",
    }
    return RunRecord.model_validate(fields | changes)


def output(trace_id: str = "0a1b2c3d-t01-a1-u1", answer: str = "I found 1 scene.") -> TurnOutput:
    """A delegated turn that found one scene."""
    usage = Usage(model="qwen3.5:9b", model_calls=3, input_tokens=300, output_tokens=30, seconds=12.5)
    return TurnOutput(
        trace_id=trace_id,
        route=Route.DELEGATE,
        status="ok",
        answer=answer,
        cited_ids=[SCENE.id],
        decision_usage=Usage(model="qwen3.5:9b", model_calls=1, input_tokens=100, output_tokens=10),
        retrieval=RetrievalResult(status="ok", scenes=[SCENE], search=MILAN_MARCH, usage=usage),
    )


def turn(**changes: Any) -> TurnRecord:  # noqa: ANN401 - any field of the record
    """A scored turn with its two logged calls and its oracle check."""
    trace_id = "0a1b2c3d-t01-a1-u1"
    fields: dict[str, Any] = {
        "trace_id": trace_id,
        "seconds": 14.2,
        "output": output(trace_id),
        "calls": [
            ToolCallRecord(
                trace_id=trace_id, request_id=1, tool="resolve_place", arguments={"name": "Milan"}, ok=True
            ),
            ToolCallRecord(
                trace_id=trace_id,
                request_id=2,
                tool="search_scenes",
                arguments={"product": "sentinel-2-l2a", "bbox": MILAN_MARCH.bbox, "max_cloud_cover": 20},
                ok=True,
                returned_ids=[SCENE.id],
            ),
        ],
        "score": PASSED,
        "oracle": OracleCheck(ids=[SCENE.id], precision=1.0),
    }
    return TurnRecord.model_validate(fields | changes)


def attempt(**changes: Any) -> AttemptRecord:  # noqa: ANN401 - any field of the record
    fields: dict[str, Any] = {
        "run_id": RUN_ID,
        "task_id": "dev-a",
        "attempt": 1,
        "outcome": "scored",
        "passed": True,
        "turns": [turn()],
    }
    return AttemptRecord.model_validate(fields | changes)


UNSCORED = turn(score=None)
RAISED = turn(
    output=None, error="RetrievalAgentError: the retrieval agent ended the turn as FAILED", score=None
)


# --- the run record


def test_a_run_record_comes_back_from_its_file_as_it_was_written(tmp_path: Path) -> None:
    folder = tmp_path / "runs" / RUN_ID  # made by the write
    record = run_record()

    write_run_record(folder, record)

    assert read_run_record(folder) == record
    assert sorted(p.name for p in folder.iterdir()) == [RUN_FILE]  # no draft is left behind
    assert (folder / RUN_FILE).read_text(encoding="utf-8").endswith("}\n")


def test_writing_the_run_record_again_replaces_it(tmp_path: Path) -> None:
    write_run_record(tmp_path, run_record())
    ended = run_record(status="complete", finished_at=STARTED + timedelta(minutes=30))

    write_run_record(tmp_path, ended)

    assert read_run_record(tmp_path) == ended


def test_a_write_that_fails_part_way_leaves_the_earlier_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = run_record()
    write_run_record(tmp_path, first)

    def disk_full(descriptor: int) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "fsync", disk_full)
    with pytest.raises(OSError, match="No space left"):
        write_run_record(tmp_path, run_record(status="complete", finished_at=STARTED))

    assert read_run_record(tmp_path) == first


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"finished_at": STARTED}, "a running run has no end time"),
        ({"status": "complete"}, "a run that ended has one"),
        ({"status": "aborted", "finished_at": STARTED}, "an aborted run says why"),
        (
            {"status": "complete", "finished_at": STARTED, "aborted_reason": "3 tasks errored"},
            "only an aborted",
        ),
        ({"aborted_reason": "3 tasks errored"}, "only an aborted"),
        ({"task_order": ["dev-a", "dev-b", "dev-a"]}, "names a task twice"),
        ({"task_order": []}, "at least 1 item"),
        ({"run_index": 0}, "greater than or equal to 1"),
        ({"started_at": datetime(2026, 11, 5, 21, 30)}, "timezone"),
        ({"split": "train"}, "'dev' or 'test'"),
        ({"shuffle_seed": None}, "valid integer"),
        ({"unknown_field": 1}, "Extra inputs are not permitted"),
    ],
)
def test_a_run_record_that_contradicts_itself_is_refused(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        run_record(**changes)


def test_a_run_counts_the_calls_it_cannot_tie_to_a_task() -> None:
    assert run_record().unattributed_calls == 0
    assert run_record(unattributed_calls=2).unattributed_calls == 2
    with pytest.raises(ValidationError, match="unattributed_calls"):
        run_record(unattributed_calls=-1)


def test_a_run_ends_complete_or_aborted_with_its_reason() -> None:
    ended = STARTED + timedelta(minutes=30)
    assert run_record(status="complete", finished_at=ended).finished_at == ended
    aborted = run_record(status="aborted", finished_at=ended, aborted_reason="3 tasks errored")
    assert aborted.aborted_reason == "3 tasks errored"


@pytest.mark.parametrize("run_id", ["", "../outside", "a/b", ".hidden", "has space", "x" * 101])
def test_a_run_id_is_a_safe_folder_name(run_id: str) -> None:
    with pytest.raises(ValidationError, match="run_id"):
        run_record(run_id=run_id)


@pytest.mark.parametrize("value", ["abc", "AB" * 32, "ab" * 33, ""])
def test_a_hash_is_64_lowercase_hex_characters(value: str) -> None:
    with pytest.raises(ValidationError, match="golden"):
        Hashes(
            retrieval_prompt=A_HASH,
            decide_prompt=A_HASH,
            tool_descriptions=A_HASH,
            golden=value,
            evaluation_code=A_HASH,
            gazetteer=A_HASH,
            evaluation_rules=A_HASH,
        )


@pytest.mark.parametrize("commit", ["470544e", "G" * 40, "ab" * 33, ""])
def test_a_commit_is_a_full_hash(commit: str) -> None:
    with pytest.raises(ValidationError, match="commit"):
        CodeVersion(commit=commit, dirty=False)


# --- turn and attempt records


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"output": None, "score": None}, "never both or neither"),
        ({"error": "ValueError: boom"}, "never both or neither"),
        ({"output": None, "error": "ValueError: boom"}, "cannot be scored"),
        ({"trace_id": "has space"}, "trace_id"),
        ({"trace_id": "x" * 65}, "trace_id"),
        ({"trace_id": ""}, "trace_id"),
        ({"seconds": -1.0}, "greater than or equal to 0"),
    ],
)
def test_a_turn_record_that_contradicts_itself_is_refused(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        turn(**changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"ids": [SCENE.id], "failed": "the catalogue did not answer"}, "no ids and no metrics"),
        ({"precision": 1.0, "failed": "the catalogue did not answer"}, "no ids and no metrics"),
        ({"recall": 0.5, "failed": "the catalogue did not answer"}, "no ids and no metrics"),
        ({"precision": 1.5}, "less than or equal to 1"),
        ({"recall": -0.1}, "greater than or equal to 0"),
    ],
)
def test_an_oracle_check_that_contradicts_itself_is_refused(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        OracleCheck(**changes)


def test_a_failed_oracle_query_says_why_and_has_no_metrics() -> None:
    check = OracleCheck(failed="the catalogue did not answer")
    assert (check.ids, check.precision, check.recall) == ([], None, None)


@pytest.mark.parametrize(
    "fields",
    [
        {"outcome": "scored", "passed": True},
        {"outcome": "scored", "passed": False},
        {"outcome": "scored", "passed": True, "turns": [turn(), turn(trace_id="0a1b2c3d-t01-a1-u2")]},
        {"outcome": "infrastructure", "passed": None, "reason": "A2A ReadTimeout", "turns": [UNSCORED]},
        {"outcome": "errored", "passed": False, "reason": "RetrievalAgentError", "turns": [RAISED]},
        # a turn that ran, in an attempt that is not scored: the agent and the log disagree
        {"outcome": "errored", "passed": False, "reason": "the log has no search", "turns": [UNSCORED]},
        {"outcome": "errored", "passed": False, "reason": "ValueError", "turns": [UNSCORED, RAISED]},
        {"outcome": "scored", "passed": True, "attempt": MAX_ATTEMPTS},
    ],
)
def test_an_attempt_is_scored_or_lost_to_the_infrastructure_or_errored(fields: dict[str, Any]) -> None:
    assert attempt(**fields).outcome == fields["outcome"]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"passed": None}, "a scored attempt passed or failed"),
        ({"reason": "why not"}, "gives no reason"),
        ({"turns": [turn(), UNSCORED]}, "a score for every turn"),
        ({"turns": [RAISED]}, "a score for every turn"),
        ({"outcome": "infrastructure", "passed": None, "turns": [UNSCORED]}, "infrastructure says why"),
        ({"outcome": "infrastructure", "passed": None, "reason": "", "turns": [UNSCORED]}, "says why"),
        (
            {"outcome": "infrastructure", "passed": False, "reason": "A2A ReadTimeout", "turns": [UNSCORED]},
            "neither passed nor failed",
        ),
        (
            {"outcome": "infrastructure", "passed": None, "reason": "A2A ReadTimeout", "turns": [turn()]},
            "its turns carry no score",
        ),
        ({"outcome": "errored", "passed": False, "turns": [RAISED]}, "errored says why"),
        (
            {"outcome": "errored", "passed": None, "reason": "ValueError", "turns": [RAISED]},
            "counts as failed",
        ),
        (
            {"outcome": "errored", "passed": True, "reason": "ValueError", "turns": [RAISED]},
            "counts as failed",
        ),
        (
            {"outcome": "errored", "passed": False, "reason": "ValueError", "turns": [turn()]},
            "carry no score",
        ),
        (
            {"outcome": "errored", "passed": False, "reason": "ValueError", "turns": [RAISED, UNSCORED]},
            "only its last turn may lack an output",
        ),
        ({"attempt": 0}, "greater than or equal to 1"),
        ({"attempt": MAX_ATTEMPTS + 1}, f"less than or equal to {MAX_ATTEMPTS}"),
        ({"turns": []}, "at least 1 item"),
        ({"outcome": "skipped"}, "'scored', 'infrastructure' or 'errored'"),
    ],
)
def test_an_attempt_record_that_contradicts_itself_is_refused(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        attempt(**changes)


def test_three_attempts_are_the_first_try_and_two_retries() -> None:
    assert MAX_ATTEMPTS == 3  # docs/decisions.md, 2026-11-02: retried at most twice


def test_attempts_come_back_from_their_file_in_the_order_they_were_added(tmp_path: Path) -> None:
    lost = attempt(outcome="infrastructure", passed=None, reason="A2A ReadTimeout", turns=[UNSCORED])
    scored = attempt(attempt=2)
    errored = attempt(
        task_id="dev-b", outcome="errored", passed=False, reason="ValueError: boom", turns=[RAISED]
    )

    for record in (lost, scored, errored):
        append_attempt(tmp_path, record)

    assert read_attempts(tmp_path) == [lost, scored, errored]
    assert (tmp_path / ATTEMPTS_FILE).read_text(encoding="utf-8").count("\n") == 3  # one line each


def test_an_attempt_is_on_disk_before_the_run_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[str] = []
    real_fsync = os.fsync

    def fsync(descriptor: int) -> None:
        synced.append((tmp_path / ATTEMPTS_FILE).read_text(encoding="utf-8"))  # what is in the file by then
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fsync)
    record = attempt()

    append_attempt(tmp_path, record)

    assert synced == [record.model_dump_json() + "\n"]


def test_a_line_separator_inside_an_answer_does_not_split_its_record(tmp_path: Path) -> None:
    # JSON leaves U+2028 and U+0085 unescaped, and str.splitlines() would split the record there.
    answer = "first\u2028second\u0085third"
    record = attempt(turns=[turn(output=output(answer=answer))])

    append_attempt(tmp_path, record)
    append_attempt(tmp_path, record)

    assert read_attempts(tmp_path) == [record, record]


def test_a_run_that_finished_no_attempt_has_none(tmp_path: Path) -> None:
    assert read_attempts(tmp_path) == []


def test_blank_lines_between_attempts_are_skipped(tmp_path: Path) -> None:
    line = attempt().model_dump_json()
    (tmp_path / ATTEMPTS_FILE).write_text(f"{line}\n\n   \n{line}\n", encoding="utf-8")

    assert len(read_attempts(tmp_path)) == 2


@pytest.mark.parametrize("bad", ['{"run_id": "half a rec', "{}", "not json"])
def test_a_line_that_is_not_an_attempt_is_reported_with_its_number(tmp_path: Path, bad: str) -> None:
    line = attempt().model_dump_json()
    (tmp_path / ATTEMPTS_FILE).write_text(f"{line}\n{bad}\n", encoding="utf-8")

    with pytest.raises(RecordError, match=rf"{ATTEMPTS_FILE}:2: not an attempt record"):
        read_attempts(tmp_path)


def test_an_attempts_file_that_cannot_be_read_is_reported(tmp_path: Path) -> None:
    (tmp_path / ATTEMPTS_FILE).mkdir()  # a folder where the file should be

    with pytest.raises(RecordError, match=rf"cannot read .*{ATTEMPTS_FILE} \(Is a directory\)"):
        read_attempts(tmp_path)


def test_a_folder_without_a_run_record_is_reported_as_such(tmp_path: Path) -> None:
    with pytest.raises(RecordError, match=rf"cannot read .*{RUN_FILE}.*is .* the folder of a run\?"):
        read_run_record(tmp_path)


@pytest.mark.parametrize("text", ["", "{}", "not json", '{"run_id": "x"}'])
def test_a_file_that_is_not_a_run_record_is_reported(tmp_path: Path, text: str) -> None:
    (tmp_path / RUN_FILE).write_text(text, encoding="utf-8")

    with pytest.raises(RecordError, match=rf"{RUN_FILE}: not a run record"):
        read_run_record(tmp_path)


# --- where the run record's contents come from


def test_every_setting_is_either_recorded_or_left_out_for_a_stated_reason() -> None:
    assert set(RECORDED_SETTINGS) | set(NOT_RECORDED) == set(Settings.model_fields)
    assert not set(RECORDED_SETTINGS) & set(NOT_RECORDED)
    assert len(set(RECORDED_SETTINGS)) == len(RECORDED_SETTINGS)
    assert all(NOT_RECORDED.values())  # each with its reason


def test_the_recorded_settings_are_the_named_ones_with_their_values() -> None:
    settings = Settings(_env_file=None, turn_deadline_s=200, max_model_calls=5)  # type: ignore[call-arg]  # a pydantic-settings init option

    recorded = recorded_settings(settings)

    assert list(recorded) == list(RECORDED_SETTINGS)
    assert recorded["turn_deadline_s"] == 200.0
    assert recorded["max_model_calls"] == 5
    assert recorded["mcp_url"] == "http://127.0.0.1:8001/mcp"
    assert all(type(value) in (str, int, float) for value in recorded.values())


def test_no_secret_and_no_local_path_gets_into_a_run_record(tmp_path: Path) -> None:
    settings = Settings(  # type: ignore[call-arg]  # _env_file is a pydantic-settings init option
        _env_file=None,
        llm_provider="mistral",
        mistral_api_key=SecretStr("example-key-value"),
        tool_log=tmp_path / "somewhere" / "calls.jsonl",
    )

    written = run_record(settings=recorded_settings(settings)).model_dump_json()

    assert "example-key-value" not in written
    assert "**" not in written  # not even a masked secret
    assert str(tmp_path) not in written


def test_the_hashes_are_those_of_the_prompts_and_of_every_frozen_file() -> None:
    found = hashes(A_HASH)

    assert found == Hashes(
        retrieval_prompt=retrieval_prompts.prompt_hash(),
        decide_prompt=decide_prompts.prompt_hash(),
        tool_descriptions=A_HASH,
        golden=golden_hash(),
        evaluation_code=evaluation_code_hash(),
        gazetteer=hashlib.sha256((ROOT / "data" / "gazetteer.json").read_bytes()).hexdigest(),
        evaluation_rules=hashlib.sha256((ROOT / "docs" / "evaluation.md").read_bytes()).hexdigest(),
    )
    assert found.retrieval_prompt != found.decide_prompt
    assert len({found.golden, found.evaluation_code, found.gazetteer, found.evaluation_rules}) == 4


def test_the_library_versions_name_every_library_and_python() -> None:
    versions = library_versions()

    assert set(versions) == {*LIBRARIES, "python"}
    assert versions["python"].startswith("3.13")
    assert all(text and text[0].isdigit() for text in versions.values())


def test_every_recorded_library_is_a_dependency_of_the_project() -> None:
    declared = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert [name for name in LIBRARIES if f'"{name}' not in declared] == []


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("qwen3.5:9b", "20261105T213045Z-qwen3.5-9b-r1"),
        ("ministral-3:8b", "20261105T213045Z-ministral-3-8b-r1"),
        ("org/some model:latest", "20261105T213045Z-org-some-model-latest-r1"),
    ],
)
def test_a_run_id_names_the_start_time_the_model_and_the_run(model: str, expected: str) -> None:
    run_id = new_run_id(model, 1, STARTED)

    assert run_id == expected
    assert run_record(run_id=run_id).run_id == run_id  # accepted as a run id


def test_a_run_id_gives_the_start_time_in_utc_and_the_run_index() -> None:
    rome = timezone(timedelta(hours=1))
    assert new_run_id("m", 3, datetime(2026, 11, 5, 22, 30, 45, tzinfo=rome)) == "20261105T213045Z-m-r3"


def test_a_start_time_without_a_time_zone_is_refused() -> None:
    with pytest.raises(ValueError, match="needs a time zone"):
        new_run_id("m", 1, datetime(2026, 11, 5, 21, 30))


# --- the commit and the dirty flag


def git(repo: Path, *arguments: str) -> str:
    """Run git in `repo`, with a test identity and without the developer's own git configuration."""
    program = shutil.which("git")
    assert program is not None, "these tests need git"
    identity = ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false"]
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
    done = subprocess.run(  # noqa: S603 - git's full path and the test's own arguments
        [program, "-C", str(repo), *identity, *arguments], check=True, capture_output=True, text=True, env=env
    )
    return done.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A git checkout with one commit: a tracked file, and an ignore rule for *.log."""
    folder = tmp_path / "checkout"
    folder.mkdir()
    git(folder, "init", "--quiet")
    (folder / "tracked.txt").write_text("as committed\n", encoding="utf-8")
    (folder / ".gitignore").write_text("*.log\n", encoding="utf-8")
    git(folder, "add", ".")
    git(folder, "commit", "--quiet", "-m", "first")
    return folder


def write(path: Path, text: str = "new\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_a_clean_checkout_gives_its_commit_and_is_not_dirty(repo: Path) -> None:
    version = code_version(repo)

    assert version.commit == git(repo, "rev-parse", "HEAD").strip()
    assert len(version.commit) == 40
    assert version.dirty is False


def test_a_changed_tracked_file_makes_the_checkout_dirty(repo: Path) -> None:
    write(repo / "tracked.txt", "changed\n")
    assert code_version(repo).dirty is True


def test_a_staged_change_makes_the_checkout_dirty(repo: Path) -> None:
    write(repo / "tracked.txt", "changed\n")
    git(repo, "add", "tracked.txt")
    assert code_version(repo).dirty is True


def test_a_deleted_tracked_file_makes_the_checkout_dirty(repo: Path) -> None:
    (repo / "tracked.txt").unlink()
    assert code_version(repo).dirty is True


@pytest.mark.parametrize("name", ["untracked.py", "src/deep/untracked.py", "results.txt", "src/results/x.py"])
def test_an_untracked_file_makes_the_checkout_dirty(repo: Path, name: str) -> None:
    write(repo / name)
    assert code_version(repo).dirty is True


def test_untracked_files_count_even_when_git_is_set_to_hide_them(repo: Path) -> None:
    git(repo, "config", "status.showUntrackedFiles", "no")
    write(repo / "untracked.py")
    assert code_version(repo).dirty is True


def test_an_ignored_file_does_not_make_the_checkout_dirty(repo: Path) -> None:
    write(repo / "server.log")
    assert code_version(repo).dirty is False


def test_what_runs_write_under_results_does_not_make_the_checkout_dirty(repo: Path) -> None:
    write(repo / "results" / "runs" / RUN_ID / RUN_FILE)
    assert code_version(repo).dirty is False


def test_a_run_for_the_report_that_is_not_committed_is_noticed_until_it_is(repo: Path) -> None:
    assert uncommitted_runs(repo) is False
    write(repo / "results" / "runs" / RUN_ID / RUN_FILE)
    assert uncommitted_runs(repo) is True
    git(repo, "add", ".")
    assert uncommitted_runs(repo) is True  # staged is not on record yet
    git(repo, "commit", "--quiet", "-m", "a run")
    assert uncommitted_runs(repo) is False


def test_scratch_runs_and_server_logs_are_not_runs_waiting_to_be_committed(repo: Path) -> None:
    write(repo / "results" / "raw" / RUN_ID / RUN_FILE)
    write(repo / "results" / "runs" / RUN_ID / "mcp_server.log")  # ignored, as *.log is in the real checkout
    assert uncommitted_runs(repo) is False


def test_a_changed_result_that_was_committed_does_not_make_the_checkout_dirty(repo: Path) -> None:
    write(repo / "results" / "runs" / "summary.md")
    git(repo, "add", ".")
    git(repo, "commit", "--quiet", "-m", "results")
    write(repo / "results" / "runs" / "summary.md", "changed\n")
    assert code_version(repo).dirty is False


def test_a_folder_inside_a_checkout_is_not_described_by_its_commit(repo: Path) -> None:
    inside = repo / "inside"
    inside.mkdir()
    with pytest.raises(ProvenanceError, match="is not the top of a git checkout"):
        code_version(inside)


def test_a_folder_that_is_no_checkout_is_refused_with_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(ProvenanceError, match="run the evaluation from a clone of the repository"):
        code_version(tmp_path)


def test_a_checkout_without_any_commit_is_refused(tmp_path: Path) -> None:
    git(tmp_path, "init", "--quiet")
    with pytest.raises(ProvenanceError, match="git rev-parse failed"):
        code_version(tmp_path)


def test_a_machine_without_git_is_told_so(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)  # the module records.py looks git up with
    with pytest.raises(ProvenanceError, match="git is not installed"):
        code_version(repo)


def test_a_git_that_does_not_answer_is_reported(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def hangs(command: list[str], **options: object) -> None:
        raise subprocess.TimeoutExpired(command, 30)

    monkeypatch.setattr(subprocess, "run", hangs)
    with pytest.raises(ProvenanceError, match=r"git rev-parse could not run in .*: TimeoutExpired"):
        code_version(repo)


def test_the_commit_of_this_repository_is_the_one_checked_out() -> None:
    assert code_version().commit == git(ROOT, "rev-parse", "HEAD").strip()


def test_a_written_run_record_is_plain_json_with_the_documented_fields(tmp_path: Path) -> None:
    # docs/evaluation.md, "Every run record stores": the commit and the dirty flag, the model id, the hashes
    # of the prompts, the tool descriptions, the golden set and the evaluation code, and the seed.
    write_run_record(tmp_path, run_record())

    stored = json.loads((tmp_path / RUN_FILE).read_text(encoding="utf-8"))

    assert stored["code"] == {"commit": "0123456789abcdef0123456789abcdef01234567", "dirty": False}
    assert stored["model"]["model"] == "qwen3.5:9b"
    assert stored["model"]["options"]["seed"] == 42
    assert set(stored["hashes"]) == {
        "retrieval_prompt",
        "decide_prompt",
        "tool_descriptions",
        "golden",
        "evaluation_code",
        "gazetteer",
        "evaluation_rules",
    }
    assert stored["shuffle_seed"] == 42
    assert stored["reference_date"] == "2025-12-15"


class TestWhatMayBePublished:
    def test_a_path_of_this_machine_never_reaches_a_run_record(self, tmp_path: Path) -> None:
        reason = f"ValueError: {ROOT}/results/raw/x/tool_calls.jsonl:1 and {Path.home()}/.ollama/models/blob"
        ended = STARTED + timedelta(minutes=30)
        write_run_record(tmp_path, run_record(status="aborted", finished_at=ended, aborted_reason=reason))
        text = (tmp_path / RUN_FILE).read_text(encoding="utf-8")
        assert str(ROOT) not in text
        assert str(Path.home()) not in text
        assert read_run_record(tmp_path).aborted_reason == (
            "ValueError: <repository>/results/raw/x/tool_calls.jsonl:1 and ~/.ollama/models/blob"
        )

    def test_a_path_of_this_machine_never_reaches_an_attempt_record(self, tmp_path: Path) -> None:
        reason = f"turn 1 raised OSError: [Errno 13] Permission denied: '{ROOT}/x'"
        turn = TurnRecord(trace_id="t01-a1-u1", seconds=1.0, error=reason)
        record = AttemptRecord(
            run_id=RUN_ID,
            task_id="dev-s2-milan-march",
            attempt=1,
            outcome="errored",
            passed=False,
            reason=reason,
            turns=[turn],
        )
        append_attempt(tmp_path, record)
        assert str(ROOT) not in (tmp_path / ATTEMPTS_FILE).read_text(encoding="utf-8")
        (read,) = read_attempts(tmp_path)
        assert read.reason == "turn 1 raised OSError: [Errno 13] Permission denied: '<repository>/x'"

    def test_a_home_folder_at_the_root_rewrites_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: Path("/"))
        write_run_record(tmp_path, run_record())
        assert read_run_record(tmp_path) == run_record()

    @pytest.mark.parametrize(
        ("url", "recorded"),
        [
            ("http://user:s3cret@127.0.0.1:11434", "http://127.0.0.1:11434"),
            ("https://token@models.example/v1", "https://models.example/v1"),
            ("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
        ],
    )
    def test_a_password_in_a_url_is_left_out(self, url: str, recorded: str) -> None:
        config = Settings(_env_file=None, ollama_url=url)  # type: ignore[call-arg]  # a pydantic-settings init option
        assert recorded_settings(config)["ollama_url"] == recorded


def test_an_attempt_whose_writing_was_cut_short_did_not_finish(tmp_path: Path) -> None:
    line = attempt().model_dump_json()
    (tmp_path / ATTEMPTS_FILE).write_text(f"{line}\n{line[:40]}", encoding="utf-8")
    assert len(read_attempts(tmp_path)) == 1


@pytest.mark.parametrize("name", ["run.json", "attempts.jsonl"])
def test_a_record_that_is_not_utf8_text_is_a_record_error(tmp_path: Path, name: str) -> None:
    write_run_record(tmp_path, run_record())
    (tmp_path / name).write_bytes(b"\xff\xfe\x00")
    with pytest.raises(RecordError, match=rf"{name} is not UTF-8 text \(byte 0\)"):
        read_run_record(tmp_path) if name == "run.json" else read_attempts(tmp_path)


def test_a_bad_attempt_line_names_its_number(tmp_path: Path) -> None:
    append_attempt(tmp_path, attempt())
    with (tmp_path / "attempts.jsonl").open("a", encoding="utf-8") as out:
        out.write('{"not": "an attempt"}\n')
    with pytest.raises(RecordError) as raised:
        read_attempts(tmp_path)
    assert raised.value.line == 2


def test_the_warm_up_time_is_optional_and_never_negative() -> None:
    assert run_record().warm_up_seconds is None  # records made before the warm-up still read
    assert run_record(warm_up_seconds=1.5).warm_up_seconds == 1.5
    with pytest.raises(ValidationError, match="warm_up_seconds"):
        run_record(warm_up_seconds=-0.1)
