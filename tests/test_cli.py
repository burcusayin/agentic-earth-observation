"""Tests for `eo-agent chat`: the typed lines are scripted, the printed lines kept. No server, no model."""

from __future__ import annotations

import asyncio
import json
import signal
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from agentic_earth_observation import cli
from agentic_earth_observation.catalog.gazetteer import GazetteerError
from agentic_earth_observation.config import Settings
from agentic_earth_observation.evals.golden import GoldenError, golden_hash, load_golden
from agentic_earth_observation.evals.model_server import ModelServerError
from agentic_earth_observation.evals.records import (
    AttemptRecord,
    CodeVersion,
    Hashes,
    ModelBuild,
    ProvenanceError,
    RunRecord,
    TaskCounts,
)
from agentic_earth_observation.evals.run_eval import MODELS_UNDER_EVALUATION, EvaluationError
from agentic_earth_observation.evals.servers import ServerError
from agentic_earth_observation.orchestrator.client import RetrievalAgentError
from agentic_earth_observation.orchestrator.graph import Orchestrator
from agentic_earth_observation.retrieval_agent.models import PreviousSearch, RetrievalResult
from tests.fakes import NEVER_FINISHES, SCENE, FakeClient, FakeOllama, OllamaStep, ollama_tool_call
from tests.test_eval_records import turn as report_turn
from tests.test_eval_report import hashes as report_hashes
from tests.test_eval_report import write_attempts

pytestmark = pytest.mark.anyio

TODAY = date(2025, 12, 15)
NAPLES_WEEK = PreviousSearch.model_validate_json(
    json.dumps(
        {
            "place": "Naples",
            "product": "sentinel-1-grd",
            "bbox": [14.1321, 40.7912, 14.3538, 40.9161],
            "start_date": "2025-09-01",
            "end_date": "2025-09-07",
        }
    )
)
FOUND = RetrievalResult(status="ok", scenes=[SCENE], search=NAPLES_WEEK)
DELEGATE = ollama_tool_call("Decision", route="delegate", reason="search")
DECLINE = ollama_tool_call("Decision", route="decline", reason="out_of_scope")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def settings(**changes: Any) -> Settings:  # noqa: ANN401 - any setting, validated by Settings itself
    return Settings(_env_file=None, **changes)  # type: ignore[call-arg]  # a pydantic-settings init option


def typed(*lines: str) -> Callable[[str], str]:
    """What the user types, line by line, then Ctrl+D (EOFError)."""
    remaining: Iterator[str] = iter(lines)

    def ask(prompt: str) -> str:
        assert prompt == "> "
        try:
            return next(remaining)
        except StopIteration:
            raise EOFError from None

    return ask


async def chat(
    lines: list[str], decisions: list[OllamaStep], client: FakeClient, *, config: Settings | None = None
) -> tuple[list[str], FakeOllama]:
    """Run a chat; return what it printed and the Ollama server it asked."""
    shown: list[str] = []
    ollama_server = FakeOllama(*decisions)
    config = config or settings()
    await cli.chat(ollama_server.model(), client, config, TODAY, ask=typed(*lines), say=shown.append)
    return shown, ollama_server


async def test_each_line_is_a_turn_of_one_conversation() -> None:
    client = FakeClient(FOUND, FOUND)
    lines = [
        "Find radar scenes over Naples for the first week of September 2025.",
        "And the week after that?",
    ]
    shown, _ = await chat(lines, [DELEGATE, DELEGATE], client)
    (_, _), (second, second_trace) = client.calls
    assert second.previous_search == NAPLES_WEEK  # the follow-up continues the first search
    conversation = second_trace.rsplit("-", 1)[0]
    assert [trace for _, trace in client.calls] == [f"{conversation}-1", f"{conversation}-2"]
    answers = [line for line in shown if SCENE.id in line and not line.startswith("  [")]
    assert len(answers) == 2
    status = [line for line in shown if line.startswith("  [")]
    assert status[0].startswith(f"  [route delegate · status ok · trace {conversation}-1 · ")
    assert status[0].endswith(" s]")


async def test_the_chat_says_what_it_is_and_how_to_end_it() -> None:
    shown, _ = await chat([], [], FakeClient())
    assert (
        shown[0]
        == "Earth Observation assistant (qwen3.5:9b; today is 2025-12-15; agent at http://127.0.0.1:8002)."
    )
    assert '"quit" or Ctrl+D ends the chat' in shown[1]


@pytest.mark.parametrize("last", ["quit", "exit", "QUIT"])
async def test_quit_ends_the_chat_before_anything_more_is_asked(last: str) -> None:
    shown, ollama_server = await chat([last, "Naples radar"], [DELEGATE], FakeClient(FOUND))
    assert ollama_server.requests == []
    assert len(shown) == 2  # the greeting only


async def test_empty_lines_are_skipped() -> None:
    client = FakeClient()
    shown, ollama_server = await chat(["", "   ", "What will the weather be like?"], [DECLINE], client)
    assert len(ollama_server.requests) == 1
    assert not any("cannot be sent" in line for line in shown)  # skipped quietly, not refused
    assert any(line.startswith("  [route decline · status declined · trace ") for line in shown)
    assert shown[-1] == ""  # Ctrl+D: a new line, so the terminal's prompt starts clean


async def test_a_line_that_cannot_be_sent_is_explained_and_the_chat_goes_on() -> None:
    shown, _ = await chat(["x" * 2001, "What will the weather be like?"], [DECLINE], FakeClient())
    assert "  (the text cannot be sent: String should have at most 2000 characters)" in shown
    assert any(line.startswith("  [route decline") for line in shown)


def a_plain_value_error() -> None:
    raise ValueError("a bug inside the turn")


def an_invalid_result() -> None:
    RetrievalResult.model_validate({"status": "nonsense"})  # pydantic's ValidationError, a ValueError too


@pytest.mark.parametrize(
    ("bug", "raised", "message"),
    [
        (a_plain_value_error, ValueError, "a bug inside the turn"),
        (an_invalid_result, ValidationError, "status"),
    ],
)
async def test_a_bug_inside_a_turn_is_not_taken_for_a_refused_line(
    monkeypatch: pytest.MonkeyPatch, bug: Callable[[], None], raised: type[ValueError], message: str
) -> None:
    # Both are ValueErrors, like a refused line, but they come from inside the turn: a bug, never hidden.
    async def failing_turn(*args: object) -> None:
        bug()

    monkeypatch.setattr(Orchestrator, "run_turn", failing_turn)
    with pytest.raises(raised, match=message):
        await chat(["Naples radar"], [], FakeClient())


async def test_an_error_of_the_retrieval_agent_is_shown_and_the_chat_goes_on() -> None:
    bug = RetrievalAgentError("the retrieval agent ended turn t as TASK_STATE_FAILED: boom")
    shown, _ = await chat(["Naples radar", "The weather?"], [DELEGATE, DECLINE], FakeClient(bug))
    assert (
        "  (the retrieval agent failed: the retrieval agent ended turn t as TASK_STATE_FAILED: boom)" in shown
    )
    assert any(line.startswith("  [route decline") for line in shown)


async def test_the_status_line_shows_a_model_failure_and_infrastructure() -> None:
    down = RetrievalResult(
        status="not_searched", infra_error="A2A ConnectError: All connection attempts failed"
    )
    shown, _ = await chat(
        ["Hmm", "Naples radar"], [{"role": "assistant", "content": "No idea."}, DELEGATE], FakeClient(down)
    )
    status = [line for line in shown if line.startswith("  [")]
    assert " · model failure parse_error: the model answered without a Decision · " in status[0]
    assert " · infrastructure: A2A ConnectError: All connection attempts failed · " in status[1]


async def test_the_decision_has_the_deadline_of_the_settings() -> None:
    config = settings(model_timeout_s=0.1, decision_deadline_s=0.3)
    async with asyncio.timeout(5):  # without the deadline the turn would never end: fail, do not hang
        shown, _ = await chat(["Naples radar"], [NEVER_FINISHES], FakeClient(), config=config)
    status = [line for line in shown if line.startswith("  [")]
    assert " · infrastructure: the decision ran past its deadline (0.3 s) · " in status[0]


def test_the_command_chats_with_the_running_servers_on_the_date_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[Settings, date]] = []

    async def chat_with(config: Settings, today: date) -> None:
        seen.append((config, today))

    monkeypatch.setattr(cli, "_chat_with_the_running_servers", chat_with)
    config = settings()
    assert cli.main(["chat", "--today", "2025-12-15"], settings=config) == 0
    assert seen == [(config, TODAY)]


def test_without_a_date_the_chat_takes_todays(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[date] = []

    async def chat_with(config: Settings, today: date) -> None:
        seen.append(today)

    monkeypatch.setattr(cli, "_chat_with_the_running_servers", chat_with)
    config = settings(reference_date=date(2025, 6, 1))
    cli.main(["chat"], settings=config)
    assert seen == [date(2025, 6, 1)]


async def test_ctrl_c_at_the_prompt_ends_the_chat() -> None:
    # `input` blocks the event loop, so without care the Ctrl+C would only take effect after the next line.
    def interrupted(prompt: str) -> str:
        signal.raise_signal(signal.SIGINT)
        raise AssertionError("Ctrl+C did not interrupt the wait for a line")

    client = FakeClient()
    said: list[str] = []
    await cli.chat(FakeOllama().model(), client, settings(), TODAY, ask=interrupted, say=said.append)
    assert said[-1] == ""
    assert client.calls == []


def test_ctrl_c_at_the_prompt_leaves_the_handler_as_it_was() -> None:
    def asyncios_own(signum: int, frame: object) -> None:
        raise AssertionError("no Ctrl+C was sent")

    before = signal.signal(signal.SIGINT, asyncios_own)
    try:
        assert cli._read_line(lambda prompt: "hello") == "hello"
        assert signal.getsignal(signal.SIGINT) is asyncios_own
    finally:
        signal.signal(signal.SIGINT, before)


def test_ctrl_c_during_a_turn_of_the_chat_ends_it_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def chat_with(config: Settings, today: date) -> None:
        signal.raise_signal(signal.SIGINT)
        await asyncio.sleep(30)

    monkeypatch.setattr(cli, "_chat_with_the_running_servers", chat_with)
    assert cli.main(["chat"], settings=settings()) == 130


def run_record(status: str = "complete", reason: str | None = None) -> RunRecord:
    return RunRecord(
        run_id="20261107T093045Z-qwen3.5-9b-r1",
        status=status,  # type: ignore[arg-type]  # the tests pass valid statuses as plain text
        aborted_reason=reason,
        split="dev",
        run_index=1,
        started_at=datetime(2026, 11, 7, 9, 30, 45, tzinfo=UTC),
        finished_at=datetime(2026, 11, 7, 9, 40, 45, tzinfo=UTC),
        code=CodeVersion(commit="0123456789abcdef0123456789abcdef01234567", dirty=False),
        model=ModelBuild(provider="ollama", model="qwen3.5:9b", options={"seed": 42}),
        hashes=Hashes(
            retrieval_prompt="ab" * 32,
            decide_prompt="ab" * 32,
            tool_descriptions="ab" * 32,
            golden="ab" * 32,
            evaluation_code="ab" * 32,
            gazetteer="ab" * 32,
            evaluation_rules="ab" * 32,
        ),
        reference_date=TODAY,
        shuffle_seed=1,
        task_order=["dev-s2-milan-march"],
        settings={"num_predict": 1024},
        libraries={"python": "3.13"},
        platform="test",
    )


class TestEval:
    def evaluation(
        self, monkeypatch: pytest.MonkeyPatch, outcome: list[RunRecord] | Exception
    ) -> list[dict[str, Any]]:
        """Replace the evaluation by one that records its options and ends with `outcome`."""
        seen: list[dict[str, Any]] = []

        async def evaluate(config: Settings, **options: Any) -> list[RunRecord]:  # noqa: ANN401 - its options
            seen.append({"settings": config, **options})
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(cli, "evaluate", evaluate)
        return seen

    def test_the_options_reach_the_evaluation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = self.evaluation(monkeypatch, [run_record()])
        config = settings()
        argv = [
            "eval",
            "--split",
            "dev",
            "--models",
            " qwen3.5:9b, ministral-3:8b ,",
            "--runs",
            "2",
            "--for-report",
            "--shuffle-seed",
            "7",
        ]
        assert cli.main(argv, settings=config) == 0
        assert seen == [
            {
                "settings": config,
                "split": "dev",
                "models": ["qwen3.5:9b", "ministral-3:8b"],
                "runs": 2,
                "for_report": True,
                "shuffle_seed": 7,
            }
        ]

    def test_the_defaults_are_every_model_three_runs_scratch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = self.evaluation(monkeypatch, [run_record()])
        assert cli.main(["eval", "--split", "test"], settings=settings()) == 0
        assert seen[0]["models"] == list(MODELS_UNDER_EVALUATION)
        assert (seen[0]["runs"], seen[0]["for_report"], seen[0]["shuffle_seed"]) == (3, False, 1)

    def test_each_run_is_reported_with_how_it_ended(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self.evaluation(monkeypatch, [run_record()])
        assert cli.main(["eval", "--split", "dev"], settings=settings()) == 0
        assert capsys.readouterr().out == (
            "20261107T093045Z-qwen3.5-9b-r1: complete (0 passed, 0 failed, 0 errored, 0 lost)\n"
        )

    def test_an_aborted_run_is_reported_and_the_command_fails(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self.evaluation(monkeypatch, [run_record("aborted", "3 tasks errored")])
        assert cli.main(["eval", "--split", "dev"], settings=settings()) == 1
        assert capsys.readouterr().out == (
            "20261107T093045Z-qwen3.5-9b-r1: aborted: 3 tasks errored "
            "(0 passed, 0 failed, 0 errored, 0 lost)\n"
        )

    @pytest.mark.parametrize(
        "error",
        [
            EvaluationError("not under evaluation: gpt-5"),
            GoldenError("dev.jsonl:1: invalid"),
            ProvenanceError("git is not installed"),
            ServerError("something already answers on port 8001"),
            ModelServerError("Ollama cannot be reached"),
        ],
    )
    def test_what_stops_the_evaluation_is_said_without_a_traceback(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], error: Exception
    ) -> None:
        self.evaluation(monkeypatch, error)
        assert cli.main(["eval", "--split", "dev"], settings=settings()) == 1
        assert capsys.readouterr().err == f"error: {error}\n"

    def test_a_bug_in_the_evaluation_is_not_hidden(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.evaluation(monkeypatch, RuntimeError("a bug"))
        with pytest.raises(RuntimeError, match="a bug"):
            cli.main(["eval", "--split", "dev"], settings=settings())

    @pytest.mark.parametrize("stop", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
    def test_a_stop_signal_ends_the_evaluation_in_order(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], stop: signal.Signals
    ) -> None:
        # The real signal, through the real asyncio.run: the evaluation is cancelled, so that it can write
        # down how the run ended and stop the servers. Ctrl+C is asyncio's; a `kill` or closed terminal ours.
        ended: list[str] = []

        async def evaluate(config: Settings, **options: Any) -> list[RunRecord]:  # noqa: ANN401 - its options
            signal.raise_signal(stop)
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                ended.append("cancelled")
                raise
            return []

        monkeypatch.setattr(cli, "evaluate", evaluate)
        assert cli.main(["eval", "--split", "dev"], settings=settings()) == 130
        assert ended == ["cancelled"]
        assert "kept in its folder" in capsys.readouterr().err

    def test_a_signal_being_ignored_stays_ignored(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # `nohup eo-agent eval ...`: closing the terminal must not end a run that takes all night.
        async def evaluate(config: Settings, **options: Any) -> list[RunRecord]:  # noqa: ANN401 - its options
            signal.raise_signal(signal.SIGHUP)
            await asyncio.sleep(0.01)
            return [run_record()]

        monkeypatch.setattr(cli, "evaluate", evaluate)
        previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)
        try:
            assert cli.main(["eval", "--split", "dev"], settings=settings()) == 0
            assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
        finally:
            signal.signal(signal.SIGHUP, previous)

    @pytest.mark.parametrize(
        "argv", [["eval"], ["eval", "--split", "train"], ["eval", "--split", "dev", "--runs", "many"]]
    )
    def test_a_command_line_it_cannot_read_is_refused(
        self, argv: list[str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exited:
            cli.main(argv, settings=settings())
        assert exited.value.code == 2
        assert "usage: eo-agent eval" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [[], ["chat", "--today", "15/12/2025"], ["serve"]])
def test_a_command_line_it_cannot_read_is_refused(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exited:
        cli.main(argv, settings=settings())
    assert exited.value.code == 2
    assert "usage: eo-agent" in capsys.readouterr().err


async def test_the_real_chat_closes_its_client_without_any_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ctrl+D at once: the A2A client is made and closed, and nothing is sent anywhere.
    def ended(prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", ended)
    monkeypatch.setattr("builtins.print", lambda *args: None)
    await cli._chat_with_the_running_servers(settings(), TODAY)


class TestReport:
    """`eo-agent report`: it reads records only, so it needs no settings, server or model."""

    def dev_run(self, runs: Path, *, dirty: bool) -> None:
        """A passed run of the real dev split: every task with as many scored turns as it has."""
        dev = [task for task in load_golden() if task.split == "dev"]
        write_attempts(
            runs,
            "r1",
            [
                AttemptRecord(
                    run_id="r1",
                    task_id=task.id,
                    attempt=1,
                    outcome="scored",
                    passed=True,
                    turns=[report_turn() for _ in task.turns],
                )
                for task in dev
            ],
            task_order=[task.id for task in dev],
            tasks=TaskCounts(passed=len(dev)),
            hashes=report_hashes(golden=golden_hash()),
            code=CodeVersion(commit="0123456789abcdef0123456789abcdef01234567", dirty=dirty),
        )

    def test_before_the_first_run_for_the_report_it_says_so_and_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli, "REPORT_RUNS", tmp_path / "runs")  # the default folder, not made yet
        assert cli.main(["report", "--split", "test"]) == 0
        out = capsys.readouterr().out
        assert out.startswith("# Evaluation report: test split\n")
        assert "No run counts yet." in out

    def test_a_run_from_a_checkout_with_changes_counts_only_when_allowed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self.dev_run(tmp_path, dirty=True)
        assert cli.main(["report", "--split", "dev", "--runs-dir", str(tmp_path)]) == 0
        assert "| r1 | dirty checkout |" in capsys.readouterr().out
        assert cli.main(["report", "--split", "dev", "--runs-dir", str(tmp_path), "--allow-dirty"]) == 0
        out = capsys.readouterr().out
        assert "| r1 | counted |" in out
        assert "not for publication" in out
        assert "| qwen3.5:9b | 4.0 of 4 (" in out

    def test_the_report_is_written_to_a_file_when_asked(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self.dev_run(tmp_path / "runs", dirty=False)
        target = tmp_path / "out" / "summary.md"
        assert (
            cli.main(
                ["report", "--split", "dev", "--runs-dir", str(tmp_path / "runs"), "--output", str(target)]
            )
            == 0
        )
        assert capsys.readouterr().out == f"report written to {target}\n"
        assert "| r1 | counted |" in target.read_text(encoding="utf-8")

    def test_a_folder_of_runs_that_is_a_file_is_an_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "runs").write_text("", encoding="utf-8")
        assert cli.main(["report", "--split", "dev", "--runs-dir", str(tmp_path / "runs")]) == 1
        captured = capsys.readouterr()
        assert (captured.out, captured.err) == (
            "",
            "error: runs is not a folder: give the folder that holds the runs' folders\n",
        )

    def test_a_report_that_cannot_be_written_is_an_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "taken").write_text("", encoding="utf-8")
        target = tmp_path / "taken" / "summary.md"  # its folder is a file
        assert (
            cli.main(["report", "--split", "dev", "--runs-dir", str(tmp_path), "--output", str(target)]) == 1
        )
        assert capsys.readouterr().err.startswith(f"error: the report cannot be written to {target}: ")

    def test_a_folder_of_runs_given_that_does_not_exist_is_an_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        typo = tmp_path / "rnus"
        assert cli.main(["report", "--split", "dev", "--runs-dir", str(typo)]) == 1
        assert capsys.readouterr().err == f"error: there is no folder {typo}: check --runs-dir\n"

    def test_a_gazetteer_that_cannot_be_read_is_an_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def broken() -> None:
            raise GazetteerError("data/gazetteer.json is not a gazetteer")

        monkeypatch.setattr(cli, "load_golden", broken)
        assert cli.main(["report", "--split", "dev", "--runs-dir", str(tmp_path)]) == 1
        assert capsys.readouterr().err == "error: data/gazetteer.json is not a gazetteer\n"

    def test_a_folder_that_cannot_be_read_is_an_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def denied(*args: object, **kwargs: object) -> None:
            raise PermissionError(13, "Permission denied", str(tmp_path))

        monkeypatch.setattr(cli, "collect_runs", denied)
        assert cli.main(["report", "--split", "dev", "--runs-dir", str(tmp_path)]) == 1
        assert capsys.readouterr().err == f"error: {tmp_path} cannot be read: Permission denied\n"

    def test_it_reads_no_settings(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def no_settings() -> None:
            raise AssertionError("the report needs no settings")

        monkeypatch.setattr(cli, "Settings", no_settings)
        assert cli.main(["report", "--split", "dev", "--runs-dir", str(tmp_path)]) == 0
