"""Tests for `eo-agent chat`: the typed lines are scripted, the printed lines kept. No server, no model."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from agentic_earth_observation import cli
from agentic_earth_observation.config import Settings
from agentic_earth_observation.orchestrator.client import RetrievalAgentError
from agentic_earth_observation.orchestrator.graph import Orchestrator
from agentic_earth_observation.retrieval_agent.models import PreviousSearch, RetrievalResult
from tests.fakes import SCENE, FakeClient, FakeOllama, OllamaStep, ollama_tool_call

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
    lines: list[str], decisions: list[OllamaStep], client: FakeClient
) -> tuple[list[str], FakeOllama]:
    """Run a chat; return what it printed and the Ollama server it asked."""
    shown: list[str] = []
    ollama_server = FakeOllama(*decisions)
    await cli.chat(ollama_server.model(), client, settings(), TODAY, ask=typed(*lines), say=shown.append)
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
