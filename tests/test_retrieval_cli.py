"""Tests for the retrieval agent's command line (the smoke test). The turn is faked: no server, no model."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from agentic_earth_observation.catalog_mcp.tool_log import ToolCallLog, ToolCallRecord
from agentic_earth_observation.config import Settings
from agentic_earth_observation.retrieval_agent import __main__ as cli
from agentic_earth_observation.retrieval_agent.models import RetrievalRequest, RetrievalResult
from tests.fakes import SCENE

QUESTION = "Show me Sentinel-2 scenes of Milan from March 2025 with little cloud."


def settings(log: Path) -> Settings:
    return Settings(_env_file=None, tool_log=log, reference_date=date(2025, 12, 15))  # type: ignore[call-arg]  # a pydantic-settings init option


class FakeTurn:
    """Stands in for run_retrieval: records what it was given, logs two calls as the server would."""

    def __init__(self, log: Path, result: RetrievalResult) -> None:
        self.log = log
        self.result = result
        self.calls: list[tuple[RetrievalRequest, str, BaseChatModel, Settings]] = []

    async def __call__(
        self,
        request: RetrievalRequest,
        trace_id: str,
        model: BaseChatModel,
        settings: Settings,
        **_: Any,  # noqa: ANN401 - unused options
    ) -> RetrievalResult:
        self.calls.append((request, trace_id, model, settings))
        log = ToolCallLog(self.log)
        log.write(
            ToolCallRecord(
                trace_id=trace_id, request_id=1, tool="resolve_place", arguments={"name": "Milan"}, ok=True
            )
        )
        log.write(
            ToolCallRecord(
                trace_id="another-turn",
                request_id=1,
                tool="resolve_place",
                arguments={"name": "Rho"},
                ok=True,
            )
        )
        log.write(
            ToolCallRecord(
                trace_id=trace_id,
                request_id=2,
                tool="search_scenes",
                arguments={"product": "sentinel-2-l2a"},
                ok=False,
                error_kind="invalid_arguments",
            )
        )
        log.write(
            ToolCallRecord(
                trace_id=trace_id,
                request_id=3,
                tool="search_scenes",
                arguments={"product": "sentinel-2-l2a", "limit": 1},
                ok=True,
                returned_ids=[SCENE.id],
            )
        )
        return self.result


def test_the_question_runs_one_turn_and_prints_the_result_and_its_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "calls.jsonl"
    turn = FakeTurn(log, RetrievalResult(status="error"))
    monkeypatch.setattr(cli, "run_retrieval", turn)
    assert cli.main([QUESTION], settings=settings(log)) == 0
    ((request, trace_id, model, used),) = turn.calls
    assert (request.text, request.today, request.previous_search) == (QUESTION, date(2025, 12, 15), None)
    assert trace_id.startswith("smoke-")
    assert isinstance(model, ChatOllama)
    assert model.model == used.llm_model
    out = capsys.readouterr().out
    assert out.startswith(f"trace id: {trace_id}\n")
    assert '"status": "error"' in out
    assert f"  1. resolve_place {json.dumps({'name': 'Milan'})}: ok\n" in out  # a lookup returns no scenes
    assert '  2. search_scenes {"product": "sentinel-2-l2a"}: failed (invalid_arguments)\n' in out
    assert '  3. search_scenes {"product": "sentinel-2-l2a", "limit": 1}: ok, 1 scene id(s)\n' in out
    assert "Rho" not in out  # another turn's call


def test_each_run_gets_its_own_trace_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = tmp_path / "calls.jsonl"
    turn = FakeTurn(log, RetrievalResult(status="not_searched"))
    monkeypatch.setattr(cli, "run_retrieval", turn)
    cli.main([QUESTION], settings=settings(log))
    cli.main([QUESTION], settings=settings(log))
    first, second = (trace_id for _, trace_id, _, _ in turn.calls)
    assert first != second


def test_an_infrastructure_failure_exits_with_status_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "calls.jsonl"
    failed = RetrievalResult(
        status="not_searched", infra_error="ConnectError: All connection attempts failed"
    )
    monkeypatch.setattr(cli, "run_retrieval", FakeTurn(log, failed))
    assert cli.main([QUESTION], settings=settings(log)) == 1


def test_a_missing_log_is_explained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class NoLog(FakeTurn):
        async def __call__(self, *args: object, **kwargs: object) -> RetrievalResult:
            return self.result  # and writes no log

    log = tmp_path / "elsewhere.jsonl"
    monkeypatch.setattr(cli, "run_retrieval", NoLog(log, RetrievalResult(status="not_searched")))
    cli.main([QUESTION], settings=settings(log))
    assert (
        f"no tool-call log at {log}: does the MCP server use the same EO_TOOL_LOG?" in capsys.readouterr().out
    )


@pytest.mark.parametrize("question", ["", "   ", "x" * 2001])
def test_a_question_that_cannot_be_sent_is_refused_before_anything_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], question: str
) -> None:
    turn = FakeTurn(tmp_path / "calls.jsonl", RetrievalResult(status="not_searched"))
    monkeypatch.setattr(cli, "run_retrieval", turn)
    with pytest.raises(SystemExit) as exited:
        cli.main([question], settings=settings(tmp_path / "calls.jsonl"))
    assert exited.value.code == 2
    assert "the question cannot be sent" in capsys.readouterr().err
    assert turn.calls == []


def test_a_found_scene_is_printed_with_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "calls.jsonl"
    search = {
        "place": "Milan",
        "product": "sentinel-2-l2a",
        "bbox": [9.0409, 45.3867, 9.2781, 45.5358],
        "start_date": "2025-03-01",
        "end_date": "2025-03-31",
        "max_cloud_cover": 20,
    }
    found = RetrievalResult.model_validate_json(
        json.dumps({"status": "ok", "scenes": [SCENE.model_dump(mode="json")], "search": search})
    )
    monkeypatch.setattr(cli, "run_retrieval", FakeTurn(log, found))
    cli.main([QUESTION], settings=settings(log))
    assert SCENE.id in capsys.readouterr().out
