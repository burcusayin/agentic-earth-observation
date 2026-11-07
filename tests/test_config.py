"""Tests for the settings and the model factory (no network: building a model makes no call).

conftest.py removes every EO_* variable before each test, so the developer's shell cannot change the results.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, tzinfo
from pathlib import Path
from typing import Any

import ollama
import pytest
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from agentic_earth_observation import config
from agentic_earth_observation.config import ROOT, Settings, is_local_url
from agentic_earth_observation.llm import (
    OLLAMA_CONTEXT,
    SEED,
    build_chat_model,
    model_seconds,
    model_server_failed,
    unreadable_reply,
)


def settings(**overrides: Any) -> Settings:  # noqa: ANN401 - forwarded to Settings unchanged
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]  # a pydantic-settings init option


def test_defaults_are_local_and_safe() -> None:
    s = settings()
    assert s.llm_provider == "ollama"
    assert s.ollama_url == "http://127.0.0.1:11434"
    assert s.mcp_url == "http://127.0.0.1:8001/mcp"
    assert s.a2a_url == "http://127.0.0.1:8002"
    assert s.mistral_api_key is None
    assert s.tool_log == ROOT / "results" / "raw" / "tool_calls.jsonl"


def test_the_repository_root_is_where_pyproject_is() -> None:
    assert (ROOT / "pyproject.toml").is_file()


def test_environment_variables_use_the_eo_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EO_LLM_MODEL", "ministral-3:8b")
    monkeypatch.setenv("EO_REFERENCE_DATE", "2025-12-15")
    s = settings()
    assert s.llm_model == "ministral-3:8b"
    assert s.today() == date(2025, 12, 15)


def test_the_key_is_never_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EO_MISTRAL_API_KEY", "not-a-real-key-123456")
    s = settings()
    assert "not-a-real-key" not in repr(s)
    assert s.mistral_api_key is not None
    assert s.mistral_api_key.get_secret_value() == "not-a-real-key-123456"


@pytest.mark.parametrize("name", ["EO_MISTRAL_API_KEY", "EO_TOOL_LOG", "EO_REFERENCE_DATE", "EO_NUM_PREDICT"])
def test_an_empty_variable_means_the_default(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    defaults = settings()
    monkeypatch.setenv(name, "")  # as copied from .env.example
    assert settings() == defaults


def test_a_misspelt_environment_variable_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EO_LLM_MODLE", "ministral-3:8b")
    with pytest.raises(ValidationError, match=r"unknown environment variable.*EO_LLM_MODLE"):
        settings()


def test_a_misspelt_key_in_the_env_file_is_refused_and_other_keys_are_ignored(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("EO_LLM_MODEL=ministral-3:8b\nOTHER_TOOL_SETTING=1\n", encoding="utf-8")
    assert Settings(_env_file=env).llm_model == "ministral-3:8b"  # type: ignore[call-arg]  # init option
    env.write_text("EO_LLM_MODLE=ministral-3:8b\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="llm_modle"):
        Settings(_env_file=env)  # type: ignore[call-arg]  # a pydantic-settings init option


def test_errors_never_show_a_settings_value(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("EO_MISTRAL_KEY=not-a-real-key-123456\n", encoding="utf-8")  # misspelt: refused
    with pytest.raises(ValidationError) as refused:
        Settings(_env_file=env)  # type: ignore[call-arg]  # a pydantic-settings init option
    assert "mistral_key" in str(refused.value)
    assert "not-a-real-key" not in str(refused.value)


@pytest.mark.parametrize("value", ["null", "None", "off", " false "])
def test_the_tool_log_cannot_be_turned_off_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("EO_TOOL_LOG", value)
    with pytest.raises(ValidationError, match="cannot be turned off from the environment"):
        settings()


def test_a_relative_tool_log_path_starts_at_the_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EO_TOOL_LOG", "results/raw/run-1.jsonl")
    assert settings().tool_log == ROOT / "results" / "raw" / "run-1.jsonl"
    assert settings(tool_log=tmp_path / "calls.jsonl").tool_log == tmp_path / "calls.jsonl"
    assert settings(tool_log=None).tool_log is None  # off, from code only


def test_settings_cannot_be_changed_after_they_are_read() -> None:
    s = settings()
    with pytest.raises(ValidationError):
        s.llm_model = "something-else"  # frozen: refused


def test_the_default_timeouts_nest() -> None:
    s = settings()
    assert s.stac_timeout_s < s.stac_deadline_s < s.mcp_timeout_s < s.turn_deadline_s
    assert s.turn_deadline_s < s.a2a_timeout_s < s.harness_timeout_s
    assert s.model_timeout_s < s.turn_deadline_s
    assert s.model_timeout_s < s.decision_deadline_s
    assert s.decision_deadline_s + s.a2a_timeout_s < s.harness_timeout_s  # a user turn: decision, then A2A


@pytest.mark.parametrize(
    ("overrides", "link"),
    [
        ({"stac_timeout_s": 90.0}, "one STAC request (90 s) must time out before the STAC deadline (90 s)"),
        ({"stac_deadline_s": 120.0}, "STAC deadline (120 s) must be below MCP (120 s)"),
        ({"mcp_timeout_s": 240.0}, "MCP (240 s) must be below retrieval turn (240 s)"),
        ({"turn_deadline_s": 300.0}, "retrieval turn (300 s) must be below A2A (300 s)"),
        ({"a2a_timeout_s": 480.0}, "A2A (480 s) must be below harness (480 s)"),
        (
            {"model_timeout_s": 240.0},
            "the model server's silence limit (240 s) must be below the retrieval turn (240 s)",
        ),
        (
            {"model_timeout_s": 150.0},
            "the model server's silence limit (150 s) must be below the decision deadline (150 s)",
        ),
        (
            {"a2a_timeout_s": 330.0},
            "the decision (150 s) and A2A (330 s) must end before the harness (480 s)",
        ),
        (
            {"decision_deadline_s": 180.0},
            "the decision (180 s) and A2A (300 s) must end before the harness (480 s)",
        ),
    ],
)
def test_each_timeout_must_be_below_the_next_one(overrides: dict[str, float], link: str) -> None:
    with pytest.raises(ValidationError, match=re.escape(link)):
        settings(**overrides)  # equal is not below


@pytest.mark.parametrize(
    "overrides",
    [
        {"stac_timeout_s": 89.9},
        {"stac_deadline_s": 119.9},
        {"mcp_timeout_s": 239.9},
        {"turn_deadline_s": 299.9},
        {"a2a_timeout_s": 329.9},  # below the harness (480 s) less the decision (150 s)
        {"model_timeout_s": 149.9},
        {"model_timeout_s": 239.9, "decision_deadline_s": 240.0, "harness_timeout_s": 540.1},
        {"decision_deadline_s": 179.9},
    ],
)
def test_a_timeout_just_below_the_next_one_is_accepted(overrides: dict[str, float]) -> None:
    settings(**overrides)


def test_the_env_file_is_the_repositorys_wherever_a_process_starts() -> None:
    # The MCP server, the retrieval agent and the chat may start in different folders; they read one file.
    assert Settings.model_config["env_file"] == ROOT / ".env"


@pytest.mark.parametrize(
    "overrides",
    [
        {"stac_timeout_s": 0.0},
        {"stac_deadline_s": 0.0},
        {"stac_max_retries": -1},
        {"max_model_calls": 3},  # resolve, search, one retry and a reply need 4
        {"num_predict": 63},
        {"llm_provider": "openai"},
        {"no_such_setting": 1},
    ],
)
def test_out_of_range_settings_are_refused(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        settings(**overrides)


def test_the_smallest_allowed_limits_are_accepted() -> None:
    s = settings(stac_max_retries=0, max_model_calls=4, num_predict=64)
    assert (s.stac_max_retries, s.max_model_calls, s.num_predict) == (0, 4, 64)


def test_today_is_the_reference_date_or_else_the_date_in_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    class LateEvening:
        """23:30 in UTC on New Year's Eve: already 1 January east of UTC."""

        @staticmethod
        def now(tz: tzinfo | None = None) -> datetime:
            assert tz is UTC
            return datetime(2025, 12, 31, 23, 30, tzinfo=UTC)

    monkeypatch.setattr(config, "datetime", LateEvening)
    assert settings().today() == date(2025, 12, 31)
    assert settings(reference_date=date(2025, 12, 15)).today() == date(2025, 12, 15)


def test_env_example_lists_every_setting_and_nothing_else() -> None:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    listed = set(re.findall(r"^(EO_[A-Z0-9_]+)=", text, re.MULTILINE))
    assert listed == {f"EO_{name.upper()}" for name in Settings.model_fields}


def test_env_example_is_a_valid_configuration() -> None:
    s = Settings(_env_file=ROOT / ".env.example")  # type: ignore[call-arg]  # a pydantic-settings init option
    assert s == settings()  # its values are the defaults


def test_the_ollama_model_gets_the_options_of_the_decisions_log() -> None:
    s = settings(llm_model="ministral-3:8b")
    model = build_chat_model(s)
    assert isinstance(model, ChatOllama)
    assert model.model == "ministral-3:8b"
    assert model.base_url == s.ollama_url
    assert model.num_ctx == OLLAMA_CONTEXT
    assert model.temperature == 0
    assert model.seed == SEED
    assert model.reasoning is False
    assert model.keep_alive == "30m"
    assert model.num_predict == s.num_predict
    assert model.client_kwargs == {"timeout": s.model_timeout_s, "trust_env": False}


def test_a_model_server_elsewhere_may_go_through_the_proxy_settings() -> None:
    model = build_chat_model(settings(ollama_url="https://models.example:11434"))
    assert isinstance(model, ChatOllama)
    assert model.client_kwargs is not None
    assert model.client_kwargs["trust_env"] is True


@pytest.mark.parametrize(
    ("url", "local"),
    [
        ("http://127.0.0.1:11434", True),
        ("http://localhost:8002", True),
        ("https://earth-search.aws.element84.com/v1", False),
        ("http://localhost.evil.example:8001", False),
    ],
)
def test_a_local_url_names_this_machine(url: str, local: bool) -> None:
    assert is_local_url(url) is local


@pytest.mark.parametrize(
    "message",
    [
        "llama-server process no longer running: -1 ",
        "timed out waiting for llama-server to start: context deadline exceeded",
        "model is too large for system memory",
    ],
)
@pytest.mark.parametrize("status", [500, -1])
def test_the_model_servers_own_trouble_is_infrastructure(message: str, status: int) -> None:
    # Whatever the model wrote: its score must not pay for a machine short of memory.
    err = ollama.ResponseError(message, status)
    assert model_server_failed(err)
    assert unreadable_reply(err) is None


def test_words_of_the_model_quoted_by_the_parser_are_the_models_failure() -> None:
    err = ollama.ResponseError("error parsing tool call: raw='process no longer running', err=bad", 500)
    assert not model_server_failed(err)
    assert unreadable_reply(err) is not None


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        pytest.param(
            {
                "load_duration": 2_000_000_000,
                "prompt_eval_duration": 500_000_000,
                "eval_duration": 1_500_000_000,
            },
            (2.0, 2.0),
            id="all three",
        ),
        pytest.param({"eval_duration": 1_000_000_000}, (1.0, 0.0), id="no load"),
        pytest.param({}, (0.0, 0.0), id="from another server"),
        pytest.param({"eval_duration": "soon"}, (0.0, 0.0), id="not a number"),
    ],
)
def test_the_model_servers_clock_is_read_in_seconds(
    metadata: dict[str, object], expected: tuple[float, float]
) -> None:
    assert model_seconds(metadata) == expected


def test_mistral_is_not_wired_up_yet() -> None:
    with pytest.raises(NotImplementedError, match="EO_LLM_PROVIDER=ollama"):
        build_chat_model(settings(llm_provider="mistral"))


@pytest.mark.parametrize(
    ("err", "busy", "unreadable"),
    [
        (ollama.ResponseError("server busy", 503), True, None),
        (
            ollama.ResponseError("XML syntax error on line 1: unexpected EOF", 500),
            False,
            "XML syntax error on line 1: unexpected EOF",
        ),
        (
            ollama.ResponseError("tool 'get_scene' not found"),
            False,
            "tool 'get_scene' not found",
        ),  # inside a stream: -1
        (ollama.ResponseError('model "x" not found, try pulling it first', 404), False, None),
        (ollama.ResponseError("does not support tools", 400), False, None),
        (ConnectionError("Failed to connect to Ollama"), False, None),
    ],
)
def test_the_model_servers_errors_are_read_by_status(
    err: BaseException, busy: bool, unreadable: str | None
) -> None:
    # Ollama 0.40.1 reports a reply its parser cannot read as 500, or inside the stream; busy is 503.
    assert model_server_failed(err) is busy
    assert unreadable_reply(err) == unreadable


def test_an_unreadable_replys_message_is_kept_on_one_short_line() -> None:
    err = ollama.ResponseError("XML syntax error\n  on line 1: " + "x" * 400, 500)
    detail = unreadable_reply(err)
    assert detail is not None
    assert detail.startswith("XML syntax error on line 1: x")
    assert len(detail) == 300
