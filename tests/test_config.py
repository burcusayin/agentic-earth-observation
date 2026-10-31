"""Tests for the settings and the model factory (no network: building a model makes no call)."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from agentic_earth_observation.config import Settings
from agentic_earth_observation.llm import OLLAMA_CONTEXT, SEED, build_chat_model


def settings(**overrides: Any) -> Settings:  # noqa: ANN401 - forwarded to Settings unchanged
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]  # a pydantic-settings init option


def test_defaults_are_local_and_safe() -> None:
    s = settings()
    assert s.llm_provider == "ollama"
    assert s.mcp_url.startswith("http://127.0.0.1")
    assert s.a2a_url.startswith("http://127.0.0.1")
    assert s.mistral_api_key is None


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


def test_the_default_timeouts_nest() -> None:
    s = settings()
    assert s.stac_timeout_s * (s.stac_max_retries + 1) < s.mcp_timeout_s < s.turn_deadline_s
    assert s.turn_deadline_s < s.a2a_timeout_s < s.harness_timeout_s


@pytest.mark.parametrize(
    "overrides",
    [{"a2a_timeout_s": 200.0}, {"mcp_timeout_s": 60.0}, {"model_timeout_s": 300.0}],
)
def test_timeouts_that_do_not_nest_are_refused(overrides: dict[str, float]) -> None:
    with pytest.raises(ValidationError, match="must"):
        settings(**overrides)


def test_the_ollama_model_gets_the_agreed_options() -> None:
    model = build_chat_model(settings(llm_model="ministral-3:8b"))
    assert isinstance(model, ChatOllama)
    assert model.model == "ministral-3:8b"
    assert model.num_ctx == OLLAMA_CONTEXT
    assert model.temperature == 0
    assert model.seed == SEED
    assert model.reasoning is False


def test_mistral_is_not_wired_up_yet() -> None:
    with pytest.raises(NotImplementedError, match="EO_LLM_PROVIDER=ollama"):
        build_chat_model(settings(llm_provider="mistral"))
