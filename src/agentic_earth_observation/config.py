"""Settings for the agents and the evaluation, read from EO_* environment variables or `.env`.

`.env.example` lists every variable. Tests build `Settings(_env_file=None)`, so a developer's `.env` never
leaks into them. An empty variable means "use the default", and a misspelt EO_* name is refused rather than
silently ignored.
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentic_earth_observation.catalog.stac import EARTH_SEARCH_URL

ROOT = Path(__file__).resolve().parents[2]  # the repository: relative paths in settings start here
ENV_PREFIX = "EO_"
LOCAL_HOSTS = ("127.0.0.1", "localhost")


def is_local_url(url: str) -> bool:
    """Whether a URL names this machine. A client of such a URL ignores proxy settings: a request to a local
    server cannot get there through a proxy, and would hand the user's words to it."""
    return urlsplit(url).hostname in LOCAL_HOSTS


def local_port(url: str, variable: str) -> int:
    """The port of a server's URL, which must be on this machine: our servers bind to 127.0.0.1 only."""
    parts = urlsplit(url)
    try:
        port = parts.port
    except ValueError:  # a port that is not a number
        port = None
    if parts.scheme != "http" or parts.hostname not in LOCAL_HOSTS or port is None:
        raise ValueError(f"{variable} must be http://127.0.0.1:<port>, not {url!r}")
    return port


class Settings(BaseSettings):
    """Every setting with its default. Timeouts are in seconds and must nest, outermost last: a user turn is
    the decision, then the A2A call, both inside the harness's timeout."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=ROOT / ".env",  # the servers and the chat may start in different folders: one file for all
        env_ignore_empty=True,  # EO_MISTRAL_API_KEY= means "no key", not an empty key
        dotenv_filtering="match_prefix",  # .env keys without the prefix are ignored ...
        extra="forbid",  # ... and misspelt EO_ keys are refused
        hide_input_in_errors=True,  # a misspelt key's value (a secret, perhaps) never appears in an error
        frozen=True,
    )

    llm_provider: Literal["ollama", "mistral"] = "ollama"
    llm_model: str = "qwen3.5:9b"
    ollama_url: str = "http://127.0.0.1:11434"
    mistral_api_key: SecretStr | None = None

    stac_api_url: str = EARTH_SEARCH_URL
    mcp_url: str = "http://127.0.0.1:8001/mcp"
    a2a_url: str = "http://127.0.0.1:8002"
    # Gitignored. None turns the log off, from code only (tests, the probe): evaluation runs need it.
    tool_log: Path | None = Path("results/raw/tool_calls.jsonl")
    reference_date: date | None = None  # "today" for the agents; None means the real date in UTC

    stac_timeout_s: float = Field(default=20.0, gt=0)  # one HTTP request to the catalogue
    stac_max_retries: int = Field(default=3, ge=0)
    stac_deadline_s: float = Field(default=90.0, gt=0)  # one tool call, retries included: a hard limit
    # The longest the model server may stay silent. ChatOllama streams every reply, so this does not bound a
    # whole call: the decision and the retrieval turn have deadlines of their own.
    model_timeout_s: float = Field(default=120.0, gt=0)
    decision_deadline_s: float = Field(default=150.0, gt=0)  # the orchestrator's decision, request to reply
    mcp_timeout_s: float = Field(default=120.0, gt=0)
    turn_deadline_s: float = Field(default=240.0, gt=0)
    a2a_timeout_s: float = Field(default=300.0, gt=0)
    harness_timeout_s: float = Field(default=480.0, gt=0)  # one user turn: the decision, then the A2A call
    max_model_calls: int = Field(default=6, ge=4)  # resolve, search, one retry, and the final reply at least
    num_predict: int = Field(default=1024, ge=64)  # cap on generated tokens per model call

    @field_validator("tool_log", mode="before")
    @classmethod
    def _tool_log_is_a_file(cls, value: Any) -> Any:  # noqa: ANN401 - raw input, before validation
        if isinstance(value, str) and value.strip().lower() in ("none", "null", "off", "false"):
            raise ValueError(
                "the tool-call log cannot be turned off from the environment: evaluation runs need it; "
                "set EO_TOOL_LOG to a file path, or leave it unset for the default"
            )
        return value

    @field_validator("tool_log")
    @classmethod
    def _relative_to_the_repository(cls, value: Path | None) -> Path | None:
        # The MCP server and the evaluation runner may start in different folders; both must use one file.
        return value if value is None or value.is_absolute() else ROOT / value

    @model_validator(mode="after")
    def _no_misspelt_variables(self) -> Self:
        known = {f"{ENV_PREFIX}{name}".upper() for name in type(self).model_fields}
        if unknown := sorted(
            n for n in os.environ if n.upper().startswith(ENV_PREFIX) and n.upper() not in known
        ):
            raise ValueError(
                f"unknown environment variable(s) {', '.join(unknown)}; "
                "the EO_ variables are listed in .env.example"
            )
        return self

    @model_validator(mode="after")
    def _timeouts_nest(self) -> Self:
        if self.stac_timeout_s >= self.stac_deadline_s:
            raise ValueError(
                f"one STAC request ({self.stac_timeout_s:g} s) must time out before the STAC deadline "
                f"({self.stac_deadline_s:g} s)"
            )
        chain = [
            ("STAC deadline", self.stac_deadline_s),
            ("MCP", self.mcp_timeout_s),
            ("retrieval turn", self.turn_deadline_s),
            ("A2A", self.a2a_timeout_s),
            ("harness", self.harness_timeout_s),
        ]
        for (inner, inner_s), (outer, outer_s) in pairwise(chain):
            if inner_s >= outer_s:
                raise ValueError(
                    f"timeouts must nest: {inner} ({inner_s:g} s) must be below {outer} ({outer_s:g} s)"
                )
        for deadline, deadline_s in (
            ("retrieval turn", self.turn_deadline_s),
            ("decision deadline", self.decision_deadline_s),
        ):
            if self.model_timeout_s >= deadline_s:
                raise ValueError(
                    f"the model server's silence limit ({self.model_timeout_s:g} s) must be below the "
                    f"{deadline} ({deadline_s:g} s)"
                )
        if self.decision_deadline_s + self.a2a_timeout_s >= self.harness_timeout_s:
            raise ValueError(
                f"the decision ({self.decision_deadline_s:g} s) and A2A ({self.a2a_timeout_s:g} s) must end "
                f"before the harness ({self.harness_timeout_s:g} s)"
            )
        return self

    def today(self) -> date:
        """The reference date the agents treat as today."""
        return self.reference_date or datetime.now(UTC).date()
