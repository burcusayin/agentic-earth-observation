"""Settings for the agents and the evaluation, read from EO_* environment variables or `.env`.

`.env.example` lists every variable. Tests build `Settings(_env_file=None)`, so a developer's `.env` never
leaks into them.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from itertools import pairwise
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentic_earth_observation.catalog.stac import EARTH_SEARCH_URL

_STAC_BACKOFF_S = 0.5 + 1.0 + 2.0  # urllib3 backoff for 3 retries with backoff_factor 0.5


class Settings(BaseSettings):
    """Every setting with its default. Timeouts are in seconds and must nest, outermost last."""

    model_config = SettingsConfigDict(env_prefix="EO_", env_file=".env", extra="ignore", frozen=True)

    llm_provider: Literal["ollama", "mistral"] = "ollama"
    llm_model: str = "qwen3.5:9b"
    ollama_url: str = "http://127.0.0.1:11434"
    mistral_api_key: SecretStr | None = None

    stac_api_url: str = EARTH_SEARCH_URL
    mcp_url: str = "http://127.0.0.1:8001/mcp"
    a2a_url: str = "http://127.0.0.1:8002"
    tool_log: Path | None = Path("results/raw/tool_calls.jsonl")  # gitignored; None turns the log off
    reference_date: date | None = None  # "today" for the agents; None means the real date in UTC

    stac_timeout_s: float = Field(default=20.0, gt=0)
    stac_max_retries: int = Field(default=3, ge=0)
    model_timeout_s: float = Field(default=120.0, gt=0)
    mcp_timeout_s: float = Field(default=120.0, gt=0)
    turn_deadline_s: float = Field(default=240.0, gt=0)
    a2a_timeout_s: float = Field(default=300.0, gt=0)
    harness_timeout_s: float = Field(default=360.0, gt=0)
    max_model_calls: int = Field(default=6, ge=2)  # resolve, search, one retry, and the final reply at least
    num_predict: int = Field(default=1024, ge=64)  # cap on generated tokens per model call

    @model_validator(mode="after")
    def _timeouts_nest(self) -> Self:
        stac_worst = self.stac_timeout_s * (self.stac_max_retries + 1) + _STAC_BACKOFF_S
        chain = [
            ("STAC worst case", stac_worst),
            ("MCP", self.mcp_timeout_s),
            ("retrieval turn", self.turn_deadline_s),
            ("A2A", self.a2a_timeout_s),
            ("harness", self.harness_timeout_s),
        ]
        for (inner, inner_s), (outer, outer_s) in pairwise(chain):
            if inner_s >= outer_s:
                raise ValueError(
                    f"timeouts must nest: {inner} ({inner_s:.0f} s) must be below {outer} ({outer_s:.0f} s)"
                )
        if self.model_timeout_s >= self.turn_deadline_s:
            raise ValueError("one model call must time out before the retrieval turn deadline")
        return self

    def today(self) -> date:
        """The reference date the agents treat as today."""
        return self.reference_date or datetime.now(UTC).date()
