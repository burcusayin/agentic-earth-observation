"""What the retrieval agent receives and returns. Validated on both sides of the A2A boundary."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agentic_earth_observation.catalog.stac import SceneSummary
from agentic_earth_observation.domain import Product

RetrievalStatus = Literal["ok", "empty", "unknown_place", "error", "not_searched"]
ModelFailure = Literal["step_limit", "malformed_tool_call", "parse_error", "turn_timeout"]


class PreviousSearch(BaseModel):
    """The search a follow-up question continues: the arguments of the last successful search."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    product: Product
    bbox: list[float] = Field(min_length=4, max_length=4)
    start_date: date
    end_date: date
    max_cloud_cover: float | None = None
    limit: int | None = None


class RetrievalRequest(BaseModel):
    """One delegated turn: the user's own words, today's date, and the search it continues, if any."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1, max_length=2000)
    today: date
    previous_search: PreviousSearch | None = None


class Usage(BaseModel):
    """Tokens and time spent by the model in one turn."""

    model_config = ConfigDict(extra="forbid")

    model: str = ""
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


class RetrievalResult(BaseModel):
    """The outcome of one delegated turn, built only from what the tools returned."""

    model_config = ConfigDict(extra="forbid")

    status: RetrievalStatus
    scenes: list[SceneSummary] = []  # exactly the scenes of the last successful search
    search: PreviousSearch | None = None  # the arguments of that search, for a follow-up
    suggestions: list[str] = []  # known place names close to an unknown one
    failure: ModelFailure | None = None  # the model's failure: fails the task
    infra_error: str | None = None  # a failure outside the model's control: retried and excluded
    usage: Usage = Usage()
