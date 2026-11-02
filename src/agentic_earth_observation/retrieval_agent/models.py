"""What the retrieval agent receives and returns. Validated on both sides of the A2A boundary."""

from __future__ import annotations

from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agentic_earth_observation.catalog.stac import SceneSummary
from agentic_earth_observation.domain import Product, SearchRequest

RetrievalStatus = Literal["ok", "empty", "unknown_place", "error", "not_searched"]
ModelFailure = Literal["step_limit", "malformed_tool_call", "parse_error", "turn_timeout"]

# Both sides of A2A validate these from JSON. strict: a number must be a number and a date a "YYYY-MM-DD"
# string, never a timestamp. Text is stripped before its length is checked.
A2A_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True, str_strip_whitespace=True)


class PreviousSearch(BaseModel):
    """The search a follow-up question continues: the arguments of the last successful search."""

    model_config = A2A_CONFIG
    place: str = Field(min_length=1, max_length=200)  # the gazetteer name of the place searched
    product: Product
    bbox: list[float] = Field(min_length=4, max_length=4)
    start_date: date
    end_date: date
    max_cloud_cover: float | None = None
    limit: int | None = None

    @model_validator(mode="after")
    def _a_search_the_catalogue_accepts(self) -> Self:
        arguments: dict[str, object] = {
            "product": self.product,
            "bbox": dict(zip(("west", "south", "east", "north"), self.bbox, strict=True)),
            "dates": {"start": self.start_date, "end": self.end_date},
            "max_cloud_cover": self.max_cloud_cover,
        }
        if self.limit is not None:
            arguments["limit"] = self.limit
        try:
            SearchRequest.model_validate(arguments)
        except ValidationError as err:
            reasons = (str(e["msg"]).removeprefix("Value error, ") for e in err.errors())
            raise ValueError("; ".join(reasons)) from None
        return self


class RetrievalRequest(BaseModel):
    """One delegated turn: the user's own words, today's date, and the search it continues, if any."""

    model_config = A2A_CONFIG
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

    model_config = ConfigDict(extra="forbid", strict=True)
    status: RetrievalStatus
    scenes: list[SceneSummary] = []  # exactly the scenes of the last successful search
    search: PreviousSearch | None = None  # the arguments of that search, for a follow-up
    suggestions: list[str] = []  # known place names close to an unknown one
    failure: ModelFailure | None = None  # the model's failure: fails the task
    infra_error: str | None = None  # a failure outside the model's control: retried and excluded
    usage: Usage = Usage()

    @model_validator(mode="after")
    def _only_what_can_happen_together(self) -> Self:
        if (self.status in ("ok", "empty")) != (self.search is not None):
            raise ValueError("ok and empty come from a successful search and carry it; other statuses do not")
        if (self.status == "ok") != bool(self.scenes):
            raise ValueError("scenes come with ok only, and ok always has scenes")
        if self.suggestions and self.status != "unknown_place":
            raise ValueError("suggestions come with unknown_place only")
        if self.failure is not None and self.infra_error is not None:
            raise ValueError("a failure is either the model's or the infrastructure's, not both")
        return self
