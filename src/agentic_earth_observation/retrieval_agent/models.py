"""What the retrieval agent receives and returns. Validated on both sides of the A2A boundary."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agentic_earth_observation.catalog.stac import SceneSummary
from agentic_earth_observation.domain import Product, SearchRequest

RetrievalStatus = Literal["ok", "empty", "unknown_place", "error", "not_searched"]
# The model failures of docs/evaluation.md, which fail the task. A failure outside the model's control (a
# timeout, a server that cannot be reached or is busy) is infrastructure; any other error is a bug and is
# raised (decisions of 2026-11-02).
ModelFailure = Literal["step_limit", "malformed_tool_call", "parse_error"]
PlaceName = Annotated[str, Field(min_length=1, max_length=200)]

# Both sides of A2A validate these from JSON. strict: a number must be a number and a date a "YYYY-MM-DD"
# string, never a timestamp. Text is stripped before its length is checked.
A2A_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True, str_strip_whitespace=True)
# How they travel over A2A (decision of 2026-11-03): each as one text part of this media type, with the turn's
# trace id in the request metadata under TRACE_KEY, and the result as the completed task's one artifact.
A2A_MEDIA_TYPE = "application/json"
TRACE_KEY = "trace_id"
RESULT_ARTIFACT = "retrieval_result"


class PreviousSearch(BaseModel):
    """The search a follow-up question continues: the arguments of the last successful search."""

    model_config = A2A_CONFIG
    # The gazetteer name of the place whose box was searched; None when the box came from no tool.
    place: str | None = Field(default=None, min_length=1, max_length=200)
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


class Clarification(BaseModel):
    """A question the service asked about the user's request, which this turn's text answers."""

    model_config = A2A_CONFIG
    request: str = Field(min_length=1, max_length=4000)  # the user's words so far, earlier answers included
    # The service's reply that asked the question; it may list up to 10 suggested places of 200 characters.
    asked: str = Field(min_length=1, max_length=3000)


class RetrievalRequest(BaseModel):
    """One delegated turn: the user's own words, today's date, the search it continues and the question it
    answers, if any."""

    model_config = A2A_CONFIG
    text: str = Field(min_length=1, max_length=2000)
    today: date
    previous_search: PreviousSearch | None = None
    clarifies: Clarification | None = None


class Usage(BaseModel):
    """The model's calls and tokens in one turn, and the turn's time, tool calls included."""

    model_config = ConfigDict(extra="forbid")

    model: str = ""
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    # The model server's own clock, so that a slow machine can be told from a slow model afterwards.
    model_seconds: float = 0.0  # reading the prompts and writing the replies
    load_seconds: float = (
        0.0  # loading the model into memory: a fraction of a second, once a run has loaded it
    )


class RetrievalResult(BaseModel):
    """The outcome of one delegated turn, built only from what the tools returned."""

    model_config = ConfigDict(extra="forbid", strict=True)
    status: RetrievalStatus
    scenes: list[SceneSummary] = []  # exactly the scenes of the last successful search
    search: PreviousSearch | None = None  # the arguments of that search, for a follow-up
    # Known place names close to an unknown one; the gazetteer offers at most 3.
    suggestions: list[PlaceName] = Field(default=[], max_length=10)
    failure: ModelFailure | None = None  # the model's failure: fails the task
    failure_detail: str | None = Field(default=None, max_length=300)  # the model server's words, if any
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
        if self.failure_detail is not None and self.failure is None:
            raise ValueError("failure_detail explains a failure: set failure too")
        return self
