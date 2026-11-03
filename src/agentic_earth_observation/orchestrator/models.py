"""What the orchestrator decides for a turn, and what the turn produces.

The route and the scored status use the evaluation's own types (`evals.golden.Route` and `Outcome`), so they
cannot drift from what the scoring expects.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_earth_observation.evals.golden import Outcome, Route
from agentic_earth_observation.retrieval_agent.models import (
    ModelFailure,
    RetrievalResult,
    RetrievalStatus,
    Usage,
)

DecisionRoute = Literal["delegate", "decline", "clarify"]  # the values of evals.golden.Route
Reason = Literal["search", "out_of_scope", "missing_place", "missing_period"]
# The reasons each route may give; a decision that pairs them otherwise contradicts itself.
REASONS: dict[DecisionRoute, tuple[Reason, ...]] = {
    "delegate": ("search",),
    "decline": ("out_of_scope",),
    "clarify": ("missing_place", "missing_period"),
}
# The scored status of a delegated turn, from the retrieval agent's status (decision of 2026-11-02).
SCORED_STATUS: dict[RetrievalStatus, Outcome] = {
    "ok": "ok",
    "empty": "empty",
    "error": "error",
    "unknown_place": "clarified",
    "not_searched": "declined",
}


class Decision(BaseModel):
    """How to handle the request: delegate a search, decline it, or ask the user for what is missing."""

    model_config = ConfigDict(extra="forbid")

    route: DecisionRoute = Field(description="delegate, decline or clarify")
    reason: Reason = Field(
        description=(
            "search (with delegate), out_of_scope (with decline), "
            "missing_place or missing_period (with clarify)"
        )
    )

    @model_validator(mode="after")
    def _route_and_reason_agree(self) -> Self:
        if self.reason not in REASONS[self.route]:
            raise ValueError(f"the route {self.route!r} cannot have the reason {self.reason!r}")
        return self


def scored_status(route: DecisionRoute | None, retrieval: RetrievalResult | None) -> Outcome:
    """The turn's status as the evaluation scores it.

    A delegated turn takes the retrieval agent's status, mapped; otherwise the decision's route says it. A
    turn without a decision (the model failed, or the infrastructure did) falls back to declined, as it
    searched nothing.
    """
    if retrieval is not None:
        return SCORED_STATUS[retrieval.status]
    return "clarified" if route == "clarify" else "declined"


class TurnOutput(BaseModel):
    """What one turn did: for the user, the answer; for the evaluation, everything else."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: str
    route: Route  # as observed: delegate exactly when the retrieval agent was called
    status: Outcome  # as scored
    answer: str
    cited_ids: list[str] = []  # the scene ids the answer cites
    failure: ModelFailure | None = None  # the model's failure, in the decision or in the retrieval
    failure_detail: str | None = None
    infra_error: str | None = None  # a failure outside the model's control: retried and excluded
    decision_usage: Usage = Usage()  # the orchestrator's own model call
    retrieval: RetrievalResult | None = None  # the retrieval agent's result, with its own usage

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if (self.route is Route.DELEGATE) != (self.retrieval is not None):
            raise ValueError("a delegated turn has the retrieval agent's result, and only a delegated turn")
        if self.failure is not None and self.infra_error is not None:
            raise ValueError("a failure is either the model's or the infrastructure's, not both")
        if self.failure_detail is not None and self.failure is None:
            raise ValueError("failure_detail explains a failure: set failure too")
        returned = {scene.id for scene in self.retrieval.scenes} if self.retrieval is not None else set()
        if not set(self.cited_ids) <= returned:
            raise ValueError("an answer may cite only scene ids that the retrieval agent's tools returned")
        return self
