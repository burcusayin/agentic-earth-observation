"""The reply the user reads, built from templates and the retrieval agent's result: never from a model's text.

Only the scenes of a successful search are listed, so an answer cites only scene ids that a tool returned
(grounding by construction, decision of 2026-11-02). Nothing a model wrote reaches the user: a decline or a
clarification is a template chosen by the decision's reason.
"""

from __future__ import annotations

from typing import NamedTuple

from agentic_earth_observation.orchestrator.models import DecisionRoute, Reason
from agentic_earth_observation.retrieval_agent.models import ModelFailure, RetrievalResult


class Reply(NamedTuple):
    text: str
    cited_ids: list[str]  # the scene ids listed in `text`, in the same order


def compose_reply(
    route: DecisionRoute | None,
    reason: Reason | None,
    retrieval: RetrievalResult | None,
    failure: ModelFailure | None,
    infra_error: str | None,
) -> Reply:
    """The reply to one turn. The first rule that applies decides it:

    1. `infra_error` is set: say the service is unavailable and to try again.
    2. `failure` is set: apologise that the request could not be handled.
    3. `route` is "decline": say what the service can do (scene metadata for a place and a period).
    4. `route` is "clarify": ask for the place ("missing_place") or for the period ("missing_period").
    5. `route` is "delegate": reply from `retrieval.status`:
       - "ok": list the scenes' ids, and name the search (place, product, dates);
       - "empty": say that no scenes were found, and name the search;
       - "unknown_place": say the place is unknown and offer the suggestions, or just ask if there are none;
       - "error": say the search could not be done;
       - "not_searched": say what the service can do.

    Only the "ok" reply cites scene ids. The search's place may be None (the box came from no tool).
    """
    if infra_error is not None:
        return Reply("Sorry, the scene search is not available right now. Please try again in a moment.", [])
    if failure is not None:
        return Reply("Sorry, I could not handle that request. Please rephrase it and try again.", [])
    if route == "decline":
        return Reply(
            "I can only find Sentinel-1 and Sentinel-2 scenes (their metadata, not the images) for a place "
            "and a period.",
            [],
        )
    if route == "clarify":
        if reason == "missing_place":
            return Reply("Which place should I search? Please name it.", [])
        return Reply("For which dates should I search? Please give a period.", [])
    if retrieval is None:
        raise ValueError("a delegated turn has the retrieval agent's result")
    search = retrieval.search
    if retrieval.status == "ok" and search is not None:
        ids = [scene.id for scene in retrieval.scenes]
        where = search.place or "the area searched"
        return Reply(
            f"Found {len(ids)} {search.product.value} scene(s) of {where} from {search.start_date} to "
            f"{search.end_date}: {', '.join(ids)}.",
            ids,
        )
    if retrieval.status == "empty" and search is not None:
        where = search.place or "the area searched"
        return Reply(
            f"No {search.product.value} scenes of {where} from {search.start_date} to {search.end_date} "
            "match the search.",
            [],
        )
    if retrieval.status == "unknown_place":
        if retrieval.suggestions:
            return Reply(f"I don't know that place. Did you mean {' or '.join(retrieval.suggestions)}?", [])
        return Reply("I don't know that place. Which place do you mean?", [])
    if retrieval.status == "error":
        return Reply("The search could not be done. Please check the request, or try again later.", [])
    return Reply("I can only search for Sentinel-1 and Sentinel-2 scenes of a place and a period.", [])
