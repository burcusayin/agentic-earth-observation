"""Scoring one turn against its golden expectation, as described in docs/evaluation.md. Pure functions only.

The evaluation harness records what the system did in a turn as a TurnTrace (route, tool calls, final status,
cited scene ids); score_turn compares it with the turn's Expectation.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict

from eo_agent_demo.catalog.gazetteer import Gazetteer
from eo_agent_demo.evals.golden import Expectation, ExpectedSearch, Outcome, Route

MIN_BBOX_IOU = 0.95
SEARCH_ARGUMENTS = frozenset({"product", "bbox", "start_date", "end_date", "max_cloud_cover", "limit"})

BBox = tuple[float, float, float, float]  # west, south, east, north


def bbox_iou(a: BBox, b: BBox) -> float:
    """Intersection over union of two lon/lat boxes (0.0 when they do not overlap)."""
    for box in (a, b):
        if len(box) != 4 or box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError(f"invalid bbox: {box!r}")
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


class ToolCall(BaseModel):
    """One tool call as the harness recorded it."""

    model_config = ConfigDict(frozen=True)

    name: str
    arguments: dict[str, Any] = {}
    ok: bool = True  # False when the tool returned an error
    returned_ids: list[str] = []


class TurnTrace(BaseModel):
    """What the system did in one turn."""

    model_config = ConfigDict(frozen=True)

    route: Route
    calls: list[ToolCall] = []
    status: Outcome
    cited_ids: list[str] = []  # scene ids that appear in the final answer


class ArgumentScore(BaseModel):
    """Whether each argument of a search_scenes call matched the expectation."""

    model_config = ConfigDict(frozen=True)

    product: bool
    bbox: bool
    dates: bool
    cloud: bool
    limit: bool
    no_extra: bool

    @property
    def all(self) -> bool:
        return all(self.model_dump().values())


MISSING_SEARCH = ArgumentScore(
    product=False, bbox=False, dates=False, cloud=False, limit=False, no_extra=False
)


class TurnScore(BaseModel):
    """The score of one turn. `arguments` is None when the turn expects no search."""

    model_config = ConfigDict(frozen=True)

    route: bool
    tools: bool
    arguments: ArgumentScore | None
    first_arguments: ArgumentScore | None
    grounded: bool
    outcome: bool

    @property
    def passed(self) -> bool:
        arguments_ok = self.arguments is None or self.arguments.all
        return self.route and self.tools and arguments_ok and self.grounded and self.outcome


def tools_in_order(called: Sequence[str], required: Sequence[str]) -> bool:
    """True when `required` appears in `called` in this order; other calls may come in between."""
    remaining = iter(called)
    return all(name in remaining for name in required)


def score_arguments(
    expected: ExpectedSearch, arguments: dict[str, Any], gazetteer: Gazetteer
) -> ArgumentScore:
    """Compare one search_scenes call's arguments with the expected search."""
    place = gazetteer.resolve(expected.place)
    if place is None:
        raise ValueError(f"'{expected.place}' is not in the gazetteer; validate the golden set first")
    box = place.bbox
    try:
        got = [float(v) for v in arguments["bbox"]]
        bbox_ok = (
            len(got) == 4
            and bbox_iou((got[0], got[1], got[2], got[3]), (box.west, box.south, box.east, box.north))
            >= MIN_BBOX_IOU
        )
    except (KeyError, TypeError, ValueError):
        bbox_ok = False
    try:
        dates_ok = (
            date.fromisoformat(str(arguments["start_date"])) == expected.start
            and date.fromisoformat(str(arguments["end_date"])) == expected.end
        )
    except (KeyError, ValueError):
        dates_ok = False
    cloud = arguments.get("max_cloud_cover")
    if expected.cloud is None:
        cloud_ok = cloud is None
    else:
        cloud_ok = isinstance(cloud, int | float) and expected.cloud[0] <= cloud <= expected.cloud[1]
    return ArgumentScore(
        product=arguments.get("product") == expected.product.value,
        bbox=bbox_ok,
        dates=dates_ok,
        cloud=cloud_ok,
        limit=expected.limit is None or arguments.get("limit") == expected.limit,
        no_extra=set(arguments) <= SEARCH_ARGUMENTS,
    )


def score_turn(
    expect: Expectation, trace: TurnTrace, gazetteer: Gazetteer, known_ids: frozenset[str] = frozenset()
) -> TurnScore:
    """Score one turn. `known_ids` are scene ids that tools returned in earlier turns of the conversation."""
    called = [call.name for call in trace.calls]
    searches = [call for call in trace.calls if call.name == "search_scenes"]
    successful = [call for call in searches if call.ok]

    arguments = first_arguments = None
    if expect.search is not None:
        arguments = (
            score_arguments(expect.search, successful[-1].arguments, gazetteer)
            if successful
            else MISSING_SEARCH
        )
        first_arguments = (
            score_arguments(expect.search, searches[0].arguments, gazetteer) if searches else MISSING_SEARCH
        )

    returned = known_ids | {scene_id for call in trace.calls for scene_id in call.returned_ids}
    return TurnScore(
        route=trace.route in expect.routes,
        tools=tools_in_order(called, expect.required_tools) and not set(called) & set(expect.forbidden_tools),
        arguments=arguments,
        first_arguments=first_arguments,
        grounded=set(trace.cited_ids) <= returned,
        outcome=expect.outcomes is None or trace.status in expect.outcomes,
    )


def retrieval_metrics(returned: list[str], oracle: list[str], limit: int) -> tuple[float, float]:
    """Precision of the returned scene ids, and recall up to `limit`, against a direct catalogue query."""
    hits = len(set(returned) & set(oracle))
    precision = hits / len(set(returned)) if returned else float(not oracle)
    wanted = min(limit, len(set(oracle)))
    recall = hits / wanted if wanted else 1.0
    return precision, recall
