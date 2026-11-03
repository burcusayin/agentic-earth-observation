"""Specification for compose_reply: what the user reads, from templates and the retrieval agent's result.

The tests check what each reply must do, not its wording, which is free: the found scenes are listed and cited
in order; nothing else contains anything like a scene id; each reason and status has its own reply.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

import pytest

from agentic_earth_observation.catalog.stac import SceneSummary
from agentic_earth_observation.domain import Product
from agentic_earth_observation.orchestrator.models import DecisionRoute, Reason
from agentic_earth_observation.orchestrator.replies import Reply, compose_reply
from agentic_earth_observation.retrieval_agent.models import PreviousSearch, RetrievalResult
from tests.fakes import SCENE

# Anything that looks like a Sentinel scene id, so that a reply that should cite nothing cannot slip one in.
SCENE_ID_LIKE = re.compile(r"S[12][A-D]_\w+")
OTHER_SCENE = SceneSummary(
    id="S2B_T32TMR_20250331T101753_L2A",
    product=Product.SENTINEL_2_L2A,
    acquired=datetime(2025, 3, 31, 10, 28, 18, tzinfo=UTC),
    cloud_cover=0.5,
    platform="sentinel-2b",
)
MILAN_MARCH: dict[str, Any] = {
    "place": "Milan",
    "product": "sentinel-2-l2a",
    "bbox": [9.0409, 45.3867, 9.2781, 45.5358],
    "start_date": "2025-03-01",
    "end_date": "2025-03-31",
    "max_cloud_cover": 20,
}


def search(**changes: object) -> PreviousSearch:
    return PreviousSearch.model_validate_json(json.dumps(MILAN_MARCH | changes))


def delegated(result: RetrievalResult) -> Reply:
    return compose_reply("delegate", "search", result, None, None)


FOUND = RetrievalResult(status="ok", scenes=[OTHER_SCENE, SCENE], search=search())
NOTHING = RetrievalResult(status="empty", search=search())
UNKNOWN = RetrievalResult(status="unknown_place", suggestions=["Milan", "Naples"])
UNKNOWN_NO_IDEA = RetrievalResult(status="unknown_place")
SEARCH_ERROR = RetrievalResult(status="error")
NOT_SEARCHED = RetrievalResult(status="not_searched")

# Every reply that must cite nothing, by what led to it.
CITING_NOTHING: dict[str, tuple[DecisionRoute | None, Reason | None, RetrievalResult | None, Any, Any]] = {
    "infrastructure": (None, None, None, None, "ConnectError: All connection attempts failed"),
    "model failure": (None, None, None, "parse_error", None),
    "decline": ("decline", "out_of_scope", None, None, None),
    "clarify place": ("clarify", "missing_place", None, None, None),
    "clarify period": ("clarify", "missing_period", None, None, None),
    "nothing found": ("delegate", "search", NOTHING, None, None),
    "unknown place": ("delegate", "search", UNKNOWN, None, None),
    "unknown place, no suggestions": ("delegate", "search", UNKNOWN_NO_IDEA, None, None),
    "search error": ("delegate", "search", SEARCH_ERROR, None, None),
    "not searched": ("delegate", "search", NOT_SEARCHED, None, None),
}


class TestFoundScenes:
    def test_the_scenes_are_listed_and_cited_in_order(self) -> None:
        reply = delegated(FOUND)
        assert reply.cited_ids == [OTHER_SCENE.id, SCENE.id]
        assert reply.text.index(OTHER_SCENE.id) < reply.text.index(SCENE.id)
        assert SCENE_ID_LIKE.findall(reply.text) == reply.cited_ids  # and no other id

    def test_the_reply_names_the_search(self) -> None:
        text = delegated(FOUND).text
        assert all(part in text for part in ("Milan", "sentinel-2-l2a", "2025-03-01", "2025-03-31"))

    def test_a_box_that_no_tool_named_is_the_area_searched(self) -> None:
        unnamed = RetrievalResult(status="ok", scenes=[SCENE], search=search(place=None))
        text = delegated(unnamed).text
        assert "the area searched" in text
        assert "None" not in text


class TestCitingNothing:
    @pytest.mark.parametrize("case", CITING_NOTHING)
    def test_no_other_reply_cites_or_shows_a_scene_id(self, case: str) -> None:
        reply = compose_reply(*CITING_NOTHING[case])
        assert reply.text.strip()
        assert reply.cited_ids == []
        assert SCENE_ID_LIKE.findall(reply.text) == []

    def test_each_case_has_its_own_reply(self) -> None:
        texts = {case: compose_reply(*args).text for case, args in CITING_NOTHING.items()}
        assert len(set(texts.values())) == len(texts), texts

    def test_nothing_found_names_the_search(self) -> None:
        text = delegated(NOTHING).text
        assert all(part in text for part in ("Milan", "2025-03-01", "2025-03-31"))

    def test_an_unknown_place_offers_each_suggestion(self) -> None:
        text = delegated(UNKNOWN).text
        assert "Milan" in text
        assert "Naples" in text


class TestPrecedence:
    """The first rule that applies decides the reply."""

    def test_infrastructure_comes_before_everything(self) -> None:
        infra = "ReadTimeout"
        assert compose_reply("delegate", "search", FOUND, None, infra) == compose_reply(
            None, None, None, None, infra
        )

    def test_a_model_failure_comes_before_found_scenes(self) -> None:
        # The task fails anyway; the reply apologises and cites nothing, even with scenes in hand.
        reply = compose_reply("delegate", "search", FOUND, "step_limit", None)
        assert reply == compose_reply(None, None, None, "parse_error", None)
        assert reply.cited_ids == []
