"""Specification for the retrieval agent's request and result: the A2A boundary, in both directions.

Payloads cross A2A as JSON and are validated with `model_validate_json`, so these tests send JSON where the
other side would. Each model refuses what cannot be right; nothing is repaired or guessed.
"""

from __future__ import annotations

import json
import math
from datetime import date
from typing import Any

import pytest
from pydantic import ValidationError

from agentic_earth_observation.domain import Product
from agentic_earth_observation.retrieval_agent.models import PreviousSearch, RetrievalRequest, RetrievalResult
from tests.fakes import SCENE

NAPLES_WEEK: dict[str, Any] = {
    "place": "Naples",
    "product": "sentinel-1-grd",
    "bbox": [14.1321, 40.7912, 14.3538, 40.9161],
    "start_date": "2025-09-01",
    "end_date": "2025-09-07",
}
MILAN_MARCH: dict[str, Any] = {
    "place": "Milan",
    "product": "sentinel-2-l2a",
    "bbox": [9.0409, 45.3867, 9.2781, 45.5358],
    "start_date": "2025-03-01",
    "end_date": "2025-03-31",
    "max_cloud_cover": 20,
    "limit": 3,
}


def previous(base: dict[str, Any] = NAPLES_WEEK, /, **changes: object) -> PreviousSearch:
    """A PreviousSearch as the other side of A2A would send it: JSON."""
    return PreviousSearch.model_validate_json(json.dumps({**base, **changes}))


class TestPreviousSearch:
    """The search a follow-up continues. The model is told to reuse it exactly, so it must be a valid one."""

    def test_a_search_the_catalogue_accepts_is_accepted(self) -> None:
        radar = previous()
        assert (radar.place, radar.product, radar.bbox) == (
            "Naples",
            Product.SENTINEL_1_GRD,
            NAPLES_WEEK["bbox"],
        )
        assert (radar.start_date, radar.end_date) == (date(2025, 9, 1), date(2025, 9, 7))
        assert (radar.max_cloud_cover, radar.limit) == (None, None)
        optical = previous(MILAN_MARCH)
        assert (optical.max_cloud_cover, optical.limit) == (20, 3)

    def test_whole_numbers_are_fine_in_the_box(self) -> None:
        assert previous(bbox=[14, 40, 15, 41]).bbox == [14.0, 40.0, 15.0, 41.0]

    def test_the_place_name_is_stripped(self) -> None:
        assert previous(place="  Naples\n").place == "Naples"

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            # The domain's own rules, reused rather than repeated: the same message as the MCP server gives.
            ({"bbox": [14.3538, 40.7912, 14.1321, 40.9161]}, "west must be less than east"),
            ({"start_date": "2025-09-07", "end_date": "2025-09-01"}, "must not be after end date"),
            ({"max_cloud_cover": 10}, "not applicable for Sentinel-1"),
            ({"start_date": "2014-03-01", "end_date": "2014-03-07"}, "before the mission start date"),
            ({"start_date": "2025-01-01", "end_date": "2026-03-01"}, "maximum allowed range"),
        ],
    )
    def test_a_search_the_catalogue_would_refuse_is_refused_with_its_reason(
        self, changes: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValidationError, match=message):
            previous(**changes)

    @pytest.mark.parametrize(
        "changes",
        [
            {"limit": 0},
            {"limit": 51},
            {"bbox": [14.1321, 40.7912, 14.3538]},
            {"bbox": ["14.1321", 40.7912, 14.3538, 40.9161]},  # a number sent as text
            {"limit": True},  # a boolean is not a number
            {"start_date": 1756684800},  # a Unix timestamp is not a date
            {"start_date": "1 September 2025"},
            {"place": ""},
            {"place": "   "},
            {"cloud_cover": 10},  # a misspelt field is refused, not dropped
        ],
    )
    def test_malformed_values_are_refused(self, changes: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            previous(**changes)

    def test_a_box_that_is_not_finite_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="finite"):
            PreviousSearch(
                place="Naples",
                product=Product.SENTINEL_1_GRD,
                bbox=[math.nan, 40.7912, 14.3538, 40.9161],
                start_date=date(2025, 9, 1),
                end_date=date(2025, 9, 7),
            )

    def test_it_survives_the_trip_through_json(self) -> None:
        search = previous(MILAN_MARCH)
        assert PreviousSearch.model_validate_json(search.model_dump_json()) == search


class TestRetrievalRequest:
    def test_the_text_is_stripped(self) -> None:
        request = RetrievalRequest.model_validate_json(
            '{"text": "  Radar over Naples\\n", "today": "2025-12-15"}'
        )
        assert request.text == "Radar over Naples"
        assert request.today == date(2025, 12, 15)
        assert request.previous_search is None

    @pytest.mark.parametrize(
        "payload",
        [
            {"text": "", "today": "2025-12-15"},
            {"text": "  \n ", "today": "2025-12-15"},  # blank once stripped
            {"text": "x" * 2001, "today": "2025-12-15"},
            {"text": "Radar over Naples", "today": 1765756800},  # a Unix timestamp is not a date
            {"text": "Radar over Naples", "today": "15/12/2025"},
            {"text": "Radar over Naples", "today": "2025-12-15", "trace": "turn-1"},  # unknown field
            {
                "text": "Radar over Naples",
                "today": "2025-12-15",
                "previous_search": {**NAPLES_WEEK, "limit": 0},
            },
        ],
    )
    def test_a_malformed_request_is_refused(self, payload: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            RetrievalRequest.model_validate_json(json.dumps(payload))

    def test_the_longest_text_is_accepted(self) -> None:
        RetrievalRequest(text="x" * 2000, today=date(2025, 12, 15))

    def test_a_follow_up_survives_the_trip_through_json(self) -> None:
        request = RetrievalRequest(
            text="And the week after that?", today=date(2025, 12, 15), previous_search=previous()
        )
        assert RetrievalRequest.model_validate_json(request.model_dump_json()) == request


class TestRetrievalResult:
    """A result says what happened, and only what can have happened together.

    ok and empty come from a successful search, so they carry it; every other status means no search
    succeeded, so there are no scenes and no search. Suggestions belong to an unknown place only.
    """

    @pytest.mark.parametrize(
        ("status", "scenes", "searched", "suggestions"),
        [
            ("ok", [SCENE], True, []),
            ("empty", [], True, []),
            ("unknown_place", [], False, ["Napoli", "Milano"]),
            ("unknown_place", [], False, []),  # nothing close enough to suggest
            ("error", [], False, []),
            ("not_searched", [], False, []),
        ],
    )
    def test_consistent_results_are_accepted(
        self, status: str, scenes: list[Any], searched: bool, suggestions: list[str]
    ) -> None:
        search = previous(MILAN_MARCH) if searched else None
        result = RetrievalResult.model_validate(
            {"status": status, "scenes": scenes, "search": search, "suggestions": suggestions}
        )
        assert RetrievalResult.model_validate_json(result.model_dump_json()) == result

    @pytest.mark.parametrize(
        ("status", "scenes", "searched", "suggestions"),
        [
            ("ok", [], True, []),  # ok means scenes were found
            ("ok", [SCENE], False, []),  # ... by a search, which a follow-up needs
            ("empty", [SCENE], True, []),
            ("empty", [], False, []),
            ("unknown_place", [SCENE], False, []),
            ("error", [], True, []),  # a successful search would make it ok or empty
            ("not_searched", [SCENE], False, []),
            ("ok", [SCENE], True, ["Napoli"]),  # suggestions belong to an unknown place
            ("lost", [], False, []),  # not a status
        ],
    )
    def test_inconsistent_results_are_refused(
        self, status: str, scenes: list[Any], searched: bool, suggestions: list[str]
    ) -> None:
        search = previous(MILAN_MARCH) if searched else None
        with pytest.raises(ValidationError):
            RetrievalResult.model_validate(
                {"status": status, "scenes": scenes, "search": search, "suggestions": suggestions}
            )

    def test_a_failure_is_either_the_models_or_the_infrastructures(self) -> None:
        RetrievalResult(status="not_searched", failure="step_limit")
        RetrievalResult(status="not_searched", infra_error="the catalogue did not answer")
        with pytest.raises(ValidationError):
            RetrievalResult(
                status="not_searched", failure="step_limit", infra_error="the catalogue did not answer"
            )

    def test_a_partial_result_keeps_what_the_tools_returned(self) -> None:
        # The model ran out of steps after a successful search: the scenes stay, and the failure is recorded.
        result = RetrievalResult(
            status="ok", scenes=[SCENE], search=previous(MILAN_MARCH), failure="step_limit"
        )
        assert result.scenes == [SCENE]

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalResult.model_validate_json(
                '{"status": "not_searched", "answer": "Here are your scenes"}'
            )
