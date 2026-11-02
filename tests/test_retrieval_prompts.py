"""Tests for the retrieval agent's instructions and request message."""

from __future__ import annotations

from datetime import date

from agentic_earth_observation.domain import Product
from agentic_earth_observation.retrieval_agent.models import PreviousSearch, RetrievalRequest
from agentic_earth_observation.retrieval_agent.prompts import (
    prompt_hash,
    retrieval_system_prompt,
    retrieval_user_message,
)

NAPLES_WEEK = PreviousSearch(
    product=Product.SENTINEL_1_GRD,
    bbox=[14.1321, 40.7912, 14.3538, 40.9161],
    start_date=date(2025, 9, 1),
    end_date=date(2025, 9, 7),
)


def test_the_prompt_states_today() -> None:
    assert "Today is 2025-12-15." in retrieval_system_prompt(date(2025, 12, 15))


def test_the_prompt_states_the_rules_the_evaluation_checks() -> None:
    prompt = retrieval_system_prompt(date(2025, 12, 15))
    for rule in (
        "resolve_place",
        "Never type coordinates",
        "at most 10",
        "at most 20",
        "leave it out",
        "radar",
    ):
        assert rule in prompt


def test_the_reply_step_names_search_scenes() -> None:
    # ministral-3:8b read "when the search returns" as resolve_place's reply and answered without searching.
    prompt = retrieval_system_prompt(date(2025, 12, 15))
    assert "When search_scenes returns" in prompt
    assert "the search returns" not in prompt


def test_a_first_request_is_just_the_users_words() -> None:
    request = RetrievalRequest(
        text="Radar over Naples, first week of September 2025", today=date(2025, 12, 15)
    )
    assert retrieval_user_message(request) == "REQUEST: Radar over Naples, first week of September 2025"


def test_a_follow_up_carries_the_previous_search_exactly() -> None:
    request = RetrievalRequest(
        text="And the week after that?", today=date(2025, 12, 15), previous_search=NAPLES_WEEK
    )
    message = retrieval_user_message(request)
    assert "PREVIOUS SEARCH: product=sentinel-1-grd, bbox=[14.1321, 40.7912, 14.3538, 40.9161]" in message
    assert "max_cloud_cover" not in message  # radar: no cloud limit to carry over


def test_the_prompt_hash_is_stable() -> None:
    assert prompt_hash() == prompt_hash()
    assert len(prompt_hash()) == 64
