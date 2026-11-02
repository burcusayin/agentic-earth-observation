"""Tests for the retrieval agent's instructions and request message."""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

from agentic_earth_observation.domain import Product
from agentic_earth_observation.retrieval_agent import prompts
from agentic_earth_observation.retrieval_agent.models import PreviousSearch, RetrievalRequest
from agentic_earth_observation.retrieval_agent.prompts import (
    prompt_hash,
    retrieval_system_prompt,
    retrieval_user_message,
)


def naples_week() -> PreviousSearch:
    return PreviousSearch(
        place="Naples",
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
        text="And the week after that?", today=date(2025, 12, 15), previous_search=naples_week()
    )
    assert retrieval_user_message(request) == (
        "REQUEST: And the week after that?\n"
        "PREVIOUS SEARCH: product=sentinel-1-grd, bbox=[14.1321, 40.7912, 14.3538, 40.9161], "
        "start_date=2025-09-01, end_date=2025-09-07"
    )  # radar: no cloud limit to carry over


def test_a_follow_up_carries_the_cloud_limit_and_the_limit_when_they_were_set() -> None:
    milan_march = PreviousSearch(
        place="Milan",
        product=Product.SENTINEL_2_L2A,
        bbox=[9.0409, 45.3867, 9.2781, 45.5358],
        start_date=date(2025, 3, 1),
        end_date=date(2025, 3, 31),
        max_cloud_cover=20.0,
        limit=3,
    )
    request = RetrievalRequest(text="And in April?", today=date(2025, 12, 15), previous_search=milan_march)
    assert retrieval_user_message(request).endswith(
        "PREVIOUS SEARCH: product=sentinel-2-l2a, bbox=[9.0409, 45.3867, 9.2781, 45.5358], "
        "start_date=2025-03-01, end_date=2025-03-31, max_cloud_cover=20, limit=3"
    )


def test_the_prompt_hash_covers_the_whole_module() -> None:
    # The request format (REQUEST:, PREVIOUS SEARCH:) reaches the model too, not only the system prompt.
    assert prompt_hash() == hashlib.sha256(Path(prompts.__file__).read_bytes()).hexdigest()
