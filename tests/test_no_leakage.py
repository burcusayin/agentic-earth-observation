"""The test split must not leak into what the system reads: prompts, tool descriptions, examples.

Everything under src/ except the evaluation package is code the agents run or text a model reads. None of it
may name a test-split place (in any language), or quote a test question.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agentic_earth_observation.catalog.gazetteer import Gazetteer
from agentic_earth_observation.evals.golden import load_golden

SRC = Path(__file__).resolve().parents[1] / "src" / "agentic_earth_observation"


def system_sources() -> list[Path]:
    return [p for p in SRC.rglob("*.py") if "evals" not in p.relative_to(SRC).parts]


def _test_split_terms() -> set[str]:
    """Names and aliases of every test-split place, and every test question."""
    gazetteer = Gazetteer.load()
    terms: set[str] = set()
    for task in load_golden():
        if task.split != "test":
            continue
        for turn in task.turns:
            terms.add(turn.user)
            if turn.expect.search is not None:
                place = gazetteer.resolve(turn.expect.search.place)
                assert place is not None
                terms.update(gazetteer.names_of(place.key))
    return terms


@pytest.mark.parametrize("source", system_sources(), ids=lambda p: str(p.relative_to(SRC)))
def test_no_test_split_place_or_question_in_system_code(source: Path) -> None:
    text = source.read_text(encoding="utf-8")
    found = sorted(
        term
        for term in _test_split_terms()
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE)
    )
    assert not found, f"{source.relative_to(SRC)} mentions test-split content: {found}"
