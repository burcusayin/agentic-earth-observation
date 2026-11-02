"""The test split must not leak into what the system reads: prompts, tool descriptions, examples.

Everything under src/ except the evaluation package is code the agents run or text a model reads, and scripts/
sends prompts too (the model probe). None of it may quote a test question, name a test-split place in any
language, or use a test-split date.

Places come from two sources: the gazetteer names of every place a test search expects, and the capitalised
words inside test questions, which also catch places that have no gazetteer entry (unknown or misspelt places,
out-of-scope questions). Words that the dev split also uses, and month, day and product names, do not count.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import cache
from pathlib import Path

import pytest

from agentic_earth_observation.catalog.gazetteer import Gazetteer
from agentic_earth_observation.evals.golden import Task, load_golden

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "agentic_earth_observation"
WORD = r"[^\W\d_][\w-]*"  # a word starting with a letter: "Napoli", "Sentinel-2"
NOT_PLACES = {
    "sentinel", "sentinel-1", "sentinel-2", "sar", "radar",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}  # fmt: skip


NOT_SCANNED = {"build_gazetteer.py"}  # builds the gazetteer, which names every place by design


def system_sources() -> list[Path]:
    """Code the agents run or text a model reads: src/ except the evaluation package, and scripts/."""
    code = [p for p in SRC.rglob("*.py") if "evals" not in p.relative_to(SRC).parts]
    scripts = [p for p in (ROOT / "scripts").rglob("*.py") if p.name not in NOT_SCANNED]
    return sorted(code) + sorted(scripts)


def capitalised_words(question: str) -> set[str]:
    """Capitalised words that do not start a sentence: the place names a question may contain."""
    words: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+", question):
        words.update(word for word in re.findall(WORD, sentence)[1:] if word[0].isupper())
    return words


def _split(name: str) -> list[Task]:
    return [task for task in load_golden() if task.split == name]


def _place_names(tasks: Iterable[Task]) -> set[str]:
    gazetteer = Gazetteer.load()
    names: set[str] = set()
    for turn in (turn for task in tasks for turn in task.turns):
        if turn.expect.search is not None:
            place = gazetteer.resolve(turn.expect.search.place)
            assert place is not None
            names.update(gazetteer.names_of(place.key))
    return names


def _dates(tasks: Iterable[Task]) -> set[str]:
    searches = [turn.expect.search for task in tasks for turn in task.turns if turn.expect.search is not None]
    return {day.isoformat() for search in searches for day in (search.start, search.end)}


@cache
def _test_split_terms() -> tuple[frozenset[str], frozenset[str]]:
    """What no system file may contain: (terms matched in any case, terms matched as written)."""
    dev, test = _split("dev"), _split("test")
    questions = {turn.user for task in test for turn in task.turns}
    dev_words = {w.lower() for task in dev for turn in task.turns for w in re.findall(WORD, turn.user)}
    known = dev_words | {name.lower() for name in _place_names(dev)} | NOT_PLACES
    words = {w for q in questions for w in capitalised_words(q) if w.lower() not in known}
    return frozenset(questions | _place_names(test)), frozenset(words | (_dates(test) - _dates(dev)))


def _mentions(text: str, term: str, flags: int = 0) -> bool:
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, flags) is not None


def test_place_like_words_are_the_capitalised_ones_inside_a_sentence() -> None:
    question = "Radar over Napels for March, please. Then Zorbania in the morning?"
    assert capitalised_words(question) == {"Napels", "March", "Zorbania"}  # not "Radar" or "Then"


def test_the_test_split_gives_terms_of_every_kind() -> None:
    any_case, as_written = _test_split_terms()
    assert any_case
    assert any(re.fullmatch(r"\d{4}-\d{2}-\d{2}", term) for term in as_written)
    assert any(not term[0].isdigit() for term in as_written)


@pytest.mark.parametrize("source", system_sources(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_test_split_content_in_system_code(source: Path) -> None:
    text = source.read_text(encoding="utf-8")
    any_case, as_written = _test_split_terms()
    found = sorted(
        {term for term in any_case if _mentions(text, term, re.IGNORECASE)}
        | {term for term in as_written if _mentions(text, term)}
    )
    assert not found, f"{source.relative_to(ROOT)} mentions test-split content: {found}"
