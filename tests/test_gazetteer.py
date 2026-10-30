"""Tests for the gazetteer: the committed data file, name matching, suggestions and file validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_earth_observation.catalog.gazetteer import Gazetteer, GazetteerError, normalise


@pytest.fixture(scope="module")
def gazetteer() -> Gazetteer:
    return Gazetteer.load()


def test_the_committed_file_loads_with_all_ten_places(gazetteer: Gazetteer) -> None:
    assert len(gazetteer.known_places()) == 10


@pytest.mark.parametrize(
    ("asked", "key"),
    [
        ("Rome", "rome"),
        ("roma", "rome"),
        ("  ROM ", "rome"),
        ("München", "munich"),
        ("munchen", "munich"),
        ("Monaco di Baviera", "munich"),
        ("Gardasee", "lake garda"),
        ("lake garda", "lake garda"),
        ("Ätna", "mount etna"),
        ("Mount Etna", "mount etna"),
        ("Laguna di Venezia", "venice lagoon"),
    ],
)
def test_resolves_names_aliases_and_spellings(gazetteer: Gazetteer, asked: str, key: str) -> None:
    place = gazetteer.resolve(asked)
    assert place is not None
    assert place.key == key


def test_unknown_place_returns_none_rather_than_a_guess(gazetteer: Gazetteer) -> None:
    assert gazetteer.resolve("Atlantis") is None


def test_suggests_close_names_for_a_typo(gazetteer: Gazetteer) -> None:
    assert gazetteer.suggest("Trentto")[0] == "Trento"


def test_an_absurdly_long_name_is_refused_without_work(gazetteer: Gazetteer) -> None:
    assert gazetteer.resolve("x" * 10_000) is None
    assert gazetteer.suggest("x" * 10_000) == []


def test_normalise_ignores_case_accents_and_spaces() -> None:
    assert normalise("  Lago   di GÀRDA ") == "lago di garda"


def test_a_malformed_file_gives_a_clear_error(tmp_path: Path) -> None:
    bad = tmp_path / "gazetteer.json"
    bad.write_text('{"source": "x", "places": {}}', encoding="utf-8")
    with pytest.raises(GazetteerError, match="cannot load the gazetteer"):
        Gazetteer.load(bad)


def test_an_invalid_box_in_the_file_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "gazetteer.json"
    bad.write_text(
        '{"source": "x", "licence": "x", "retrieved": "2026-10-26", "bbox_order": "x", "places": {"nowhere": '
        '{"name": "Nowhere", "bbox": [13.0, 41.0, 12.0, 42.0], "osm": "relation/1", "kind": "city"}}}',
        encoding="utf-8",
    )
    with pytest.raises(GazetteerError, match="west must be less than east"):
        Gazetteer.load(bad)


def entry(name: str, aliases: list[str]) -> str:
    return (
        f'{{"name": "{name}", "aliases": {json.dumps(aliases)}, "bbox": [12.0, 41.0, 13.0, 42.0], '
        '"osm": "relation/1", "kind": "city"}'
    )


def write_gazetteer(path: Path, places: dict[str, str]) -> Path:
    body = ", ".join(f'"{key}": {value}' for key, value in places.items())
    path.write_text(
        '{"source": "x", "licence": "x", "retrieved": "2026-10-26", "bbox_order": "x", "places": {'
        + body
        + "}}",
        encoding="utf-8",
    )
    return path


def test_a_name_shared_by_two_places_is_refused(tmp_path: Path) -> None:
    bad = write_gazetteer(
        tmp_path / "g.json", {"a": entry("Springfield", []), "b": entry("Shelbyville", ["springfield"])}
    )
    with pytest.raises(GazetteerError, match="belongs to both"):
        Gazetteer.load(bad)


def test_an_empty_alias_is_refused(tmp_path: Path) -> None:
    bad = write_gazetteer(tmp_path / "g.json", {"a": entry("Somewhere", ["  "])})
    with pytest.raises(GazetteerError, match="cannot load"):
        Gazetteer.load(bad)


def test_empty_input_and_a_zero_limit_give_no_suggestions(gazetteer: Gazetteer) -> None:
    assert gazetteer.resolve("") is None
    assert gazetteer.suggest("") == []
    assert gazetteer.suggest("   ") == []
    assert gazetteer.suggest("Trentto", limit=0) == []
