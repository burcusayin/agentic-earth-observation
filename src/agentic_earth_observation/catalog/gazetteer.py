"""Place names to bounding boxes, from the committed gazetteer (OpenStreetMap-derived data, ODbL).

Lookups ignore case, accents and extra spaces. Unknown names return None, never a guess; `suggest` offers the
closest known names so the caller can ask the user which one they meant.
"""

from __future__ import annotations

import difflib
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agentic_earth_observation.domain import BBox

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "gazetteer.json"
MAX_NAME_LENGTH = 200


class GazetteerError(Exception):
    """The gazetteer file is missing or malformed."""


def normalise(text: str) -> str:
    """Case-, accent- and whitespace-insensitive form of a name: '  MÜNCHEN ' -> 'munchen'."""
    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(without_accents.casefold().split())


class Place(BaseModel):
    """A known place, as tools return it."""

    model_config = ConfigDict(frozen=True)

    key: str
    name: str
    bbox: BBox


class _Entry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    aliases: list[str] = []
    bbox: BBox
    osm: str
    kind: str

    @field_validator("bbox", mode="before")
    @classmethod
    def _bbox_from_list(cls, value: Any) -> Any:  # noqa: ANN401 - raw JSON before validation
        if isinstance(value, list) and len(value) == 4:
            return dict(zip(("west", "south", "east", "north"), value, strict=True))
        return value


class _File(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    licence: str
    retrieved: date
    bbox_order: str
    places: dict[str, _Entry] = Field(min_length=1)


class Gazetteer:
    """An index from every name and alias of every place to that place."""

    def __init__(self, entries: dict[str, _Entry]) -> None:
        self._places: dict[str, Place] = {}
        self._index: dict[str, str] = {}
        for key, entry in entries.items():
            self._places[key] = Place(key=key, name=entry.name, bbox=entry.bbox)
            for alias in (key, entry.name, *entry.aliases):
                folded = normalise(alias)
                if self._index.get(folded, key) != key:
                    raise GazetteerError(
                        f"the name {alias!r} belongs to both {self._index[folded]!r} and {key!r}"
                    )
                self._index[folded] = key

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> Gazetteer:
        try:
            data = _File.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as err:
            raise GazetteerError(f"cannot load the gazetteer from {path}: {err}") from err
        return cls(data.places)

    def resolve(self, name: str) -> Place | None:
        """The place with this name or alias, or None if it is not in the gazetteer."""
        if len(name) > MAX_NAME_LENGTH:
            return None
        key = self._index.get(normalise(name))
        return self._places[key] if key else None

    def suggest(self, name: str, limit: int = 3) -> list[str]:
        """Up to `limit` known place names that look like `name`, best first (for typos)."""
        if len(name) > MAX_NAME_LENGTH:
            return []
        matches = difflib.get_close_matches(normalise(name), list(self._index), n=limit * 3, cutoff=0.6)
        names: list[str] = []
        for match in matches:
            place_name = self._places[self._index[match]].name
            if place_name not in names:
                names.append(place_name)
        return names[:limit]

    def known_places(self) -> list[str]:
        """The display names of all places, sorted."""
        return sorted(place.name for place in self._places.values())
