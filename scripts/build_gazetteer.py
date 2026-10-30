"""Build data/gazetteer.json from OpenStreetMap through Nominatim. Run rarely, and only after asking.

    uv run python scripts/build_gazetteer.py

Follows the Nominatim usage policy (https://operations.osmfoundation.org/policies/nominatim/): at most one
request per second, an identifying User-Agent, and a handful of requests in total. The output is
OpenStreetMap-derived data: (c) OpenStreetMap contributors, licensed under the ODbL 1.0.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from agentic_earth_observation.catalog.stac import USER_AGENT
from agentic_earth_observation.domain import BBox

NOMINATIM = "https://nominatim.openstreetmap.org/search"
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "gazetteer.json"
NAME_KEYS = ("name:en", "name", "name:it", "name:de", "name:fr")


@dataclass(frozen=True)
class Spec:
    """One place to look up: the search text, and the kinds of result accepted for it."""

    key: str
    query: str
    accept: frozenset[str]  # acceptable Nominatim "addresstype" or "type" values
    names_query: str | None = None  # search whose names to use, when the box comes from a different feature


CITY = frozenset({"city", "town", "municipality"})
PLACES = [
    Spec("rome", "Roma, Lazio, Italia", CITY),
    Spec("milan", "Milano, Lombardia, Italia", CITY),
    Spec("florence", "Firenze, Toscana, Italia", CITY),
    Spec("naples", "Napoli, Campania, Italia", CITY),
    Spec("trento", "Trento, Trentino-Alto Adige, Italia", CITY),
    Spec("paris", "Paris, Île-de-France, France", CITY),
    Spec("munich", "München, Bayern, Deutschland", CITY),
    Spec("venice lagoon", "Laguna di Venezia", frozenset({"water", "bay", "lagoon", "wetland"})),
    Spec("lake garda", "Lago di Garda", frozenset({"water", "lake"})),
    Spec(
        "mount etna",
        "Parco dell'Etna",
        frozenset({"protected_area", "nature_reserve", "national_park", "park"}),
        names_query="Etna, Sicilia",
    ),
]


def search(client: httpx.Client, query: str) -> list[dict[str, Any]]:
    response = client.get(
        NOMINATIM,
        params={"q": query, "format": "jsonv2", "limit": 5, "namedetails": 1, "accept-language": "en"},
    )
    response.raise_for_status()
    time.sleep(1.1)  # usage policy: at most one request per second
    results: list[dict[str, Any]] = response.json()
    return results


def pick(results: list[dict[str, Any]], accept: frozenset[str]) -> dict[str, Any] | None:
    for result in results:
        if result.get("addresstype") in accept or result.get("type") in accept:
            return result
    return None


def names_of(result: dict[str, Any]) -> list[str]:
    details: dict[str, str] = result.get("namedetails") or {}
    found = [details[k] for k in NAME_KEYS if details.get(k)] + [result.get("name") or ""]
    unique: list[str] = []
    for name in found:
        if name and name.casefold() not in {u.casefold() for u in unique}:
            unique.append(name)
    return unique


def bbox_of(result: dict[str, Any]) -> BBox:
    south, north, west, east = (float(v) for v in result["boundingbox"])  # Nominatim's order
    return BBox(west=round(west, 4), south=round(south, 4), east=round(east, 4), north=round(north, 4))


def main() -> int:
    places: dict[str, dict[str, Any]] = {}
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30) as client:
        for spec in PLACES:
            chosen = pick(search(client, spec.query), spec.accept)
            if chosen is None:
                print(f"ERROR {spec.key}: no result of kind {sorted(spec.accept)} for {spec.query!r}")
                return 1
            name_source = chosen
            if spec.names_query:
                name_source = (search(client, spec.names_query) or [chosen])[0]
            names = names_of(name_source)
            bbox = bbox_of(chosen)
            places[spec.key] = {
                "name": names[0],
                "aliases": names[1:],
                "bbox": bbox.as_list(),
                "osm": f"{chosen['osm_type']}/{chosen['osm_id']}",
                "kind": chosen.get("addresstype") or chosen.get("type"),
            }
            area = (bbox.east - bbox.west) * (bbox.north - bbox.south)
            print(f"{spec.key:14} {places[spec.key]['osm']:18} area {area:6.3f} deg2  {bbox.as_list()}")
            print(f"{'':14} {chosen.get('display_name', '')[:90]}")
            print(f"{'':14} names: {names}")

    gazetteer = {
        "source": "OpenStreetMap via Nominatim (https://nominatim.openstreetmap.org)",
        "licence": "(c) OpenStreetMap contributors, Open Database License (ODbL) 1.0",
        "retrieved": datetime.now(UTC).date().isoformat(),
        "bbox_order": "west, south, east, north (degrees)",
        "places": places,
    }
    OUTPUT.write_text(json.dumps(gazetteer, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(places)} places to {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
