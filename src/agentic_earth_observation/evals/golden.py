"""The golden set: task format, loading, and validation against the gazetteer and the domain rules.

See docs/evaluation.md for the labelling rules. To validate the committed golden set and print a summary:

    uv run python -m agentic_earth_observation.evals.golden
"""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agentic_earth_observation.catalog.gazetteer import Gazetteer
from agentic_earth_observation.domain import Product, SearchRequest

GOLDEN_DIR = Path(__file__).resolve().parents[3] / "data" / "golden"
SPLITS = ("dev", "test")
REFERENCE_DATE = date(2025, 12, 15)  # "today" for every task; relative dates are resolved against it

ToolName = Literal["list_products", "resolve_place", "search_scenes", "get_scene"]
Outcome = Literal["ok", "empty", "error", "declined", "clarified"]


class GoldenError(Exception):
    """The golden set is missing or contains invalid tasks. The message lists every problem found."""


class Category(StrEnum):
    SINGLE_SEARCH = "single_search"
    FOLLOW_UP = "follow_up"
    ROBUSTNESS = "robustness"
    OUT_OF_SCOPE = "out_of_scope"
    CLARIFY = "clarify"
    EDGE = "edge"


class Route(StrEnum):
    DELEGATE = "delegate"
    DECLINE = "decline"
    CLARIFY = "clarify"


class ExpectedSearch(BaseModel):
    """The expected search_scenes call. `place` is a gazetteer key; `cloud` is the accepted range or None."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    product: Product
    place: str
    start: date
    end: date
    cloud: tuple[float, float] | None = None
    limit: int | None = Field(default=None, ge=1, le=50)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.cloud is not None:
            low, high = self.cloud
            if not 0 <= low <= high <= 100:
                raise ValueError(f"cloud must be a range [low, high] within 0-100, got {list(self.cloud)}")
            if self.product is Product.SENTINEL_1_GRD:
                raise ValueError("a Sentinel-1 search never has a cloud limit; set cloud to null")
        return self


class Expectation(BaseModel):
    """What should happen in one turn."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    route: Route | list[Route] = Field(description="The expected route, or a list of acceptable routes")
    required_tools: list[ToolName] = []
    forbidden_tools: list[ToolName] = []
    search: ExpectedSearch | None = None
    outcomes: list[Outcome] | None = None

    @property
    def routes(self) -> frozenset[Route]:
        """The acceptable routes, whether one or several were given."""
        return frozenset([self.route] if isinstance(self.route, Route) else self.route)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.routes:
            raise ValueError("route must name at least one route")
        if overlap := set(self.required_tools) & set(self.forbidden_tools):
            raise ValueError(f"tools both required and forbidden: {sorted(overlap)}")
        if Route.DELEGATE not in self.routes:
            if self.search is not None:
                raise ValueError(f"a {sorted(self.routes)} turn cannot expect a search")
            if "search_scenes" not in self.forbidden_tools:
                raise ValueError(f"a {sorted(self.routes)} turn must forbid search_scenes")
        if self.search is not None and "search_scenes" not in self.required_tools:
            raise ValueError("an expected search needs search_scenes in required_tools")
        return self


class Turn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    user: str = Field(min_length=1, max_length=1000)
    expect: Expectation


class Task(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9-]+$", max_length=60)
    split: Literal["dev", "test"]
    category: Category
    turns: list[Turn] = Field(min_length=1, max_length=3)
    notes: str = ""

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.category is Category.FOLLOW_UP and len(self.turns) < 2:
            raise ValueError("a follow_up task needs at least two turns")
        if self.category is not Category.FOLLOW_UP and len(self.turns) != 1:
            raise ValueError(f"a {self.category} task has exactly one turn")
        return self


def load_tasks(path: Path) -> list[Task]:
    """All tasks in one JSONL file. Raises GoldenError naming the line of every invalid task."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as err:
        raise GoldenError(f"cannot read {path}: {err}") from err
    tasks: list[Task] = []
    problems: list[str] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            tasks.append(Task.model_validate_json(line))
        except ValidationError as err:
            problems.append(f"{path.name}:{number}: {err}")
    if problems:
        raise GoldenError("\n".join(problems))
    return tasks


def check_tasks(tasks: list[Task], gazetteer: Gazetteer) -> list[str]:
    """Problems that need the gazetteer or several tasks to see. An empty list means the tasks are valid."""
    problems: list[str] = []
    for task_id, count in Counter(task.id for task in tasks).items():
        if count > 1:
            problems.append(f"{task_id}: the id is used {count} times")
    for task in tasks:
        for number, turn in enumerate(task.turns, start=1):
            search = turn.expect.search
            if search is None:
                continue
            where = f"{task.id} turn {number}"
            place = gazetteer.resolve(search.place)
            if place is None or place.key != search.place:
                problems.append(f"{where}: '{search.place}' is not a gazetteer key")
                continue
            if search.end > REFERENCE_DATE:
                problems.append(f"{where}: ends after the reference date {REFERENCE_DATE}")
            try:
                SearchRequest.model_validate(
                    {
                        "product": search.product,
                        "bbox": place.bbox,
                        "dates": {"start": search.start, "end": search.end},
                        "max_cloud_cover": None if search.cloud is None else search.cloud[1],
                        "limit": search.limit or 10,
                    }
                )
            except ValidationError as err:
                problems.append(f"{where}: the expected search is not a valid request: {err}")
    return problems


def load_golden(golden_dir: Path = GOLDEN_DIR, gazetteer: Gazetteer | None = None) -> list[Task]:
    """Both splits, validated. Raises GoldenError listing every problem."""
    tasks: list[Task] = []
    for split in SPLITS:
        loaded = load_tasks(golden_dir / f"{split}.jsonl")
        wrong = [t.id for t in loaded if t.split != split]
        if wrong:
            raise GoldenError(f"{split}.jsonl contains tasks of another split: {wrong}")
        tasks.extend(loaded)
    problems = check_tasks(tasks, gazetteer or Gazetteer.load())
    if problems:
        raise GoldenError("\n".join(problems))
    return tasks


def golden_hash(golden_dir: Path = GOLDEN_DIR) -> str:
    """SHA-256 over both split files, recorded with every run so results name the exact golden set."""
    digest = hashlib.sha256()
    for split in SPLITS:
        digest.update((golden_dir / f"{split}.jsonl").read_bytes())
    return digest.hexdigest()


def main() -> None:
    tasks = load_golden()
    for split in SPLITS:
        counts = Counter(t.category.value for t in tasks if t.split == split)
        print(
            f"{split}: {sum(counts.values())} tasks  "
            + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        )
    print(f"golden hash: {golden_hash()[:16]}")


if __name__ == "__main__":
    main()
