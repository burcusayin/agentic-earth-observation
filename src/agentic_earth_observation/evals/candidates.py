"""Checks on a file of candidate tasks, before they join the test split (decision of 2026-11-12).

    uv run python -m agentic_earth_observation.evals.candidates check FILE --details OUT
    uv run python -m agentic_earth_observation.evals.candidates live FILE --details OUT
    uv run python -m agentic_earth_observation.evals.candidates questions FILE --out OUT
    uv run python -m agentic_earth_observation.evals.candidates compare FILE SECOND --details OUT

New test tasks are written by one session and labelled a second time by another, from the questions alone
(`questions`, then `compare`); whoever writes prompts must not read them. So every command prints totals
only, and writes what it found, task by task, to the file it is given. `live` asks the catalogue of
EO_STAC_API_URL, a public service by default: one search at a time, with a pause between two.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentic_earth_observation.catalog.gazetteer import Gazetteer, GazetteerError
from agentic_earth_observation.catalog.stac import CatalogError
from agentic_earth_observation.catalog_mcp.server import Catalog
from agentic_earth_observation.config import Settings
from agentic_earth_observation.evals.golden import (
    REFERENCE_DATE,
    Category,
    Expectation,
    ExpectedSearch,
    GoldenError,
    Task,
    check_tasks,
    load_golden,
    load_tasks,
)
from agentic_earth_observation.evals.groups import PERIODS, V2_PREFIX, added_in_v2, relative_period
from agentic_earth_observation.evals.run_eval import live_catalog, oracle_request

PAUSE_S = 1.0  # between two searches of the live check: the catalogue is a public service
LIVE_NOTE = re.compile(r"Live check \d{4}-\d{2}-\d{2}")  # docs/evaluation.md: the notes record what it found
# docs/evaluation.md: the month each meteorological season starts in; a season is three whole months
SEASON_STARTS = {"last-spring": 3, "last-summer": 6, "last-autumn": 9, "last-winter": 12}


class SecondLabel(BaseModel):
    """One task as the second labelling answers it: its number in the questions file, and what each of its
    turns expects."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int = Field(ge=1)
    turns: list[Expectation] = Field(min_length=1)


@dataclass(frozen=True)
class LiveResult:
    """What the catalogue answered to one expected search: how many scenes, or why it failed."""

    task_id: str
    turn: int
    scenes: int | None
    failed: str | None = None


@dataclass(frozen=True)
class Difference:
    """One way the second labelling differs from the first. `number` is the task's place in the file."""

    number: int
    text: str


def _searches(task: Task) -> list[ExpectedSearch]:
    return [turn.expect.search for turn in task.turns if turn.expect.search is not None]


def period_dates(period: str, today: date) -> tuple[date, date]:
    """The first and the last day of a relative period, resolved against `today` as docs/evaluation.md says.

    "Last month" is the calendar month before today's. "Last <season>" is the most recent meteorological
    season that is complete, that is, whose last day is before today. ValueError for any other period.
    """
    if period == "last-month":
        last = today.replace(day=1) - timedelta(days=1)
        return last.replace(day=1), last
    if period not in SEASON_STARTS:
        raise ValueError(f"{period} is not one of the periods {', '.join(PERIODS)}")
    month = SEASON_STARTS[period]
    year = today.year
    while True:
        # The season ends the day before the fourth month begins; for winter that is in the next year.
        last = date(year + (month + 3 > 12), (month + 2) % 12 + 1, 1) - timedelta(days=1)
        if last < today:
            return date(year, month, 1), last
        year -= 1


def problems(tasks: Sequence[Task], existing: Sequence[Task], gazetteer: Gazetteer) -> list[str]:
    """What keeps the candidates out of the test split; an empty list when nothing does.

    The golden set's own checks, and the rules of the decision of 2026-11-12: a new task is a test task
    whose id starts with `v2-` and is not taken; a relative-date id names its period, belongs to the
    robustness category, and expects a search over exactly that period on the reference date; no expected
    search uses a place of the dev split, on which prompts are tuned; and the notes of a task that expects
    a search say what the live check found. `existing`: the golden set.
    """
    found = check_tasks(list(tasks), gazetteer)
    taken = {task.id for task in existing}
    dev_places = {search.place for task in existing if task.split == "dev" for search in _searches(task)}
    for task in tasks:
        if task.split != "test":
            found.append(f"{task.id}: a candidate is a test task, not a {task.split} one")
        if not added_in_v2(task.id):
            found.append(f"{task.id}: the id of a new task starts with {V2_PREFIX}")
        if task.id in taken:
            found.append(f"{task.id}: the id is already a task's")
        try:
            period = relative_period(task.id)
        except ValueError as err:
            found.append(str(err))
            period = None
        if period is not None and task.category is not Category.ROBUSTNESS:
            found.append(
                f"{task.id}: a relative-date task is a robustness task, not a {task.category.value} one"
            )
        searches = _searches(task)
        if period is not None:
            first, last = period_dates(period, REFERENCE_DATE)
            if not searches:
                found.append(f"{task.id}: a relative-date task expects a search")
            found.extend(
                f"{task.id}: {period} is {first} to {last} on {REFERENCE_DATE}, "
                f"but its search expects {search.start} to {search.end}"
                for search in searches
                if (search.start, search.end) != (first, last)
            )
        found.extend(
            f"{task.id}: {place} is a place of the dev split"
            for place in sorted({search.place for search in searches} & dev_places)
        )
        if searches and not LIVE_NOTE.search(task.notes):
            found.append(
                f'{task.id}: its notes do not say what the live check found ("Live check <date>: ...")'
            )
    return found


def live_check(
    tasks: Sequence[Task], catalog: Catalog, gazetteer: Gazetteer, *, pause: Callable[[], None]
) -> list[LiveResult]:
    """Every expected search of the tasks, asked of the catalogue as the evaluation's oracle asks it
    (`oracle_request`), one at a time with `pause` between two. The tasks have passed `check_tasks`."""
    results: list[LiveResult] = []
    for task in tasks:
        for number, turn in enumerate(task.turns, start=1):
            if turn.expect.search is None:
                continue
            if results:
                pause()
            try:
                scenes = catalog.search(oracle_request(turn.expect.search, gazetteer))
            except CatalogError as err:
                results.append(LiveResult(task.id, number, None, str(err)))
            else:
                results.append(LiveResult(task.id, number, len(scenes)))
    return results


def questions(tasks: Sequence[Task]) -> list[dict[str, Any]]:
    """The users' words of every task under its place in the file, and nothing else: not the id, which
    names a relative-date task's period, and not what the task expects."""
    return [
        {"n": number, "turns": [turn.user for turn in task.turns]}
        for number, task in enumerate(tasks, start=1)
    ]


def differences(tasks: Sequence[Task], second: Sequence[SecondLabel]) -> list[Difference]:
    """How the second labelling differs from the tasks' own expectations, in the order of the tasks.

    Routes, forbidden tools and outcomes are compared as sets, required tools in their order, and the
    expected search as a whole. An entry given twice, or for no task, is a difference too.
    """
    given = Counter(label.n for label in second)
    by_number = {label.n: label for label in second}
    found: list[Difference] = []
    for number, task in enumerate(tasks, start=1):
        where = f"task {number} ({task.id})"
        label = by_number.get(number)
        if label is None:
            found.append(Difference(number, f"{where}: the second labelling has no entry for it"))
        elif given[number] > 1:
            found.append(
                Difference(number, f"entry {number} of the second labelling is given {given[number]} times")
            )
        elif len(label.turns) != len(task.turns):
            found.append(
                Difference(
                    number,
                    f"{where}: the second labelling has {len(label.turns)} turn(s) where the task has "
                    f"{len(task.turns)}",
                )
            )
        else:
            for position, (turn, other) in enumerate(zip(task.turns, label.turns, strict=True), start=1):
                found.extend(
                    Difference(number, f"{where} turn {position}, {field}")
                    for field in _fields_that_differ(turn.expect, other)
                )
    found.extend(
        Difference(number, f"entry {number} of the second labelling has no task")
        for number in sorted(given)
        if number > len(tasks)
    )
    return found


def _fields_that_differ(first: Expectation, second: Expectation) -> list[str]:
    """Each field in which two expectations of one turn differ, as "field: the first against the second"."""
    pairs: list[tuple[str, object, object]] = [
        ("route", sorted(route.value for route in first.routes), sorted(r.value for r in second.routes)),
        ("required tools", first.required_tools, second.required_tools),
        ("forbidden tools", sorted(set(first.forbidden_tools)), sorted(set(second.forbidden_tools))),
        ("search", _shown(first.search), _shown(second.search)),
        ("outcomes", _outcomes(first), _outcomes(second)),
    ]
    return [f"{field}: {mine} against {theirs}" for field, mine, theirs in pairs if mine != theirs]


def _shown(search: ExpectedSearch | None) -> str:
    # The dump is normalised by the model, so 10 and 10.0 show, and compare, the same.
    return "none" if search is None else search.model_dump_json()


def _outcomes(expectation: Expectation) -> list[str] | None:
    return None if expectation.outcomes is None else sorted(set(expectation.outcomes))


def load_second(path: Path) -> list[SecondLabel]:
    """The second labelling, one JSON object per line. GoldenError naming the line of every invalid one."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as err:
        raise GoldenError(f"cannot read {path}: {err}") from err
    labels: list[SecondLabel] = []
    invalid: list[str] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            labels.append(SecondLabel.model_validate_json(line))
        except ValidationError as err:
            invalid.append(f"{path.name}:{number}: {err}")
    if invalid:
        raise GoldenError("\n".join(invalid))
    return labels


def main(
    argv: Sequence[str] | None = None, *, catalog: Catalog | None = None, pause_s: float = PAUSE_S
) -> int:
    """Run one command. 0 when the candidates passed it, 1 when they did not or a file could not be used."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "questions":
            return _questions(args.file, args.out)
        return _checked(args, catalog, pause_s)
    except OSError as err:
        print(f"error: {err.filename} cannot be written: {err.strerror}", file=sys.stderr)
        return 1
    except (GoldenError, GazetteerError):  # the text of such an error can quote a task of the golden set
        print(
            "error: the golden set or the gazetteer cannot be loaded: "
            "`uv run python -m agentic_earth_observation.evals.golden` says why",
            file=sys.stderr,
        )
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agentic_earth_observation.evals.candidates",
        description="Check a file of candidate tasks before they join the test split. Prints totals only.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, what in (
        ("check", "the format, the gazetteer keys and the rules for new tasks"),
        ("live", "every expected search against the catalogue, one per second"),
        ("compare", "a second labelling, made from the questions alone, with the tasks' own"),
    ):
        command = commands.add_parser(name, help=what)
        command.add_argument("file", type=Path, help="the candidate tasks, one JSON object per line")
        if name == "compare":
            command.add_argument("second", type=Path, help='the second labelling: {"n": 1, "turns": [...]}')
        command.add_argument("--details", type=Path, required=True, help="where to write what was found")
    asked = commands.add_parser("questions", help="write the questions alone, for the second labelling")
    asked.add_argument("file", type=Path, help="the candidate tasks, one JSON object per line")
    asked.add_argument("--out", type=Path, required=True, help="where to write the questions")
    return parser


def _questions(file: Path, out: Path) -> int:
    try:
        tasks = load_tasks(file)
    except GoldenError:
        print("the file cannot be read as tasks; `check` writes the details")
        return 1
    _write(out, "".join(json.dumps(asked, ensure_ascii=False) + "\n" for asked in questions(tasks)))
    print(f"tasks: {len(tasks)}; the questions are in {out}")
    return 0


def _checked(args: argparse.Namespace, catalog: Catalog | None, pause_s: float) -> int:
    """`check`, `live` or `compare`: the totals on the screen, what was found in the details file."""
    details: Path = args.details
    try:
        tasks = load_tasks(args.file)
    except GoldenError as err:
        _write(details, f"{err}\n")
        print(f"the file cannot be read as tasks; the details are in {details}")
        return 1
    gazetteer = Gazetteer.load()
    if args.command == "check":
        found = problems(tasks, load_golden(gazetteer=gazetteer), gazetteer)
        print(f"tasks: {len(tasks)} ({_tally(Counter(task.category.value for task in tasks))})")
        print(f"relative dates: {_relative_dates(tasks)}")
        return _report(details, found, f"problems: {len(found)}", nothing="no problems")
    if args.command == "compare":
        try:
            second = load_second(args.second)
        except GoldenError as err:
            _write(details, f"{err}\n")
            print(f"the second labelling cannot be read; the details are in {details}")
            return 1
        differing = differences(tasks, second)
        numbers = {difference.number for difference in differing}
        differ = sum(number <= len(tasks) for number in numbers)
        totals = f"tasks compared: {len(tasks)}; agree: {len(tasks) - differ}; differ: {differ}"
        if without_a_task := len(numbers) - differ:
            totals += f"; entries without a task: {without_a_task}"
        return _report(
            details, [difference.text for difference in differing], totals, nothing="no differences"
        )
    if invalid := check_tasks(tasks, gazetteer):  # a search of an invalid task cannot be built
        return _report(details, invalid, f"problems: {len(invalid)}; nothing was searched", nothing="")
    results = live_check(
        tasks,
        catalog if catalog is not None else live_catalog(Settings()),
        gazetteer,
        pause=lambda: time.sleep(pause_s),
    )
    none = sum(result.scenes == 0 for result in results)
    failed = sum(result.failed is not None for result in results)
    _write(details, "".join(f"{_live_line(result)}\n" for result in results))
    print(
        f"searches: {len(results)}; found scenes: {len(results) - none - failed}; found none: {none}; "
        f"failed: {failed}; the details are in {details}"
    )
    return 1 if none or failed else 0


def _report(details: Path, found: Sequence[str], totals: str, *, nothing: str) -> int:
    """Write what was found to the details file and print the totals. 1 when something was found."""
    _write(details, "".join(f"{line}\n" for line in found) if found else f"{nothing}\n")
    print(f"{totals}; the details are in {details}" if found else totals)
    return 1 if found else 0


def _relative_dates(tasks: Sequence[Task]) -> str:
    try:
        periods = Counter(period for task in tasks if (period := relative_period(task.id)) is not None)
    except ValueError:
        return "cannot be counted, an id names no period"
    return f"{sum(periods.values())} ({_tally(periods)})" if periods else "0"


def _tally(counts: Counter[str]) -> str:
    return ", ".join(f"{name} {count}" for name, count in sorted(counts.items()))


def _live_line(result: LiveResult) -> str:
    found = f"failed: {result.failed}" if result.failed is not None else f"{result.scenes} scene(s)"
    return f"{result.task_id} turn {result.turn}: {found}"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
