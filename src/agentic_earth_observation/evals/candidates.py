"""Checks on a file of candidate tasks, before they join the test split (decision of 2026-11-12).

    uv run python -m agentic_earth_observation.evals.candidates check FILE --details OUT
    uv run python -m agentic_earth_observation.evals.candidates live FILE --details OUT
    uv run python -m agentic_earth_observation.evals.candidates questions FILE --out OUT
    uv run python -m agentic_earth_observation.evals.candidates compare FILE SECOND --details OUT

New test tasks are written by one session and labelled a second time by another, from the questions
(`questions`, then `compare`); whoever writes prompts must not read them. So every command prints totals
only, on both streams, and writes what it found, task by task, to the file it is given.

`live` asks the catalogue of EO_STAC_API_URL, a public service by default. It sends one request to open the
catalogue and then one per expected search, with a pause between two searches and no retries, and it stops
at the first search that fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
import warnings
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
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
# docs/evaluation.md: the notes record how many scenes each expected search returned when it was checked.
# Every new search must have found some: the runner allows one task without, and v1 has it.
LIVE_NOTE = re.compile(r"Live check (\d{4}-\d{2}-\d{2}): the expected search returns [1-9]\d* scenes?\.")
LIVE_NOTE_FORM = '"Live check <date>: the expected search returns N scenes."'
# docs/evaluation.md: the month each meteorological season starts in; a season is three whole months
SEASON_STARTS = {"last-spring": 3, "last-summer": 6, "last-autumn": 9, "last-winter": 12}
RELATIVE_WORDS = re.compile(r"\blast\s+(month|spring|summer|autumn|winter)\b", re.IGNORECASE)
LONGEST_NAME = 4  # words: how long a place name in a question may be for the check that it is named
KEY_LENGTH = 12
# What may stand around a word in a question: punctuation, brackets and quotation marks of several kinds.
EDGES = ".,;:!?()[]{}\"'\u00ab\u00bb\u201c\u201d\u2018\u2019\u201e"


class SecondLabel(BaseModel):
    """One task as the second labelling answers it: the key of its questions (`question_key`), copied from
    the questions file, and what each of its turns expects."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(pattern=rf"^[0-9a-f]{{{KEY_LENGTH}}}$")
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
    """One way the second labelling differs from the first. `number` is the task's place in the file; None
    for an entry of the second labelling that answers no task."""

    number: int | None
    text: str


def _searches(task: Task) -> list[ExpectedSearch]:
    return [turn.expect.search for turn in task.turns if turn.expect.search is not None]


def _same_words(text: str) -> str:
    return " ".join(text.casefold().split())


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


def places_named(text: str, gazetteer: Gazetteer) -> set[str]:
    """The keys of the places that the text names in a way the gazetteer resolves: any run of up to
    LONGEST_NAME words, without the punctuation around it and without a possessive ending."""
    words = text.split()
    keys: set[str] = set()
    for start in range(len(words)):
        for end in range(start + 1, min(start + LONGEST_NAME, len(words)) + 1):
            name = " ".join(words[start:end]).strip(EDGES)
            for written in (name, name.removesuffix("'s").removesuffix("\u2019s")):
                place = gazetteer.resolve(written)
                if place is not None:
                    keys.add(place.key)
    return keys


def problems(tasks: Sequence[Task], existing: Sequence[Task], gazetteer: Gazetteer) -> list[str]:
    """What keeps the candidates out of the test split; an empty list when nothing does.

    The golden set's own checks, and the rules of the decision of 2026-11-12. `existing`: the golden set.

    - A new task is a test task whose id starts with `v2-` and is not taken.
    - None of its questions is asked by an existing task or by another candidate.
    - A relative-date id names its period; the task belongs to the robustness category, names the period
      in the rules' words, and expects a search over exactly that period on the reference date. No other
      task names such a period.
    - No expected search uses a place of the dev split, on which prompts are tuned, and the questions up
      to an expected search name its place in a way the gazetteer resolves.
    - The notes of a task that expects a search say what the live check found, for every search.
    """
    found = check_tasks(list(tasks), gazetteer)
    taken = {task.id for task in existing}
    asked = Counter(_same_words(turn.user) for task in [*existing, *tasks] for turn in task.turns)
    dev_places = {search.place for task in existing if task.split == "dev" for search in _searches(task)}
    for task in tasks:
        if task.split != "test":
            found.append(f"{task.id}: a candidate is a test task, not a {task.split} one")
        if not added_in_v2(task.id):
            found.append(f"{task.id}: the id of a new task starts with {V2_PREFIX}")
        if task.id in taken:
            found.append(f"{task.id}: the id is already a task's")
        if any(asked[_same_words(turn.user)] > 1 for turn in task.turns):
            found.append(f"{task.id}: one of its questions is asked a second time, by it or by another task")
        found.extend(_relative_date_problems(task))
        found.extend(_place_problems(task, dev_places, gazetteer))
        if (note := _note_problem(task)) is not None:
            found.append(note)
    return found


def _relative_date_problems(task: Task) -> list[str]:
    try:
        period = relative_period(task.id)
    except ValueError as err:
        return [str(err)]
    if period is None:
        if any(RELATIVE_WORDS.search(turn.user) for turn in task.turns):
            return [
                f"{task.id}: a question names a relative period, so its id starts with v2-relative-<period>"
            ]
        return []
    found: list[str] = []
    if task.category is not Category.ROBUSTNESS:
        found.append(f"{task.id}: a relative-date task is a robustness task, not a {task.category.value} one")
    words = period.replace("-", " ")
    named = {
        " ".join(match.group(0).lower().split())
        for turn in task.turns
        for match in RELATIVE_WORDS.finditer(turn.user)
    }
    if named != {words}:
        found.append(
            f'{task.id}: its questions name "{words}", the rules\' words for its period, and no other period'
        )
    first, last = period_dates(period, REFERENCE_DATE)
    searches = _searches(task)
    if not searches:
        found.append(f"{task.id}: a relative-date task expects a search")
    found.extend(
        f"{task.id}: {period} is {first} to {last} on {REFERENCE_DATE}, "
        f"but its search expects {search.start} to {search.end}"
        for search in searches
        if (search.start, search.end) != (first, last)
    )
    return found


def _place_problems(task: Task, dev_places: set[str], gazetteer: Gazetteer) -> list[str]:
    found: list[str] = []
    named: set[str] = set()
    for number, turn in enumerate(task.turns, start=1):
        named |= places_named(turn.user, gazetteer)
        search = turn.expect.search
        if search is None:
            continue
        if search.place in dev_places:
            found.append(f"{task.id} turn {number}: {search.place} is a place of the dev split")
        if search.place not in named:
            found.append(
                f"{task.id} turn {number}: no question up to it names {search.place} "
                "in a way the gazetteer resolves"
            )
    return found


def _note_problem(task: Task) -> str | None:
    """What is wrong with the notes of a task that expects a search: they give the live check's date and a
    count of at least one scene for every expected search, in the form the golden set uses."""
    turns = [number for number, turn in enumerate(task.turns, start=1) if turn.expect.search is not None]
    if not turns:
        return None
    sentence = LIVE_NOTE.search(task.notes)
    if sentence is None:
        return (
            f"{task.id}: its notes do not say what the live check found, "
            f"as {LIVE_NOTE_FORM} with N at least 1"
        )
    try:
        date.fromisoformat(sentence.group(1))
    except ValueError:
        return f"{task.id}: the date of the live check in its notes is not a date"
    rest = task.notes[sentence.end() :]
    for number in turns[1:]:
        if re.search(rf"\bTurn {number}: [1-9]\d*\.", rest) is None:
            return (
                f"{task.id}: its notes give no count for turn {number}, "
                f'as "Turn {number}: N." with N at least 1'
            )
    return None


def live_check(
    tasks: Sequence[Task], catalog: Catalog, gazetteer: Gazetteer, *, pause: Callable[[], None]
) -> Iterator[LiveResult]:
    """Every expected search of the tasks, asked of the catalogue as the evaluation's oracle asks it
    (`oracle_request`), one at a time with `pause` between two. The tasks have passed `check_tasks`.

    It stops after the first search that fails: a search of a valid task fails when the catalogue does not
    answer, and asking a public service again and again then helps nobody.
    """
    asked = False
    for task in tasks:
        for number, turn in enumerate(task.turns, start=1):
            if turn.expect.search is None:
                continue
            if asked:
                pause()
            asked = True
            try:
                scenes = catalog.search(oracle_request(turn.expect.search, gazetteer))
            except CatalogError as err:
                yield LiveResult(task.id, number, None, str(err))
                return
            yield LiveResult(task.id, number, len(scenes))


def question_key(task: Task) -> str:
    """A name for a task made of its users' words alone. The second labelling copies it, so that its labels
    are compared with the task they were made for, and are of no task once a question has changed."""
    words = json.dumps([turn.user for turn in task.turns], ensure_ascii=True)
    return hashlib.sha256(words.encode("ascii")).hexdigest()[:KEY_LENGTH]


def questions(tasks: Sequence[Task]) -> list[dict[str, Any]]:
    """The users' words of every task under its key, and nothing else: not the id, which names a
    relative-date task's period, not what the task expects, and not its place in the file, which follows
    the categories. They are in the order of their keys."""
    return sorted(
        ({"key": question_key(task), "turns": [turn.user for turn in task.turns]} for task in tasks),
        key=lambda entry: str(entry["key"]),
    )


def differences(tasks: Sequence[Task], second: Sequence[SecondLabel]) -> list[Difference]:
    """How the second labelling differs from the tasks' own expectations, in the order of the tasks.

    A label belongs to the task with its key. Routes, forbidden tools and outcomes are compared as sets,
    required tools in their order, and the expected search as a whole. A task without a label, with
    several, or with the words of another task is a difference too, and so is a label for no task.
    """
    keys = [question_key(task) for task in tasks]
    shared = Counter(keys)
    given = Counter(label.key for label in second)
    by_key = {label.key: label for label in second}
    found: list[Difference] = []
    for number, (task, key) in enumerate(zip(tasks, keys, strict=True), start=1):
        where = f"task {number} ({task.id})"
        label = by_key.get(key)
        if shared[key] > 1:
            found.append(
                Difference(number, f"{where}: another task asks the same words, so its label is not its own")
            )
        elif label is None:
            found.append(Difference(number, f"{where}: the second labelling has no entry for its questions"))
        elif given[key] > 1:
            found.append(Difference(number, f"{where}: the second labelling has {given[key]} entries for it"))
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
        Difference(None, f"entry {key} of the second labelling answers the questions of no task")
        for key in sorted(set(given) - set(keys))
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
    except (OSError, UnicodeDecodeError) as err:
        raise GoldenError(f"cannot read {path} as UTF-8 text: {type(err).__name__}") from err
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


def _candidates(path: Path) -> list[Task]:
    try:
        return load_tasks(path)
    except UnicodeDecodeError as err:
        raise GoldenError(f"{path.name} is not UTF-8 text") from err


def main(
    argv: Sequence[str] | None = None, *, catalog: Catalog | None = None, pause_s: float = PAUSE_S
) -> int:
    """Run one command. 0 when the candidates passed it, 1 when they did not or a file could not be used."""
    args = _parser().parse_args(argv)
    written: Path = args.out if args.command == "questions" else args.details
    read = [args.file, args.second] if args.command == "compare" else [args.file]
    if any(written.resolve() == path.resolve() for path in read):
        print(f"error: {written} is a file the command reads; name another file to write to", file=sys.stderr)
        return 1
    try:
        with _nothing_logged():
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


@contextmanager
def _nothing_logged() -> Iterator[None]:
    """No log record and no warning reaches the screen while a command runs: the catalogue client logs what
    a catalogue answered, and such an answer can quote the search it refused."""
    before = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        logging.disable(before)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agentic_earth_observation.evals.candidates",
        description="Check a file of candidate tasks before they join the test split. Prints totals only.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, what in (
        ("check", "the format, the gazetteer keys and the rules for new tasks"),
        ("live", "every expected search against the catalogue, one per second, until one fails"),
        ("compare", "a second labelling, made from the questions, with the tasks' own"),
    ):
        command = commands.add_parser(name, help=what)
        command.add_argument("file", type=Path, help="the candidate tasks, one JSON object per line")
        if name == "compare":
            command.add_argument(
                "second", type=Path, help='the second labelling: {"key": "...", "turns": [...]}'
            )
        command.add_argument("--details", type=Path, required=True, help="where to write what was found")
    asked = commands.add_parser("questions", help="write the questions alone, for the second labelling")
    asked.add_argument("file", type=Path, help="the candidate tasks, one JSON object per line")
    asked.add_argument("--out", type=Path, required=True, help="where to write the questions")
    return parser


def _questions(file: Path, out: Path) -> int:
    try:
        tasks = _candidates(file)
    except GoldenError:
        print("the file cannot be read as tasks; `check` writes the details")
        return 1
    if not tasks:
        print("the file has no tasks")
        return 1
    _write(out, "".join(f"{_one_line(asked)}\n" for asked in questions(tasks)))
    print(f"tasks: {len(tasks)}; the questions are in {out}")
    return 0


def _one_line(entry: dict[str, Any]) -> str:
    """The entry as JSON on one line, for a reader that splits lines as Python does: the three line
    separators that JSON allows inside a string are written as escapes."""
    text = json.dumps(entry, ensure_ascii=False)
    for separator in ("\u0085", "\u2028", "\u2029"):
        text = text.replace(separator, f"\\u{ord(separator):04x}")
    return text


def _checked(args: argparse.Namespace, catalog: Catalog | None, pause_s: float) -> int:
    """`check`, `live` or `compare`: the totals on the screen, what was found in the details file."""
    details: Path = args.details
    try:
        tasks = _candidates(args.file)
    except GoldenError as err:
        _write(details, f"{err}\n")
        print(f"the file cannot be read as tasks; the details are in {details}")
        return 1
    if not tasks:
        _write(details, "the file has no tasks\n")
        print("the file has no tasks")
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
        differ = len({difference.number for difference in differing if difference.number is not None})
        totals = f"tasks compared: {len(tasks)}; agree: {len(tasks) - differ}; differ: {differ}"
        if without_a_task := sum(difference.number is None for difference in differing):
            totals += f"; entries without a task: {without_a_task}"
        return _report(
            details, [difference.text for difference in differing], totals, nothing="no differences"
        )
    if invalid := check_tasks(tasks, gazetteer):  # a search of an invalid task cannot be built
        return _report(details, invalid, f"problems: {len(invalid)}; nothing was searched", nothing="")
    return _live(tasks, details, catalog, gazetteer, pause_s)


def _live(
    tasks: Sequence[Task], details: Path, catalog: Catalog | None, gazetteer: Gazetteer, pause_s: float
) -> int:
    """The live check. Each answer is written as it comes, so that a run that is stopped keeps what it
    learnt, and the details file is opened before anything is asked."""
    if catalog is None:  # without retries: a search that fails is not asked again, here or by the client
        catalog = live_catalog(Settings().model_copy(update={"stac_max_retries": 0}))
    expected = sum(len(_searches(task)) for task in tasks)
    results: list[LiveResult] = []
    details.parent.mkdir(parents=True, exist_ok=True)
    with details.open("w", encoding="utf-8") as out:
        for result in live_check(tasks, catalog, gazetteer, pause=lambda: time.sleep(pause_s)):
            out.write(f"{_live_line(result)}\n")
            out.flush()
            results.append(result)
    none = sum(result.scenes == 0 for result in results)
    failed = sum(result.failed is not None for result in results)
    print(
        f"searches: {expected}; asked: {len(results)}; found scenes: {len(results) - none - failed}; "
        f"found none: {none}; failed: {failed}; the details are in {details}"
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
