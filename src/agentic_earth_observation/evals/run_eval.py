"""The evaluation runner: every task of a split through the running system, scored, and written down.

    uv run eo-agent eval --split dev --models qwen3.5:9b --runs 1

A run is one model's pass over one split, with its own folder and servers (records.py, servers.py). In order:
the tasks are loaded and validated, the commit read, the model found on its server; then, per run,
the servers start, the model alone is loaded, `run.json` is written, the tasks run in a seeded order, and the
run ends complete, or aborted with its reason (decisions of 2026-11-05 and 2026-11-06).

Per task, up to three attempts: a fresh conversation each, a trace id per turn, each turn under the harness
timeout. What a turn counts as is `outcome.py`'s; a turn that raises is `errored`. An attempt lost to the
infrastructure is tried again after a pause, with a new retrieval agent; an errored one fails the task. A
scored turn is scored with the frozen `score_turn`, and a turn that expects a search gets the oracle query of
docs/evaluation.md. The third errored task and the third lost task abort the run, and so do a server found
dead, a retrieval agent on another model, a catalogue that returns nothing, a checkout that changes, and a bug
in the runner itself; what was written stays. See docs/decisions.md, 2026-11-05 to 2026-11-07.
"""

from __future__ import annotations

import asyncio
import contextlib
import platform
import random
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from langchain_core.language_models import BaseChatModel

from agentic_earth_observation.catalog.gazetteer import DEFAULT_PATH as GAZETTEER_PATH
from agentic_earth_observation.catalog.gazetteer import Gazetteer
from agentic_earth_observation.catalog.stac import CatalogError, CatalogRefused, StacCatalog
from agentic_earth_observation.catalog_mcp.server import Catalog
from agentic_earth_observation.catalog_mcp.tool_log import read_log, read_tool_calls
from agentic_earth_observation.config import ROOT, Settings
from agentic_earth_observation.domain import SearchRequest
from agentic_earth_observation.evals.golden import (
    GOLDEN_DIR,
    REFERENCE_DATE,
    ExpectedSearch,
    Task,
    evaluation_code_hash,
    load_golden,
)
from agentic_earth_observation.evals.model_server import ModelServer, ModelServerError, OllamaServer
from agentic_earth_observation.evals.outcome import task_passed, turn_outcome, turn_trace
from agentic_earth_observation.evals.records import (
    EVALUATION_RULES,
    MAX_ATTEMPTS,
    OUTPUT_FOLDER,
    RAW_RUNS,
    REPORTED_RUNS,
    TOOL_LOG_FILE,
    AttemptOutcome,
    AttemptRecord,
    CodeVersion,
    ModelBuild,
    OracleCheck,
    ProvenanceError,
    RunRecord,
    TaskCounts,
    TurnRecord,
    Value,
    append_attempt,
    code_version,
    file_hash,
    hashes,
    library_versions,
    new_run_id,
    recorded_settings,
    uncommitted_runs,
    write_run_record,
)
from agentic_earth_observation.evals.scoring import TurnScore, retrieval_metrics, score_turn
from agentic_earth_observation.evals.servers import RunServers, ServerError
from agentic_earth_observation.failures import describe
from agentic_earth_observation.llm import OLLAMA_CHECKED, OLLAMA_CONTEXT, build_chat_model
from agentic_earth_observation.orchestrator.client import A2ARetrievalClient
from agentic_earth_observation.orchestrator.graph import Orchestrator
from agentic_earth_observation.orchestrator.models import TurnOutput

Split = Literal["dev", "test"]
MODELS_UNDER_EVALUATION = ("qwen3.5:9b", "ministral-3:8b")  # fixed: another model needs a dated decision
ERRORED_LIMIT = 3  # the run aborts at the third errored task: by then the system is broken, not a task
LOST_LIMIT = 3  # the run aborts at the third task lost to the infrastructure: too little would be measured
RETRY_PAUSE_S = 30.0  # before a task is tried again, so that a short outage does not take all its attempts
# docs/evaluation.md: every expected search returned scenes when checked, except the one "no results" task.
# More than this means the catalogue or the client changed, and every search would score as "none found".
EMPTY_SEARCH_TASKS = 1
# The frozen files at tag golden-test-v1 (docs/evaluation.md, "What is frozen"; tests/test_golden.py checks
# them too). A fix to one needs a dated decision and a new tag, and then its constant. The dev split is not
# frozen: the record's golden hash covers both splits, the check here only the test split.
FROZEN_TEST_SPLIT = "86123f066e640af1464347956058eac5a9c04799f5ce7400549fb9422b16f784"
FROZEN_EVALUATION_CODE = "72d140b004d9a42eaa017ea53db68334407cea46a82b5d4b871ff45d94c3b9cb"
FROZEN_GAZETTEER = "af181e26cb48c0631f8e3e70bbe71aa7ef2e12d59b1bae0f1b141d76f842d91e"
FROZEN_EVALUATION_RULES = "9f3ae9ba0643eebc8082a4b6dc549224fc5674a85af3fa8d75f18ca3f10abb5b"
DEFAULT_RUNS = 3
DEFAULT_SHUFFLE_SEED = 1  # one order for every run of every model
ORACLE_LIMIT = 50  # the oracle query's limit when the task asks for no number of scenes
CANCEL_GRACE_S = 5.0  # how long a turn past the harness timeout gets to end once cancelled
OPTION_NAMES = ("num_ctx", "temperature", "seed", "reasoning", "num_predict", "keep_alive")


class EvaluationError(Exception):
    """The evaluation cannot start. The message says what to fix."""


class Aborted(Exception):
    """The run cannot go on. The message is the reason its record keeps."""


class HarnessTimeout(Exception):
    """A turn ran past the harness timeout. `seconds`: how long it ran before it was cancelled."""

    def __init__(self, limit_s: float, seconds: float) -> None:
        super().__init__(f"the turn ran past the harness timeout ({limit_s:g} s)")
        self.seconds = seconds


class Servers(Protocol):
    """The run's servers, as the loop needs them (RunServers, or a fake)."""

    @property
    def tool_descriptions(self) -> str: ...

    def check(self) -> None: ...

    async def restart_agent(self) -> None: ...


class Turns(Protocol):
    """The orchestrator, as the loop needs it."""

    async def run_turn(self, conversation_id: str, text: str, trace_id: str, today: date) -> TurnOutput: ...


@asynccontextmanager
async def live_orchestrator(settings: Settings) -> AsyncIterator[Turns]:
    """The real orchestrator: the configured chat model and the A2A client of the run's retrieval agent."""
    async with A2ARetrievalClient(settings) as client:
        yield Orchestrator(
            build_chat_model(settings),
            client,
            settings.llm_model,
            decision_deadline_s=settings.decision_deadline_s,
        )


def live_model_server(settings: Settings) -> AbstractAsyncContextManager[ModelServer]:
    return OllamaServer(settings.ollama_url)


def live_catalog(settings: Settings) -> Catalog:
    return StacCatalog(
        settings.stac_api_url, timeout_s=settings.stac_timeout_s, max_retries=settings.stac_max_retries
    )


@dataclass(frozen=True)
class Services:
    """What a run talks to and reads. The defaults are the real ones; tests pass fakes."""

    servers: Callable[[Settings, Path], AbstractAsyncContextManager[Servers]] = RunServers
    orchestrator: Callable[[Settings], AbstractAsyncContextManager[Turns]] = live_orchestrator
    model_server: Callable[[Settings], AbstractAsyncContextManager[ModelServer]] = live_model_server
    catalog: Callable[[Settings], Catalog] = live_catalog
    code: Callable[[], CodeVersion] = code_version
    uncommitted_runs: Callable[[], bool] = uncommitted_runs
    golden_dir: Path = GOLDEN_DIR


@dataclass(frozen=True)
class Plan:
    """What every run of an evaluation shares, checked before the first one starts."""

    settings: Settings
    split: Split
    tasks: tuple[Task, ...]  # in the order they run
    shuffle_seed: int
    code: CodeVersion
    gazetteer: Gazetteer
    reference_date: date = REFERENCE_DATE


@dataclass
class _Running:
    """One run while it goes on: what it uses, and what it has counted so far."""

    record: RunRecord  # as written at the start of the run
    plan: Plan
    folder: Path
    servers: Servers
    orchestrator: Turns
    catalog: Catalog
    code: Callable[[], CodeVersion]
    say: Callable[[str], None]
    trace_ids: set[str] = field(default_factory=set)
    tasks: Counter[str] = field(default_factory=Counter)  # passed, failed, errored, lost
    empty_searches: set[str] = field(default_factory=set)  # tasks whose expected search returned nothing

    @property
    def model(self) -> str:
        return self.record.model.model

    @property
    def log(self) -> Path:
        return self.folder / TOOL_LOG_FILE


def _print(line: str) -> None:
    """Progress on the screen. A closed pipe (`| head`) must not end a run that takes hours."""
    with contextlib.suppress(OSError):
        print(line, flush=True)


async def evaluate(
    settings: Settings,
    *,
    split: Split,
    models: Sequence[str],
    runs: int = DEFAULT_RUNS,
    for_report: bool = False,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    root: Path = ROOT / OUTPUT_FOLDER,
    services: Services | None = None,
    say: Callable[[str], None] = _print,
) -> list[RunRecord]:
    """`runs` runs of every model over the split, in order. Stops after an aborted run.

    Raises EvaluationError before any run when something would make every run worthless; what a running
    server or model server raises is passed on.
    """
    services = services or Services()
    if unknown := [m for m in models if m not in MODELS_UNDER_EVALUATION]:
        raise EvaluationError(
            f"not under evaluation: {', '.join(unknown)}; the models are {', '.join(MODELS_UNDER_EVALUATION)}"
            " (another one needs a dated decision in docs/decisions.md)"
        )
    if not models or runs < 1:
        raise EvaluationError("name at least one model and one run")
    if len(set(models)) < len(models):
        raise EvaluationError("a model is named twice: use --runs for several runs of one model")
    if split == "test" and not for_report:
        raise EvaluationError(
            "the test split runs only with --for-report: every run of it is reported, from a clean checkout. "
            "Debug on the dev split"
        )
    if split == "test" and (runs, shuffle_seed) != (DEFAULT_RUNS, DEFAULT_SHUFFLE_SEED):
        raise EvaluationError(
            f"the test split is measured in {DEFAULT_RUNS} runs per model, in the order of seed "
            f"{DEFAULT_SHUFFLE_SEED} (docs/evaluation.md): leave out --runs and --shuffle-seed"
        )
    if settings.llm_provider != "ollama":
        raise EvaluationError(
            "the Mistral provider arrives with the paid evaluation runs; set EO_LLM_PROVIDER=ollama for now"
        )
    code = services.code()
    if for_report and code.dirty:
        raise EvaluationError(
            "the checkout has changes: a run for the report needs a clean one, which its commit describes"
        )
    if for_report:
        _ready_for_the_report(services)
    gazetteer = Gazetteer.load()
    tasks = _tasks_in_order(services.golden_dir, split, shuffle_seed, gazetteer)
    plan = Plan(settings, split, tasks, shuffle_seed, code, gazetteer)
    records: list[RunRecord] = []
    async with services.model_server(settings) as model_server:
        for model in models:
            if not await model_server.has(model):
                raise EvaluationError(f"the model server does not have {model}: pull it first")
        for model in models:
            try:
                for run_index in range(1, runs + 1):
                    record = await _run(plan, model, run_index, for_report, root, model_server, services, say)
                    records.append(record)
                    if record.status == "aborted":
                        return records
            finally:
                await _unload(model_server, model, say)
    return records


def _ready_for_the_report(services: Services) -> None:
    """Refuse a run for the report that could not be reported as it stands."""
    if services.uncommitted_runs():
        raise EvaluationError(
            f"{OUTPUT_FOLDER}/{REPORTED_RUNS} holds runs that are not committed: commit them first, "
            "so that every run for the report stays on record"
        )
    frozen = {
        "the test split": (file_hash(services.golden_dir / "test.jsonl"), FROZEN_TEST_SPLIT),
        "the evaluation code": (evaluation_code_hash(), FROZEN_EVALUATION_CODE),
        "the gazetteer": (file_hash(GAZETTEER_PATH), FROZEN_GAZETTEER),
        "the labelling rules": (file_hash(EVALUATION_RULES), FROZEN_EVALUATION_RULES),
    }
    if changed := [name for name, (now, then) in frozen.items() if now != then]:
        raise EvaluationError(
            f"frozen at tag golden-test-v1 and changed since: {', '.join(changed)}. A fix to a frozen file "
            "needs a dated decision and a new tag"
        )


async def _unload(model_server: ModelServer, model: str, say: Callable[[str], None]) -> None:
    """Free the model's memory. A model left loaded changes nothing about how the runs ended."""
    try:
        await model_server.unload(model)
    except ModelServerError as err:
        say(f"warning: {model} may still be loaded: {err}")


def _tasks_in_order(
    golden_dir: Path, split: Split, shuffle_seed: int, gazetteer: Gazetteer
) -> tuple[Task, ...]:
    """The split's tasks, validated, in the seeded order."""
    tasks = [task for task in load_golden(golden_dir, gazetteer) if task.split == split]
    if not tasks:
        raise EvaluationError(f"the {split} split has no tasks")
    random.Random(shuffle_seed).shuffle(tasks)  # noqa: S311 - a reproducible order, nothing secret
    return tuple(tasks)


async def _run(
    plan: Plan,
    model: str,
    run_index: int,
    for_report: bool,
    root: Path,
    model_server: ModelServer,
    services: Services,
    say: Callable[[str], None],
) -> RunRecord:
    """One run: its folder and servers, the model loaded alone, its record written, then the tasks."""
    # The servers start from the checkout, while this process runs the code it started with.
    if services.code() != plan.code:
        raise EvaluationError(
            "the checkout changed since this evaluation started: its runs would not describe one commit. "
            "Start the evaluation again"
        )
    started = datetime.now(UTC)
    run_id = new_run_id(model, run_index, started)
    folder = root / (REPORTED_RUNS if for_report else RAW_RUNS) / run_id
    if folder.exists():
        raise EvaluationError(f"the folder of run {run_id} exists already: every run gets a new one")
    settings = plan.settings.model_copy(update={"llm_model": model})
    say(f"run {run_id}: {len(plan.tasks)} {plan.split} tasks, writing to {folder}")
    version = await model_server.version()
    if version != OLLAMA_CHECKED:
        say(
            f"warning: Ollama {version}: how its errors are read was checked on {OLLAMA_CHECKED} (see llm.py)"
        )
    async with services.servers(settings, folder) as servers:
        loaded = await model_server.prepare(model, OLLAMA_CONTEXT)
        build = ModelBuild(
            provider=settings.llm_provider,
            model=model,
            digest=loaded.digest,
            server_version=version,
            options=chat_model_options(build_chat_model(settings)),
        )
        record = RunRecord(
            run_id=run_id,
            status="running",
            split=plan.split,
            run_index=run_index,
            started_at=started,
            code=plan.code,
            model=build,
            hashes=hashes(servers.tool_descriptions),
            reference_date=plan.reference_date,
            shuffle_seed=plan.shuffle_seed,
            task_order=[task.id for task in plan.tasks],
            settings=recorded_settings(settings),
            libraries=library_versions(),
            platform=platform.platform(),
        )
        async with services.orchestrator(settings) as orchestrator:
            catalog = services.catalog(settings)
            write_run_record(folder, record)
            return await run_tasks(
                record, plan, folder, servers, orchestrator, catalog, code=services.code, say=say
            )


def chat_model_options(model: BaseChatModel) -> dict[str, Value]:
    """The options the chat model was built with, read from the model itself, for the run record."""
    return {name: getattr(model, name) for name in OPTION_NAMES if hasattr(model, name)}


async def run_tasks(
    record: RunRecord,
    plan: Plan,
    folder: Path,
    servers: Servers,
    orchestrator: Turns,
    catalog: Catalog,
    *,
    code: Callable[[], CodeVersion],
    say: Callable[[str], None] = _print,
) -> RunRecord:
    """Every task of the run, then the run's final record, written and returned.

    `record` is the run's record as written at its start. The run ends `complete`, or `aborted` with the
    reason; Ctrl+C and an error of the runner's own are written down as aborted and then raised.
    """
    run = _Running(record, plan, folder, servers, orchestrator, catalog, code, say)
    try:
        await _tasks(run)
        _alive(servers)
    except Aborted as err:
        return _ended(run, "aborted", str(err))
    except BaseException as err:
        interrupted = isinstance(err, KeyboardInterrupt | asyncio.CancelledError)
        with contextlib.suppress(Exception):  # the error in hand is the one to see, whatever the disk says
            _ended(run, "aborted", "interrupted" if interrupted else describe(err))
        raise
    return _ended(run, "complete", None)


async def _tasks(run: _Running) -> None:
    for position, task in enumerate(run.plan.tasks, start=1):
        for number in range(1, MAX_ATTEMPTS + 1):
            if number > 1:
                await asyncio.sleep(RETRY_PAUSE_S)
            _alive(run.servers)
            attempt = await _attempt(task, f"{run.record.run_id}-t{position:02d}-a{number}", number, run)
            append_attempt(run.folder, attempt)
            run.say(_line(position, len(run.plan.tasks), attempt))
            if attempt.outcome != "infrastructure":
                break
            if number < MAX_ATTEMPTS:
                # The agent may still be at the turn its client gave up on, holding the model: a new one is
                # ready in under a second (decision of 2026-11-07).
                await _fresh_agent(run.servers)
        # Counted before anything else can fail: a task that ended is in the record whatever happens next.
        _count(attempt, run)
        if attempt.outcome == "errored" and run.tasks["errored"] >= ERRORED_LIMIT:
            raise Aborted(f"{ERRORED_LIMIT} tasks errored")
        if run.tasks["lost"] >= LOST_LIMIT:
            raise Aborted(f"{LOST_LIMIT} tasks were lost to the infrastructure; the last: {attempt.reason}")
        if attempt.outcome != "scored":  # errored, or lost: the agent may be stuck at the last turn
            await _fresh_agent(run.servers)


def _count(attempt: AttemptRecord, run: _Running) -> None:
    """What the task came to, by its last attempt."""
    if attempt.outcome == "scored":
        run.tasks["passed" if attempt.passed else "failed"] += 1
    else:
        run.tasks["errored" if attempt.outcome == "errored" else "lost"] += 1


async def _attempt(task: Task, conversation: str, number: int, run: _Running) -> AttemptRecord:
    """One attempt of a task: its turns, in order, until one is not scored."""
    plan = run.plan
    turns: list[TurnRecord] = []
    outputs: list[TurnOutput] = []
    scores: list[TurnScore] = []
    known: set[str] = set()  # scene ids the tools returned in earlier turns of this conversation
    outcome: AttemptOutcome = "scored"
    reason: str | None = None
    for position, turn in enumerate(task.turns, start=1):
        trace_id = f"{conversation}-u{position}"
        run.trace_ids.add(trace_id)
        started = time.perf_counter()
        try:
            output = await _bounded(
                run.orchestrator.run_turn(conversation, turn.user, trace_id, plan.reference_date),
                plan.settings.harness_timeout_s,
            )
        except HarnessTimeout as err:
            outcome, reason = "infrastructure", str(err)
            seconds = err.seconds
            calls = read_tool_calls(run.log, trace_id)
            turns.append(TurnRecord(trace_id=trace_id, seconds=seconds, error=reason, calls=calls))
            break
        except Exception as err:  # the retrieval agent failed or rejected the request, or a bug: rule 1
            outcome, reason = "errored", f"turn {position} raised {describe(err)}"
            seconds = time.perf_counter() - started
            calls = read_tool_calls(run.log, trace_id)
            turns.append(TurnRecord(trace_id=trace_id, seconds=seconds, error=describe(err), calls=calls))
            break
        seconds = time.perf_counter() - started
        calls = read_tool_calls(run.log, trace_id)
        _same_model(output, run.model)
        try:
            counts_as, why = turn_outcome(output, calls)
            trace = turn_trace(output, calls)
        except ValueError as err:
            raise Aborted(f"a bug in the runner: {err}") from err
        if counts_as != "scored":
            turns.append(TurnRecord(trace_id=trace_id, seconds=seconds, output=output, calls=calls))
            outcome, reason = counts_as, f"turn {position}: {why}"
            break
        score = score_turn(turn.expect, trace, plan.gazetteer, frozenset(known))
        known.update(scene_id for call in calls for scene_id in call.returned_ids)
        oracle = None
        if turn.expect.search is not None:
            oracle = await _oracle(turn.expect.search, output, run.catalog, plan)
            _notice_an_empty_catalogue(oracle, task, run)
        turns.append(
            TurnRecord(
                trace_id=trace_id, seconds=seconds, output=output, calls=calls, score=score, oracle=oracle
            )
        )
        outputs.append(output)
        scores.append(score)
    passed: bool | None = None
    if outcome == "scored":
        try:
            passed = task_passed(scores, outputs)
        except ValueError as err:
            raise Aborted(f"a bug in the runner: {err}") from err
    else:  # an attempt that is not scored counts for nothing, not even with the turns before the last
        turns = [turn.model_copy(update={"score": None, "oracle": None}) for turn in turns]
        passed = False if outcome == "errored" else None
    return AttemptRecord(
        run_id=run.record.run_id,
        task_id=task.id,
        attempt=number,
        outcome=outcome,
        passed=passed,
        reason=reason,
        turns=turns,
    )


async def _bounded(turn: Coroutine[Any, Any, TurnOutput], limit_s: float) -> TurnOutput:
    """The turn's output, or HarnessTimeout after `limit_s` seconds whatever the turn does.

    `asyncio.wait` returns on time even when the turn swallows its cancellation, where `asyncio.timeout` would
    return late (decision of 2026-11-06). A turn past the limit is cancelled and gets a moment to end.
    """
    task = asyncio.ensure_future(turn)
    started = time.perf_counter()
    try:
        done, _ = await asyncio.wait({task}, timeout=limit_s)
    except asyncio.CancelledError:  # the run itself is cancelled (Ctrl+C): the turn goes with it
        task.cancel()
        raise
    if not done:
        seconds = time.perf_counter() - started  # the turn's own time: the grace below is the runner's
        task.add_done_callback(_outcome_not_needed)
        task.cancel()
        await asyncio.wait({task}, timeout=CANCEL_GRACE_S)
        raise HarnessTimeout(limit_s, seconds)
    if task.cancelled():  # nobody here asked for it, so this is not an interruption of the run
        raise RuntimeError("the turn ended as cancelled, though nothing cancelled it")
    return task.result()


def _outcome_not_needed(task: asyncio.Future[TurnOutput]) -> None:
    """Take the outcome of an abandoned turn, so that asyncio does not report it as never retrieved."""
    if not task.cancelled():
        task.exception()


def _same_model(output: TurnOutput, model: str) -> None:
    """A retrieval agent on another model than the run's makes the run worthless (decision of 2026-11-05)."""
    if output.retrieval is None:
        return
    reported = output.retrieval.usage.model  # empty when the agent could not be reached at all
    if reported and reported != model:
        raise Aborted(f"the retrieval agent runs {reported!r}, not the run's model {model!r}")


def _notice_an_empty_catalogue(oracle: OracleCheck, task: Task, run: _Running) -> None:
    """Nothing found where something should be: the scoring takes the run's own search as the truth about
    whether scenes exist, so a catalogue that returned nothing would pass every search as "none found"."""
    if oracle.failed is None and not oracle.ids:
        run.empty_searches.add(task.id)
    if len(run.empty_searches) > EMPTY_SEARCH_TASKS:
        raise Aborted(
            f"the catalogue returned nothing for {len(run.empty_searches)} tasks that expect a search, where "
            f"the golden set has {EMPTY_SEARCH_TASKS}: has the catalogue or its client changed?"
        )


async def _oracle(search: ExpectedSearch, output: TurnOutput, catalog: Catalog, plan: Plan) -> OracleCheck:
    """The direct catalogue query of a turn that expects a search (docs/evaluation.md, Reported alongside)."""
    place = plan.gazetteer.resolve(search.place)
    if place is None:  # load_golden checked every expected place against the gazetteer
        raise Aborted(f"a bug in the runner: the expected place {search.place!r} is not in the gazetteer")
    limit = search.limit or ORACLE_LIMIT
    request = SearchRequest.model_validate(
        {
            "product": search.product,
            "bbox": place.bbox,
            "dates": {"start": search.start, "end": search.end},
            "max_cloud_cover": None if search.cloud is None else search.cloud[1],
            "limit": limit,
        }
    )
    try:
        async with asyncio.timeout(plan.settings.stac_deadline_s):
            scenes = await asyncio.to_thread(catalog.search, request)
    except TimeoutError:
        return OracleCheck(failed=f"the catalogue did not answer within {plan.settings.stac_deadline_s:g} s")
    except CatalogRefused as err:  # the expected search is ours: a refusal is a bug, not a missing metric
        raise Aborted(f"the catalogue refused an expected search: {describe(err)}") from err
    except CatalogError as err:
        return OracleCheck(failed=describe(err))
    ids = [scene.id for scene in scenes]
    returned = [scene.id for scene in output.retrieval.scenes] if output.retrieval is not None else []
    # Precision is the share of what the answer cites, so a turn that cites nothing has none. Recall is the
    # share of the right scenes that the tools returned, only for a task that asks for a number of scenes.
    precision = retrieval_metrics(output.cited_ids, ids, limit)[0] if output.cited_ids else None
    recall = retrieval_metrics(returned, ids, limit)[1] if search.limit is not None else None
    return OracleCheck(ids=ids, precision=precision, recall=recall)


def _alive(servers: Servers) -> None:
    try:
        servers.check()
    except ServerError as err:
        raise Aborted(str(err)) from err


async def _fresh_agent(servers: Servers) -> None:
    try:
        await servers.restart_agent()
    except ServerError as err:
        raise Aborted(str(err)) from err


def _checkout_changed(run: _Running) -> str | None:
    """Why the record's commit no longer describes the run, if so: a new agent imports the checkout again."""
    try:
        now = run.code()
    except ProvenanceError as err:
        return f"the commit could not be read at the end of the run: {err}"
    if now != run.record.code:
        return "the checkout changed during the run: its record would not describe what ran"
    return None


def _ended(run: _Running, status: Literal["complete", "aborted"], reason: str | None) -> RunRecord:
    """The run's final record, written: how and when it ended, and what it counted."""
    problems = []
    try:
        unattributed = sum(1 for call in read_log(run.log) if call.trace_id not in run.trace_ids)
    except (OSError, ValueError) as err:
        unattributed = 0
        problems.append(f"the tool-call log cannot be read: {describe(err)}")
    if (changed := _checkout_changed(run)) is not None:
        problems.append(changed)
    if problems:  # a run with either problem is not complete; an aborted one keeps its first reason as well
        status, reason = "aborted", "; ".join(([reason] if reason else []) + problems)
    final = run.record.model_copy(
        update={
            "status": status,
            "aborted_reason": reason,
            "finished_at": datetime.now(UTC),
            "unattributed_calls": unattributed,
            "tasks": TaskCounts(**run.tasks),
        }
    )
    counts = ", ".join(f"{run.tasks[what]} {what}" for what in ("passed", "failed", "errored", "lost"))
    try:
        write_run_record(run.folder, final)
    except OSError as err:  # the attempts are on disk and run.json still says `running`: say so, with counts
        raise EvaluationError(
            f"run {run.record.run_id} ended ({status}; {counts}) but its record could not be written: "
            f"{describe(err)}. Its attempts are in {run.folder.name}; run.json still says running"
        ) from err
    run.say(f"run {run.record.run_id}: {counts}" + (f"; aborted: {reason}" if status == "aborted" else ""))
    return final


def _line(position: int, total: int, record: AttemptRecord) -> str:
    """One line of progress per attempt."""
    seconds = sum(turn.seconds for turn in record.turns)
    if record.outcome == "scored":
        what = "passed" if record.passed else "failed"
    else:
        what = f"{record.outcome}: {record.reason}"
    return f"[{position}/{total}] {record.task_id} attempt {record.attempt}: {what} ({seconds:.0f} s)"
