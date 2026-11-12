"""Tests for the evaluation runner: the loop over tasks and attempts, what ends a run, and the records it
writes. No network and no model: the servers, the orchestrator, the model server and the catalogue are fakes,
and the tool-call log is a real file that the fake orchestrator writes as the MCP server would.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import random
import time
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import pytest
from pydantic import ValidationError

from agentic_earth_observation.catalog.gazetteer import Gazetteer, Place
from agentic_earth_observation.catalog.stac import CatalogRefused, SceneSummary, StacCatalog
from agentic_earth_observation.catalog_mcp.tool_log import ToolCallLog, ToolCallRecord
from agentic_earth_observation.config import Settings
from agentic_earth_observation.domain import Product, SearchRequest
from agentic_earth_observation.evals import run_eval
from agentic_earth_observation.evals.golden import (
    GOLDEN_DIR,
    REFERENCE_DATE,
    Category,
    Expectation,
    ExpectedSearch,
    Route,
    Task,
    Turn,
    load_golden,
    load_tasks,
)
from agentic_earth_observation.evals.model_server import LoadedModel, ModelServerError, OllamaServer
from agentic_earth_observation.evals.records import (
    TOOL_LOG_FILE,
    AttemptRecord,
    CodeVersion,
    Hashes,
    ModelBuild,
    ProvenanceError,
    RunRecord,
    TaskCounts,
    read_attempts,
    read_run_record,
)
from agentic_earth_observation.evals.run_eval import (
    MODELS_UNDER_EVALUATION,
    EvaluationError,
    Plan,
    Services,
    chat_model_options,
    evaluate,
    run_tasks,
)
from agentic_earth_observation.evals.servers import RunServers, ServerDied, ServerError
from agentic_earth_observation.llm import OLLAMA_CONTEXT, build_chat_model
from agentic_earth_observation.orchestrator.client import RetrievalAgentError
from agentic_earth_observation.orchestrator.graph import Orchestrator
from agentic_earth_observation.orchestrator.models import TurnOutput, scored_status
from agentic_earth_observation.retrieval_agent.models import (
    PreviousSearch,
    RetrievalRequest,
    RetrievalResult,
    Usage,
)
from tests.fakes import SCENE, FakeCatalog

pytestmark = pytest.mark.anyio

MODEL = "qwen3.5:9b"
OTHER_MODEL = "ministral-3:8b"
GAZETTEER = Gazetteer.load()
DEV_TASKS = load_tasks(GOLDEN_DIR / "dev.jsonl")  # milan, naples follow-up, german milan, weather
MILAN, NAPLES, GERMAN, WEATHER = (task.turns[0].user for task in DEV_TASKS)
NAPLES_AGAIN = DEV_TASKS[1].turns[1].user
STARTED = datetime(2026, 11, 7, 9, 30, 45, tzinfo=UTC)
RUN_ID = "20261107T093045Z-qwen3.5-9b-r1"
A_HASH = "ab" * 32
COMMIT = CodeVersion(commit="0123456789abcdef0123456789abcdef01234567", dirty=False)
OTHER_SCENE = SCENE.model_copy(update={"id": "S2B_T32TNR_20250312T101729_L2A"})
NAPLES_SCENE = SCENE.model_copy(
    update={"id": "S1A_IW_GRDH_1SDV_20250903T050000_NAPLES", "product": Product.SENTINEL_1_GRD}
)
LATER_SCENE = NAPLES_SCENE.model_copy(update={"id": "S1A_IW_GRDH_1SDV_20250910T050000_NAPLES"})
# Timeouts a test can wait for: every limit nests as Settings requires, and the harness gives 0.3 s.
QUICK: dict[str, float] = {
    "stac_timeout_s": 0.01,
    "stac_deadline_s": 0.02,
    "mcp_timeout_s": 0.03,
    "turn_deadline_s": 0.04,
    "a2a_timeout_s": 0.05,
    "model_timeout_s": 0.01,
    "decision_deadline_s": 0.1,
    "harness_timeout_s": 0.3,
}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _no_pause_before_a_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_eval, "RETRY_PAUSE_S", 0.0)


def settings(**changes: Any) -> Settings:  # noqa: ANN401 - any setting, validated by Settings itself
    return Settings(_env_file=None, **changes)  # type: ignore[call-arg]  # a pydantic-settings init option


def box(place: str) -> list[float]:
    found = GAZETTEER.resolve(place)
    assert found is not None
    return found.bbox.as_list()


def arguments(place: str, product: str, start: str, end: str, **more: Any) -> dict[str, Any]:  # noqa: ANN401 - cloud, limit
    return {"product": product, "bbox": box(place), "start_date": start, "end_date": end, **more}


MILAN_MARCH = arguments("Milan", "sentinel-2-l2a", "2025-03-01", "2025-03-31", max_cloud_cover=20)
MILAN_OCTOBER = arguments("Milan", "sentinel-2-l2a", "2025-10-01", "2025-10-31", max_cloud_cover=10)
NAPLES_FIRST_WEEK = arguments("Naples", "sentinel-1-grd", "2025-09-01", "2025-09-07")
NAPLES_SECOND_WEEK = arguments("Naples", "sentinel-1-grd", "2025-09-08", "2025-09-14")


def call(
    tool: str,
    request_id: int,
    /,
    *,
    ok: bool = True,
    ids: list[str] | None = None,
    **tool_arguments: Any,  # noqa: ANN401 - the tool's arguments
) -> ToolCallRecord:
    """One call as the MCP server records it; the trace id is filled in when the call is logged."""
    return ToolCallRecord(
        trace_id="unfiled",
        request_id=request_id,
        tool=tool,
        arguments=tool_arguments,
        ok=ok,
        returned_ids=ids or [],
    )


def found(
    search: dict[str, Any], place: str, scenes: list[SceneSummary], model: str = MODEL
) -> RetrievalResult:
    """The retrieval agent's result for a search that found these scenes."""
    previous = PreviousSearch.model_validate_json(json.dumps({**search, "place": place}))
    return RetrievalResult(
        status="ok", scenes=scenes, search=previous, usage=Usage(model=model, model_calls=2, seconds=1.0)
    )


def the_search(search: dict[str, Any], place: str) -> PreviousSearch:
    return PreviousSearch.model_validate_json(json.dumps({**search, "place": place}))


def delegated(result: RetrievalResult, cited: list[str] | None = None) -> TurnOutput:
    """A delegated turn that ended with this result, as run_turn reports it."""
    return TurnOutput(
        trace_id="unfiled",
        route=Route.DELEGATE,
        status=scored_status("delegate", result),
        answer="(the reply)",
        cited_ids=[scene.id for scene in result.scenes] if cited is None else cited,
        failure=result.failure,
        failure_detail=result.failure_detail,
        infra_error=result.infra_error,
        retrieval=result,
    )


DECLINED = TurnOutput(trace_id="unfiled", route=Route.DECLINE, status="declined", answer="(the reply)")
UNREACHABLE = delegated(RetrievalResult(status="not_searched", infra_error="A2A ReadTimeout"))


class Step:
    """What the orchestrator does for one turn: logs the calls and answers, raises, or never answers."""

    def __init__(
        self,
        output: TurnOutput = DECLINED,
        calls: list[ToolCallRecord] | None = None,
        *,
        raises: BaseException | None = None,
        hangs: bool = False,
        swallows_cancellation: bool = False,
        wrong_trace_id: bool = False,
        corrupt_line: bool = False,
    ) -> None:
        self.output = output
        self.calls = calls or []
        self.raises = raises
        self.hangs = hangs
        self.swallows_cancellation = swallows_cancellation
        self.wrong_trace_id = wrong_trace_id
        self.corrupt_line = corrupt_line


class ScriptedOrchestrator:
    """Answers each user text from its list of steps, consumed in order; the last step repeats."""

    def __init__(self, log: Path, script: dict[str, list[Step]]) -> None:
        self.log = ToolCallLog(log)
        self.script = {text: list(steps) for text, steps in script.items()}
        self.turns: list[tuple[str, str, str]] = []  # conversation id, trace id, text
        self.cancelled: list[str] = []
        self.model: str | None = None  # the model the retrieval agent reports, if not the one in the script
        self.abandoned: list[asyncio.Task[Any]] = []  # turns that swallowed their cancellation

    async def run_turn(self, conversation_id: str, text: str, trace_id: str, today: date) -> TurnOutput:
        assert today == REFERENCE_DATE
        self.turns.append((conversation_id, trace_id, text))
        steps = self.script[text]
        step = steps.pop(0) if len(steps) > 1 else steps[0]
        for record in step.calls:
            self.log.write(record.model_copy(update={"trace_id": trace_id}))
        if step.hangs:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.append(trace_id)
                if not step.swallows_cancellation:
                    raise
                task = asyncio.current_task()
                assert task is not None
                self.abandoned.append(task)
                await asyncio.sleep(30)  # the test cancels it again, once it has seen the run go on
            return step.output
        if step.raises is not None:
            raise step.raises
        if step.corrupt_line:
            assert self.log.path is not None
            with self.log.path.open("a", encoding="utf-8") as log:
                log.write('{"oops"\n')
        output = step.output
        if self.model is not None and output.retrieval is not None and output.retrieval.usage.model:
            usage = output.retrieval.usage.model_copy(update={"model": self.model})
            retrieval = output.retrieval.model_copy(update={"usage": usage})
            output = output.model_copy(update={"retrieval": retrieval})
        return output if step.wrong_trace_id else output.model_copy(update={"trace_id": trace_id})


def passing_script() -> dict[str, list[Step]]:
    """Every dev task answered as the rules expect."""
    return {
        MILAN: [
            Step(
                delegated(found(MILAN_MARCH, "Milan", [SCENE])),
                [
                    call("resolve_place", 1, name="Milan"),
                    call("search_scenes", 2, ids=[SCENE.id], **MILAN_MARCH),
                ],
            )
        ],
        GERMAN: [
            Step(
                delegated(found(MILAN_OCTOBER, "Mailand", [SCENE])),
                [
                    call("resolve_place", 1, name="Mailand"),
                    call("search_scenes", 2, ids=[SCENE.id], **MILAN_OCTOBER),
                ],
            )
        ],
        NAPLES: [
            Step(
                delegated(found(NAPLES_FIRST_WEEK, "Naples", [NAPLES_SCENE])),
                [
                    call("resolve_place", 1, name="Naples"),
                    call("search_scenes", 2, ids=[NAPLES_SCENE.id], **NAPLES_FIRST_WEEK),
                ],
            )
        ],
        NAPLES_AGAIN: [
            Step(
                delegated(found(NAPLES_SECOND_WEEK, "Naples", [LATER_SCENE])),
                [call("search_scenes", 1, ids=[LATER_SCENE.id], **NAPLES_SECOND_WEEK)],
            )
        ],
        WEATHER: [Step(DECLINED)],
    }


class FakeServers:
    """Stands in for RunServers: records what the loop asks of it, and can be told that a server died."""

    def __init__(self, folder: Path | None = None) -> None:
        self.folder = folder  # when given, entering creates it and the tool-call log, as RunServers does
        self.tool_descriptions = A_HASH
        self.dead: str | None = None  # from `dies_at_check` on, check() raises ServerDied with this message
        self.dies_at_check = 1
        self.restart_fails: str | None = None
        self.restarts_that_work: int | None = None  # with restart_fails: how many succeed before it applies
        self.on_check: Callable[[int], None] | None = None  # called with the number of the check
        self.checks = 0
        self.restarts = 0
        self.stopped = False

    async def __aenter__(self) -> Self:
        if self.folder is not None:
            self.folder.mkdir(parents=True)
            ToolCallLog(self.folder / TOOL_LOG_FILE)
        return self

    async def __aexit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None
    ) -> None:
        self.stopped = True

    def check(self) -> None:
        self.checks += 1
        if self.on_check is not None:
            self.on_check(self.checks)
        if self.dead is not None and self.checks >= self.dies_at_check:
            raise ServerDied(self.dead)

    async def restart_agent(self) -> None:
        self.check()
        if self.restart_fails is not None and self.restarts >= (self.restarts_that_work or 0):
            raise ServerError(self.restart_fails)
        self.restarts += 1


def replacements_that_work(run: Run, count: int) -> None:
    """The first `count` replacements of the agent succeed; the next finds the port taken."""
    run.servers.restart_fails = "something already answers on port 8002"
    run.servers.restarts_that_work = count


class FakeModelServer:
    def __init__(self, *models: str) -> None:
        self.models = set(models)
        self.calls: list[str] = []
        self.prepare_fails: str | None = None
        self.warm_up_fails: str | None = None
        self.unload_fails: str | None = None
        self.server_version = "0.40.1"

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None
    ) -> None:
        self.calls.append("closed")

    async def version(self) -> str | None:
        return self.server_version

    async def has(self, model: str) -> bool:
        self.calls.append(f"has {model}")
        return model in self.models

    async def prepare(self, model: str, context: int) -> LoadedModel:
        self.calls.append(f"prepare {model} {context}")
        if self.prepare_fails is not None:
            raise ModelServerError(self.prepare_fails)
        return LoadedModel(model=model, digest="d63f" * 16, context_length=context)

    async def warm_up(self, model: str, context: int) -> float:
        self.calls.append(f"warm up {model} {context}")
        if self.warm_up_fails is not None:
            raise ModelServerError(self.warm_up_fails)
        return 1.25

    async def unload(self, model: str) -> None:
        self.calls.append(f"unload {model}")
        if self.unload_fails is not None:
            raise ModelServerError(self.unload_fails)


class EmptyForMilan(FakeCatalog):
    def search(self, request: SearchRequest) -> list[SceneSummary]:
        self.requests.append(request)
        return [] if request.bbox.as_list() == box("Milan") and request.dates.start.month == 3 else [SCENE]


class RefusingCatalog(FakeCatalog):
    def search(self, request: SearchRequest) -> list[SceneSummary]:
        raise CatalogRefused("HTTP 400")


class SlowCatalog(FakeCatalog):
    def search(self, request: SearchRequest) -> list[SceneSummary]:
        time.sleep(0.3)
        return super().search(request)


def running_record(tasks: tuple[Task, ...], model: str = MODEL) -> RunRecord:
    return RunRecord(
        run_id=RUN_ID,
        status="running",
        split="dev",
        run_index=1,
        started_at=STARTED,
        code=COMMIT,
        model=ModelBuild(provider="ollama", model=model, options={"seed": 42}),
        hashes=Hashes(
            retrieval_prompt=A_HASH,
            decide_prompt=A_HASH,
            tool_descriptions=A_HASH,
            golden=A_HASH,
            evaluation_code=A_HASH,
            gazetteer=A_HASH,
            evaluation_rules=A_HASH,
        ),
        reference_date=REFERENCE_DATE,
        shuffle_seed=1,
        task_order=[task.id for task in tasks],
        settings={"num_predict": 1024},
        libraries={"python": "3.13"},
        platform="test",
    )


class Run:
    """One run of `run_tasks` over the fakes, with its folder and log in place."""

    def __init__(
        self,
        tmp_path: Path,
        tasks: tuple[Task, ...] = tuple(DEV_TASKS),
        *,
        catalog: FakeCatalog | None = None,
        config: Settings | None = None,
    ) -> None:
        self.folder = tmp_path / RUN_ID
        self.folder.mkdir()
        self.log = ToolCallLog(self.folder / TOOL_LOG_FILE)
        self.plan = Plan(config or settings(), "dev", tasks, 1, COMMIT, GAZETTEER)
        self.record = running_record(tasks)
        self.servers = FakeServers()
        self.catalog = catalog or FakeCatalog()
        self.lines: list[str] = []
        self.say: Callable[[str], None] = self.lines.append  # a test may replace it
        self.orchestrator: ScriptedOrchestrator | None = None
        self.code: Callable[[], CodeVersion] = lambda: COMMIT

    async def go(self, script: dict[str, list[Step]]) -> RunRecord:
        self.orchestrator = ScriptedOrchestrator(self.folder / TOOL_LOG_FILE, script)
        return await run_tasks(
            self.record,
            self.plan,
            self.folder,
            self.servers,
            self.orchestrator,
            self.catalog,
            code=lambda: self.code(),
            say=self.say,
        )

    def turns(self) -> list[tuple[str, str, str]]:
        assert self.orchestrator is not None
        return self.orchestrator.turns

    def attempts(self) -> list[AttemptRecord]:
        return read_attempts(self.folder)

    def written(self) -> RunRecord:
        return read_run_record(self.folder)


def summary(attempt: AttemptRecord) -> tuple[str, int, str, bool | None]:
    return attempt.task_id, attempt.attempt, attempt.outcome, attempt.passed


class TestARunOverTheDevSplit:
    async def test_every_task_runs_once_and_the_run_completes(self, tmp_path: Path) -> None:
        run = Run(tmp_path)
        final = await run.go(passing_script())
        assert final.status == "complete"
        assert final.aborted_reason is None
        assert final.finished_at is not None
        assert final.unattributed_calls == 0
        assert run.written() == final
        assert [summary(a) for a in run.attempts()] == [
            ("dev-s2-milan-march", 1, "scored", True),
            ("dev-follow-naples-weeks", 1, "scored", True),
            ("dev-robust-german-milan", 1, "scored", True),
            ("dev-oos-weather-naples", 1, "scored", True),
        ]
        assert all(attempt.run_id == RUN_ID and attempt.reason is None for attempt in run.attempts())

    async def test_each_attempt_is_a_fresh_conversation_with_a_trace_id_per_turn(
        self, tmp_path: Path
    ) -> None:
        run = Run(tmp_path)
        await run.go(passing_script())
        assert run.turns() == [
            (f"{RUN_ID}-t01-a1", f"{RUN_ID}-t01-a1-u1", MILAN),
            (f"{RUN_ID}-t02-a1", f"{RUN_ID}-t02-a1-u1", NAPLES),
            (f"{RUN_ID}-t02-a1", f"{RUN_ID}-t02-a1-u2", NAPLES_AGAIN),
            (f"{RUN_ID}-t03-a1", f"{RUN_ID}-t03-a1-u1", GERMAN),
            (f"{RUN_ID}-t04-a1", f"{RUN_ID}-t04-a1-u1", WEATHER),
        ]
        naples = run.attempts()[1]
        assert [turn.trace_id for turn in naples.turns] == [f"{RUN_ID}-t02-a1-u1", f"{RUN_ID}-t02-a1-u2"]

    async def test_a_turn_record_holds_the_output_the_logged_calls_and_the_score(
        self, tmp_path: Path
    ) -> None:
        run = Run(tmp_path)
        await run.go(passing_script())
        (turn,) = run.attempts()[0].turns
        assert turn.output is not None
        assert turn.output.trace_id == f"{RUN_ID}-t01-a1-u1"
        assert [(c.tool, c.trace_id) for c in turn.calls] == [
            ("resolve_place", turn.trace_id),
            ("search_scenes", turn.trace_id),
        ]
        assert turn.score is not None
        assert turn.score.passed
        assert turn.error is None
        assert turn.seconds >= 0

    async def test_the_servers_are_checked_before_every_attempt_and_at_the_end(self, tmp_path: Path) -> None:
        run = Run(tmp_path)
        await run.go(passing_script())
        assert run.servers.checks == 5
        assert run.servers.restarts == 0

    async def test_progress_is_said_per_attempt(self, tmp_path: Path) -> None:
        run = Run(tmp_path)
        await run.go(passing_script())
        assert run.lines == [
            "[1/4] dev-s2-milan-march attempt 1: passed (0 s)",
            "[2/4] dev-follow-naples-weeks attempt 1: passed (0 s)",
            "[3/4] dev-robust-german-milan attempt 1: passed (0 s)",
            "[4/4] dev-oos-weather-naples attempt 1: passed (0 s)",
            f"run {RUN_ID}: 4 passed, 0 failed, 0 errored, 0 lost",
        ]


class TestTheOracleQuery:
    async def test_a_turn_that_expects_a_search_gets_the_expected_query(self, tmp_path: Path) -> None:
        run = Run(tmp_path)
        await run.go(passing_script())
        milan = run.catalog.requests[0]
        assert milan.product is Product.SENTINEL_2_L2A
        assert milan.bbox.as_list() == box("Milan")
        assert (str(milan.dates.start), str(milan.dates.end)) == ("2025-03-01", "2025-03-31")
        assert milan.max_cloud_cover == 20  # the top of the accepted range
        assert milan.limit == 50  # no number of scenes asked for
        assert len(run.catalog.requests) == 4  # not for the declined task
        (turn,) = run.attempts()[0].turns
        assert turn.oracle is not None
        assert (turn.oracle.ids, turn.oracle.precision, turn.oracle.recall) == ([SCENE.id], 1.0, None)
        weather = run.attempts()[3].turns[0]
        assert weather.oracle is None
        assert run.catalog.requests[1].max_cloud_cover is None  # a radar search has no cloud limit

    async def test_an_expected_place_the_gazetteer_lacks_is_a_bug_of_the_runner(self, tmp_path: Path) -> None:
        # load_golden checks every expected place against the gazetteer, so only a bug gets this far.
        run = Run(tmp_path)
        plan = dataclasses.replace(run.plan, gazetteer=Gazetteer({}))
        search = DEV_TASKS[0].turns[0].expect.search
        assert search is not None
        with pytest.raises(
            run_eval.Aborted, match="a bug in the runner: the expected place 'milan' is not in the gazetteer"
        ):
            await run_eval._oracle(search, DECLINED, run.catalog, plan)
        assert run.catalog.requests == []

    async def test_another_lookup_error_is_not_taken_for_a_missing_place(self, tmp_path: Path) -> None:
        # Only the place the gazetteer lacks is reported as the runner's bug: any other error keeps its type.
        class Broken(Gazetteer):
            def resolve(self, name: str) -> Place | None:
                raise KeyError(name)

        run = Run(tmp_path)
        plan = dataclasses.replace(run.plan, gazetteer=Broken({}))
        search = DEV_TASKS[0].turns[0].expect.search
        assert search is not None
        with pytest.raises(KeyError):
            await run_eval._oracle(search, DECLINED, run.catalog, plan)

    async def test_recall_is_measured_when_the_task_asks_for_a_number_of_scenes(self, tmp_path: Path) -> None:
        text = "Two Sentinel-2 scenes of Milan from March 2025"
        task = Task(
            id="dev-two-milan",
            split="dev",
            category=Category.SINGLE_SEARCH,
            turns=[
                Turn(
                    user=text,
                    expect=Expectation(
                        route=Route.DELEGATE,
                        required_tools=["resolve_place", "search_scenes"],
                        search=ExpectedSearch(
                            product=Product.SENTINEL_2_L2A,
                            place="milan",
                            start=date(2025, 3, 1),
                            end=date(2025, 3, 31),
                            cloud=(0, 20),
                            limit=2,
                        ),
                    ),
                )
            ],
        )
        search = {**MILAN_MARCH, "limit": 2}
        script = {
            text: [
                Step(
                    delegated(found(search, "Milan", [SCENE, OTHER_SCENE])),
                    [
                        call("resolve_place", 1, name="Milan"),
                        call("search_scenes", 2, ids=[SCENE.id, OTHER_SCENE.id], **search),
                    ],
                )
            ]
        }
        run = Run(tmp_path, (task,), catalog=FakeCatalog([SCENE]))
        await run.go(script)
        assert run.catalog.requests[0].limit == 2
        (turn,) = run.attempts()[0].turns
        assert turn.oracle is not None
        assert (turn.oracle.precision, turn.oracle.recall) == (0.5, 1.0)

    async def test_a_turn_that_cites_nothing_has_no_precision_but_its_tools_have_recall(
        self, tmp_path: Path
    ) -> None:
        # A model failure: the tools found the scene, the answer cites nothing.
        text = "One Sentinel-2 scene of Milan from March 2025"
        task = Task(
            id="dev-one-milan",
            split="dev",
            category=Category.SINGLE_SEARCH,
            turns=[
                Turn(
                    user=text,
                    expect=Expectation(
                        route=Route.DELEGATE,
                        required_tools=["resolve_place", "search_scenes"],
                        search=ExpectedSearch(
                            product=Product.SENTINEL_2_L2A,
                            place="milan",
                            start=date(2025, 3, 1),
                            end=date(2025, 3, 31),
                            cloud=(0, 20),
                            limit=1,
                        ),
                    ),
                )
            ],
        )
        search = {**MILAN_MARCH, "limit": 1}
        result = found(search, "Milan", [SCENE]).model_copy(update={"failure": "step_limit"})
        calls = [
            call("resolve_place", 1, name="Milan"),
            call("search_scenes", 2, ids=[SCENE.id], **search),
        ]
        run = Run(tmp_path, (task,))
        await run.go({text: [Step(delegated(result, cited=[]), calls)]})
        attempt = run.attempts()[0]
        assert attempt.passed is False  # the model's failure fails the task
        (turn,) = attempt.turns
        assert turn.oracle is not None
        assert (turn.oracle.precision, turn.oracle.recall) == (None, 1.0)

    async def test_a_catalogue_that_returns_nothing_aborts_the_run(self, tmp_path: Path) -> None:
        # Every search would otherwise pass as "none found", against the oracle's own empty answer.
        script = passing_script()
        script[MILAN] = [
            Step(
                delegated(RetrievalResult(status="empty", search=the_search(MILAN_MARCH, "Milan"))),
                [call("resolve_place", 1, name="Milan"), call("search_scenes", 2, **MILAN_MARCH)],
            )
        ]
        script[NAPLES] = [
            Step(
                delegated(RetrievalResult(status="empty", search=the_search(NAPLES_FIRST_WEEK, "Naples"))),
                [call("resolve_place", 1, name="Naples"), call("search_scenes", 2, **NAPLES_FIRST_WEEK)],
            )
        ]
        run = Run(tmp_path, catalog=FakeCatalog([]))
        final = await run.go(script)
        assert final.status == "aborted"
        assert final.aborted_reason == (
            "the catalogue returned nothing for 2 tasks that expect a search, where the golden set has 1: "
            "has the catalogue or its client changed?"
        )

    async def test_the_one_task_without_results_is_no_alarm_even_when_tried_again(
        self, tmp_path: Path
    ) -> None:
        empty = Step(
            delegated(RetrievalResult(status="empty", search=the_search(MILAN_MARCH, "Milan"))),
            [call("resolve_place", 1, name="Milan"), call("search_scenes", 2, **MILAN_MARCH)],
        )
        script = passing_script()
        script[MILAN] = [empty]
        catalog = EmptyForMilan()
        run = Run(tmp_path, catalog=catalog)
        assert (await run.go(script)).status == "complete"

    async def test_a_catalogue_that_refuses_an_expected_search_aborts_the_run(self, tmp_path: Path) -> None:
        # The expected search is ours, so its refusal is a bug, not a metric that goes missing.
        run = Run(tmp_path, catalog=RefusingCatalog())
        final = await run.go(passing_script())
        assert final.status == "aborted"
        assert final.aborted_reason == "the catalogue refused an expected search: CatalogRefused: HTTP 400"

    async def test_a_query_that_fails_is_recorded_and_changes_no_score(self, tmp_path: Path) -> None:
        run = Run(tmp_path, catalog=FakeCatalog(fail=True))
        final = await run.go(passing_script())
        assert final.status == "complete"
        (turn,) = run.attempts()[0].turns
        assert turn.oracle is not None
        assert (
            turn.oracle.failed
            == "CatalogError: scene search failed: the catalogue did not answer (Read timed out)"
        )
        assert turn.oracle.ids == []
        assert turn.oracle.precision is None
        assert run.attempts()[0].passed

    async def test_an_attempt_stops_at_its_first_failed_turn(self, tmp_path: Path) -> None:
        # Turn 1 searches the wrong week, so the task has failed; turn 2 would be lost to the infrastructure,
        # which must not give the model another go at turn 1 through a retry.
        wrong_week = arguments("Naples", "sentinel-1-grd", "2025-09-08", "2025-09-14")
        script = passing_script()
        script[NAPLES] = [
            Step(
                delegated(found(wrong_week, "Naples", [NAPLES_SCENE])),
                [
                    call("resolve_place", 1, name="Naples"),
                    call("search_scenes", 2, ids=[NAPLES_SCENE.id], **wrong_week),
                ],
            )
        ]
        script[NAPLES_AGAIN] = [Step(UNREACHABLE)]
        run = Run(tmp_path)
        final = await run.go(script)
        assert final.status == "complete"
        assert final.tasks == TaskCounts(passed=3, failed=1)
        (attempt,) = [a for a in run.attempts() if a.task_id == "dev-follow-naples-weeks"]
        assert (attempt.outcome, attempt.passed) == ("scored", False)
        assert len(attempt.turns) == 1
        assert attempt.turns[0].score is not None
        assert not attempt.turns[0].score.passed
        assert run.orchestrator is not None
        assert [text for _, _, text in run.orchestrator.turns if text == NAPLES_AGAIN] == []

    async def test_a_query_that_does_not_answer_in_time_is_recorded(self, tmp_path: Path) -> None:
        run = Run(tmp_path, catalog=SlowCatalog(), config=settings(stac_timeout_s=0.01, stac_deadline_s=0.05))
        await run.go(passing_script())
        (turn,) = run.attempts()[0].turns
        assert turn.oracle is not None
        assert turn.oracle.failed == "the catalogue did not answer within 0.05 s"


class TestInfrastructure:
    async def test_a_task_lost_to_the_infrastructure_is_tried_again(self, tmp_path: Path) -> None:
        script = passing_script()
        script[MILAN] = [Step(UNREACHABLE), *script[MILAN]]
        run = Run(tmp_path)
        final = await run.go(script)
        assert final.status == "complete"
        assert [summary(a) for a in run.attempts()[:2]] == [
            ("dev-s2-milan-march", 1, "infrastructure", None),
            ("dev-s2-milan-march", 2, "scored", True),
        ]
        assert run.attempts()[0].reason == "turn 1: A2A ReadTimeout"
        assert run.attempts()[0].turns[0].score is None
        assert [conversation for conversation, _, text in run.turns() if text == MILAN] == [
            f"{RUN_ID}-t01-a1",
            f"{RUN_ID}-t01-a2",
        ]
        assert (
            run.lines[0]
            == "[1/4] dev-s2-milan-march attempt 1: infrastructure: turn 1: A2A ReadTimeout (0 s)"
        )

    async def test_the_agent_is_replaced_after_an_attempt_its_client_gave_up_on(self, tmp_path: Path) -> None:
        # It may still be at that turn, holding the model, while the retry starts.
        script = passing_script()
        script[MILAN] = [Step(UNREACHABLE), *script[MILAN]]
        run = Run(tmp_path)
        await run.go(script)
        assert run.servers.restarts == 1

    async def test_a_dead_agent_is_noticed_when_it_would_be_replaced(self, tmp_path: Path) -> None:
        script = passing_script()
        script[MILAN] = [Step(UNREACHABLE)]
        run = Run(tmp_path)
        run.servers.dead = "the retrieval agent exited with code 1 during the run"
        run.servers.dies_at_check = 2  # the check in restart_agent, right after the first attempt
        final = await run.go(script)
        assert (final.status, final.aborted_reason) == ("aborted", run.servers.dead)
        assert len(run.attempts()) == 1

    async def test_a_retry_waits_first_and_nothing_else_does(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pause = 12.5
        monkeypatch.setattr(run_eval, "RETRY_PAUSE_S", pause)
        real_sleep = asyncio.sleep
        pauses: list[float] = []

        async def sleep(delay: float, *args: Any) -> None:  # noqa: ANN401 - asyncio.sleep's own
            if delay == pause:
                pauses.append(delay)
                delay = 0
            await real_sleep(delay, *args)

        monkeypatch.setattr(asyncio, "sleep", sleep)
        script = passing_script()
        script[MILAN] = [Step(UNREACHABLE)]  # all three attempts lost: two pauses
        run = Run(tmp_path)
        await run.go(script)
        assert pauses == [pause, pause]

    async def test_the_third_task_lost_ends_the_run_in_a_row_or_not(self, tmp_path: Path) -> None:
        # Ollama or the catalogue is down: otherwise the run would say `complete` with little measured.
        script = passing_script()
        for text in (MILAN, GERMAN, WEATHER):
            script[text] = [Step(UNREACHABLE)]
        run = Run(tmp_path)
        final = await run.go(script)
        reason = "3 tasks were lost to the infrastructure; the last: turn 1: A2A ReadTimeout"
        assert (final.status, final.aborted_reason) == ("aborted", reason)
        assert final.tasks == TaskCounts(passed=1, lost=3)
        assert run.lines[-1] == f"run {RUN_ID}: 1 passed, 0 failed, 0 errored, 3 lost; aborted: {reason}"

    async def test_two_tasks_lost_leave_a_complete_run_that_says_so(self, tmp_path: Path) -> None:
        script = passing_script()
        for text in (MILAN, GERMAN):
            script[text] = [Step(UNREACHABLE)]
        run = Run(tmp_path)
        final = await run.go(script)
        assert final.status == "complete"
        assert final.tasks == TaskCounts(passed=2, lost=2)
        assert run.written().tasks == final.tasks
        assert run.lines[-1] == f"run {RUN_ID}: 2 passed, 0 failed, 0 errored, 2 lost"

    async def test_a_task_lost_three_times_is_excluded_and_the_run_goes_on(self, tmp_path: Path) -> None:
        script = passing_script()
        script[MILAN] = [Step(UNREACHABLE)]
        run = Run(tmp_path)
        final = await run.go(script)
        assert final.status == "complete"
        assert [summary(a) for a in run.attempts()] == [
            ("dev-s2-milan-march", 1, "infrastructure", None),
            ("dev-s2-milan-march", 2, "infrastructure", None),
            ("dev-s2-milan-march", 3, "infrastructure", None),
            ("dev-follow-naples-weeks", 1, "scored", True),
            ("dev-robust-german-milan", 1, "scored", True),
            ("dev-oos-weather-naples", 1, "scored", True),
        ]

    async def test_a_turn_past_the_harness_timeout_is_infrastructure_and_the_agent_is_replaced(
        self, tmp_path: Path
    ) -> None:
        script = passing_script()
        script[MILAN] = [Step(hangs=True), *script[MILAN]]
        run = Run(tmp_path, config=settings(**QUICK))
        final = await run.go(script)
        assert final.status == "complete"
        first = run.attempts()[0]
        assert summary(first) == ("dev-s2-milan-march", 1, "infrastructure", None)
        assert first.reason == "the turn ran past the harness timeout (0.3 s)"
        (turn,) = first.turns
        assert (turn.output, turn.error) == (None, first.reason)
        assert 0.3 <= turn.seconds < 3
        assert run.orchestrator is not None
        assert run.orchestrator.cancelled == [f"{RUN_ID}-t01-a1-u1"]
        assert run.servers.restarts == 1
        assert summary(run.attempts()[1]) == ("dev-s2-milan-march", 2, "scored", True)

    async def test_a_turn_that_swallows_its_cancellation_still_ends_on_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(run_eval, "CANCEL_GRACE_S", 0.05)
        script = passing_script()
        script[MILAN] = [Step(hangs=True, swallows_cancellation=True), *script[MILAN]]
        run = Run(tmp_path, config=settings(**QUICK))
        started = time.monotonic()
        final = await run.go(script)
        assert time.monotonic() - started < 5  # the turn itself would go on for 30 s
        assert final.status == "complete"
        assert run.attempts()[0].reason == "the turn ran past the harness timeout (0.3 s)"
        assert run.orchestrator is not None
        (abandoned,) = run.orchestrator.abandoned
        assert not abandoned.done()
        abandoned.cancel()
        await asyncio.wait({abandoned})

    async def test_the_time_of_a_turn_past_the_harness_timeout_leaves_out_the_cancel_grace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The grace is the runner's wait for the turn to end, not time the model spent.
        monkeypatch.setattr(run_eval, "CANCEL_GRACE_S", 2.0)
        script = passing_script()
        script[MILAN] = [Step(hangs=True, swallows_cancellation=True), *script[MILAN]]
        run = Run(tmp_path, config=settings(**QUICK))
        await run.go(script)
        (turn,) = run.attempts()[0].turns
        assert 0.3 <= turn.seconds < 1.0
        assert run.orchestrator is not None
        for abandoned in run.orchestrator.abandoned:
            abandoned.cancel()
        await asyncio.wait(set(run.orchestrator.abandoned))

    async def test_a_task_lost_on_its_last_attempt_is_counted_even_when_the_agent_cannot_be_replaced(
        self, tmp_path: Path
    ) -> None:
        # The agent is replaced after each of the task's first two attempts, and again after the task is
        # lost. That third replacement fails: the task ended before it, so it is in the count.
        script = passing_script()
        script[MILAN] = [Step(UNREACHABLE)]
        run = Run(tmp_path)
        replacements_that_work(run, 2)
        final = await run.go(script)
        assert (final.status, final.aborted_reason) == ("aborted", "something already answers on port 8002")
        assert final.tasks == TaskCounts(lost=1)
        assert run.written().tasks == final.tasks
        assert [a.outcome for a in run.attempts()] == ["infrastructure"] * 3

    async def test_the_third_task_lost_is_the_abort_reason_whatever_the_replacement_does(
        self, tmp_path: Path
    ) -> None:
        # Two replacements per lost task before the count; the third task's own is never asked for.
        script = passing_script()
        for text in (MILAN, GERMAN, WEATHER):
            script[text] = [Step(UNREACHABLE)]
        run = Run(tmp_path)
        replacements_that_work(run, 8)
        final = await run.go(script)
        assert final.tasks == TaskCounts(passed=1, lost=3)
        assert final.aborted_reason == (
            "3 tasks were lost to the infrastructure; the last: turn 1: A2A ReadTimeout"
        )
        assert run.servers.restarts == 8

    async def test_a_turn_past_the_harness_timeout_keeps_the_calls_it_made(self, tmp_path: Path) -> None:
        script = passing_script()
        script[MILAN] = [Step(hangs=True, calls=[call("resolve_place", 1, name="Milan")]), *script[MILAN]]
        run = Run(tmp_path, config=settings(**QUICK))
        await run.go(script)
        assert [c.tool for c in run.attempts()[0].turns[0].calls] == ["resolve_place"]


class TestErrored:
    async def test_a_turn_that_disagrees_with_the_log_is_errored_and_fails_the_task(
        self, tmp_path: Path
    ) -> None:
        script = passing_script()
        script[MILAN] = [
            Step(delegated(found(MILAN_MARCH, "Milan", [SCENE])), [])
        ]  # a search, but no logged call
        run = Run(tmp_path)
        final = await run.go(script)
        assert final.status == "complete"
        first = run.attempts()[0]
        assert summary(first) == ("dev-s2-milan-march", 1, "errored", False)
        assert (
            first.reason
            == "turn 1: the agent reports a search (ok), yet the log has no successful search_scenes"
        )
        assert first.turns[0].output is not None
        assert first.turns[0].score is None
        assert len(run.attempts()) == 4  # not tried again; the run goes on
        assert run.servers.restarts == 1  # the next task does not meet what the bug left behind

    async def test_the_third_errored_task_aborts_the_run(self, tmp_path: Path) -> None:
        script = passing_script()
        for text in (MILAN, NAPLES, GERMAN):
            script[text] = [Step(delegated(found(MILAN_MARCH, "Milan", [SCENE])), [])]
        run = Run(tmp_path)
        final = await run.go(script)
        assert (final.status, final.aborted_reason) == ("aborted", "3 tasks errored")
        assert [summary(a) for a in run.attempts()] == [
            ("dev-s2-milan-march", 1, "errored", False),
            ("dev-follow-naples-weeks", 1, "errored", False),
            ("dev-robust-german-milan", 1, "errored", False),
        ]
        assert WEATHER not in [text for _, _, text in run.turns()]
        assert run.written() == final
        assert run.servers.restarts == 2  # not after the third: the run ends there

    async def test_a_turn_that_raises_is_errored_and_the_attempt_stops_there(self, tmp_path: Path) -> None:
        script = passing_script()
        failed = RetrievalAgentError("the retrieval agent ended turn x as TASK_STATE_FAILED: RuntimeError")
        script[NAPLES_AGAIN] = [Step(raises=failed)]
        run = Run(tmp_path)
        await run.go(script)
        naples = run.attempts()[1]
        assert summary(naples) == ("dev-follow-naples-weeks", 1, "errored", False)
        assert naples.reason == f"turn 2 raised RetrievalAgentError: {failed}"
        first, second = naples.turns
        assert first.output is not None
        assert first.score is None
        assert first.oracle is None  # an attempt that is not scored counts for nothing
        assert (second.output, second.error) == (None, f"RetrievalAgentError: {failed}")

    async def test_a_turn_that_raises_keeps_the_calls_it_made(self, tmp_path: Path) -> None:
        script = passing_script()
        made = [call("resolve_place", 1, name="Milan")]
        script[MILAN] = [Step(calls=made, raises=ValueError("a bug"))]
        run = Run(tmp_path)
        await run.go(script)
        (turn,) = run.attempts()[0].turns
        assert turn.error == "ValueError: a bug"
        assert [c.tool for c in turn.calls] == ["resolve_place"]

    async def test_a_turn_that_ends_cancelled_by_itself_is_errored_not_an_interruption(
        self, tmp_path: Path
    ) -> None:
        script = passing_script()
        script[NAPLES] = [Step(raises=asyncio.CancelledError())]
        run = Run(tmp_path)
        final = await run.go(script)
        assert final.status == "complete"
        naples = run.attempts()[1]
        assert summary(naples) == ("dev-follow-naples-weeks", 1, "errored", False)
        assert naples.reason == (
            "turn 1 raised RuntimeError: the turn ended as cancelled, though nothing cancelled it"
        )


class TestAborted:
    async def test_a_server_found_dead_aborts_the_run_before_the_next_attempt(self, tmp_path: Path) -> None:
        run = Run(tmp_path)
        run.servers.dead = (
            "the retrieval agent was killed by SIGKILL during the run; its output is in retrieval_agent.log"
        )
        run.servers.dies_at_check = 2
        final = await run.go(passing_script())
        assert (final.status, final.aborted_reason) == ("aborted", run.servers.dead)
        assert [summary(a) for a in run.attempts()] == [("dev-s2-milan-march", 1, "scored", True)]
        assert run.written() == final

    async def test_a_replacement_agent_that_fails_aborts_the_run(self, tmp_path: Path) -> None:
        script = passing_script()
        script[MILAN] = [Step(hangs=True), *script[MILAN]]
        run = Run(tmp_path, config=settings(**QUICK))
        run.servers.restart_fails = "something already answers on port 8002"
        final = await run.go(script)
        assert (final.status, final.aborted_reason) == ("aborted", "something already answers on port 8002")
        assert [summary(a) for a in run.attempts()] == [("dev-s2-milan-march", 1, "infrastructure", None)]

    async def test_a_retrieval_agent_on_another_model_aborts_the_run(self, tmp_path: Path) -> None:
        script = passing_script()
        script[MILAN] = [
            Step(
                delegated(found(MILAN_MARCH, "Milan", [SCENE], model=OTHER_MODEL)),
                [
                    call("resolve_place", 1, name="Milan"),
                    call("search_scenes", 2, ids=[SCENE.id], **MILAN_MARCH),
                ],
            )
        ]
        run = Run(tmp_path)
        final = await run.go(script)
        assert (
            final.aborted_reason
            == "the retrieval agent runs 'ministral-3:8b', not the run's model 'qwen3.5:9b'"
        )
        assert run.attempts() == []

    async def test_a_result_without_a_model_call_is_not_a_mismatch(self, tmp_path: Path) -> None:
        # The A2A client's own placeholder, when the agent could not be reached, names no model.
        script = passing_script()
        script[MILAN] = [Step(UNREACHABLE), *script[MILAN]]
        run = Run(tmp_path)
        assert (await run.go(script)).status == "complete"

    async def test_an_output_filed_under_another_trace_id_is_a_bug_that_aborts_the_run(
        self, tmp_path: Path
    ) -> None:
        script = passing_script()
        step = script[MILAN][0]
        script[MILAN] = [Step(step.output, step.calls, wrong_trace_id=True)]
        run = Run(tmp_path)
        final = await run.go(script)
        assert final.status == "aborted"
        assert final.aborted_reason is not None
        assert final.aborted_reason.startswith(
            "a bug in the runner: turn unfiled was given a tool call with the trace id"
        )

    async def test_ctrl_c_during_a_turn_ends_the_turn_too(self, tmp_path: Path) -> None:
        # asyncio.run turns the first Ctrl+C into a cancellation of the run's task.
        script = passing_script()
        script[NAPLES] = [Step(hangs=True)]
        run = Run(tmp_path)
        running = asyncio.ensure_future(run.go(script))
        await asyncio.sleep(0.1)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        written = run.written()
        assert (written.status, written.aborted_reason) == ("aborted", "interrupted")
        assert written.finished_at is not None
        assert [summary(a) for a in run.attempts()] == [("dev-s2-milan-march", 1, "scored", True)]
        assert run.orchestrator is not None
        assert run.orchestrator.cancelled == [f"{RUN_ID}-t02-a1-u1"]

    async def test_an_error_of_the_runners_own_is_written_down_and_raised(self, tmp_path: Path) -> None:
        script = passing_script()
        step = script[MILAN][0]
        script[MILAN] = [Step(step.output, step.calls, corrupt_line=True)]
        run = Run(tmp_path)
        with pytest.raises(ValueError, match="not a tool-call record"):
            await run.go(script)
        written = run.written()
        assert written.status == "aborted"
        assert written.aborted_reason is not None
        assert written.aborted_reason.startswith("ValueError: ")
        assert "not a tool-call record" in written.aborted_reason

    async def test_a_record_that_cannot_be_written_does_not_hide_the_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def full_disk(folder: Path, record: RunRecord) -> None:
            raise OSError("No space left on device")

        monkeypatch.setattr(run_eval, "write_run_record", full_disk)
        script = passing_script()
        step = script[MILAN][0]
        script[MILAN] = [Step(step.output, step.calls, corrupt_line=True)]
        run = Run(tmp_path)
        with pytest.raises(ValueError, match="not a tool-call record"):
            await run.go(script)

    @pytest.mark.parametrize("dead_server", [None, "the MCP server exited with code 1 during the run"])
    async def test_a_final_record_that_cannot_be_written_is_said_with_the_counts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dead_server: str | None
    ) -> None:
        # The attempts are on disk and run.json still says `running`: the message says so, with what the
        # run came to, instead of a bare traceback.
        def full_disk(folder: Path, record: RunRecord) -> None:
            raise OSError("No space left on device")

        monkeypatch.setattr(run_eval, "write_run_record", full_disk)
        run = Run(tmp_path)
        if dead_server:
            run.servers.dead, run.servers.dies_at_check = dead_server, 2
        with pytest.raises(EvaluationError) as raised:
            await run.go(passing_script())
        said = str(raised.value)
        assert said.startswith(f"run {RUN_ID} ended ({'aborted' if dead_server else 'complete'}; ")
        assert ("1 passed" if dead_server else "4 passed") in said
        assert "could not be written: OSError: No space left on device" in said
        assert said.endswith(f"Its attempts are in {run.folder.name}; run.json still says running")
        assert len(run.attempts()) == (1 if dead_server else 4)

    async def test_a_record_written_before_a_failing_print_is_not_reported_as_unwritten(
        self, tmp_path: Path
    ) -> None:
        # The run's last line goes to a closed pipe (`... | head`): the record is on disk and says so.
        run = Run(tmp_path)

        def closed_pipe(line: str) -> None:
            if line.startswith("run "):
                raise BrokenPipeError(32, "Broken pipe")
            run.lines.append(line)

        run.say = closed_pipe
        with pytest.raises(BrokenPipeError):
            await run.go(passing_script())
        assert run.written().status == "complete"
        assert run.written().tasks == TaskCounts(passed=4)

    async def test_a_checkout_that_changes_is_noted_on_a_run_that_aborted_anyway(
        self, tmp_path: Path
    ) -> None:
        run = Run(tmp_path)
        run.servers.dead = "the MCP server exited with code 1 during the run"
        run.code = lambda: CodeVersion(commit=COMMIT.commit, dirty=True)
        final = await run.go(passing_script())
        assert final.aborted_reason == (
            "the MCP server exited with code 1 during the run; "
            "the checkout changed during the run: its record would not describe what ran"
        )

    async def test_a_checkout_that_changes_during_the_run_aborts_it(self, tmp_path: Path) -> None:
        # A replaced agent imports the checkout again, while the record names what was there at the start.
        run = Run(tmp_path)
        run.code = lambda: CodeVersion(commit=COMMIT.commit, dirty=True)
        final = await run.go(passing_script())
        assert final.status == "aborted"
        assert final.aborted_reason == (
            "the checkout changed during the run: its record would not describe what ran"
        )

    async def test_a_commit_that_cannot_be_read_at_the_end_aborts_the_run(self, tmp_path: Path) -> None:
        def unreadable() -> CodeVersion:
            raise ProvenanceError("git is not installed")

        run = Run(tmp_path)
        run.code = unreadable
        final = await run.go(passing_script())
        assert final.status == "aborted"
        assert final.aborted_reason == (
            "the commit could not be read at the end of the run: git is not installed"
        )


class TestTheLog:
    async def test_a_log_that_cannot_be_read_at_the_end_makes_the_run_aborted(self, tmp_path: Path) -> None:
        run = Run(tmp_path)

        def corrupt_at_the_last_check(number: int) -> None:
            if number == 5:
                with (run.folder / TOOL_LOG_FILE).open("a", encoding="utf-8") as log:
                    log.write('{"oops"\n')

        run.servers.on_check = corrupt_at_the_last_check
        final = await run.go(passing_script())
        assert final.status == "aborted"
        assert final.aborted_reason is not None
        assert final.aborted_reason.startswith("the tool-call log cannot be read: ValueError: ")
        assert run.written() == final
        assert len(run.attempts()) == 4

    async def test_calls_under_no_turn_of_the_run_are_counted_and_change_no_attempt(
        self, tmp_path: Path
    ) -> None:
        run = Run(tmp_path)
        run.log.write(call("resolve_place", 1, name="Milan").model_copy(update={"trace_id": "someone-else"}))
        run.log.write(call("resolve_place", 1, name="Milan").model_copy(update={"trace_id": None}))
        final = await run.go(passing_script())
        assert final.unattributed_calls == 2
        assert all(attempt.passed for attempt in run.attempts())

    async def test_every_golden_text_is_one_the_system_accepts(self) -> None:
        def accepted(text: str) -> bool:
            try:
                RetrievalRequest(text=text, today=REFERENCE_DATE)
            except ValidationError:
                return False
            return True

        texts = [turn.user for task in load_golden() for turn in task.turns]
        refused = sum(
            not accepted(text) for text in texts
        )  # a count: the test split's words stay out of the output
        assert refused == 0


class World:
    """The fakes behind `evaluate`: the servers of the latest run, and an orchestrator that writes its log."""

    def __init__(self, tmp_path: Path, script: dict[str, list[Step]], *models: str) -> None:
        self.root = tmp_path / "results"
        self.script = script
        self.model_server = FakeModelServer(*(models or MODELS_UNDER_EVALUATION))
        self.servers: list[FakeServers] = []
        self.orchestrators: list[ScriptedOrchestrator] = []
        self.dead: str | None = None
        self.agent_model: str | None = None  # the model the retrieval agent reports; by default the run's
        self.code = COMMIT
        self.uncommitted = False
        self.lines: list[str] = []

    def start_servers(self, config: Settings, folder: Path) -> FakeServers:
        made = FakeServers(folder)
        made.dead = self.dead
        self.servers.append(made)
        return made

    @asynccontextmanager
    async def orchestrator(self, config: Settings) -> AsyncIterator[ScriptedOrchestrator]:
        folder = self.servers[-1].folder
        assert folder is not None
        made = ScriptedOrchestrator(folder / TOOL_LOG_FILE, self.script)
        made.model = self.agent_model or config.llm_model
        self.orchestrators.append(made)
        yield made

    def services(self) -> Services:
        return Services(
            servers=self.start_servers,
            orchestrator=self.orchestrator,
            model_server=lambda config: self.model_server,
            catalog=lambda config: FakeCatalog(),
            code=lambda: self.code,
            uncommitted_runs=lambda: self.uncommitted,
        )

    async def evaluate(self, **options: Any) -> list[RunRecord]:  # noqa: ANN401 - evaluate's own options
        return await evaluate(
            settings(),
            root=self.root,
            services=self.services(),
            say=self.lines.append,
            **{"split": "dev", **options},
        )


def ids_shuffled(seed: int) -> list[str]:
    ids = [task.id for task in DEV_TASKS]
    random.Random(seed).shuffle(ids)  # noqa: S311 - the runner's own order, nothing secret
    return ids


class TestEvaluate:
    async def test_every_model_runs_the_given_number_of_times(self, tmp_path: Path) -> None:
        world = World(tmp_path, passing_script())
        records = await world.evaluate(models=[MODEL, OTHER_MODEL], runs=2)
        assert [(r.model.model, r.run_index, r.status) for r in records] == [
            (MODEL, 1, "complete"),
            (MODEL, 2, "complete"),
            (OTHER_MODEL, 1, "complete"),
            (OTHER_MODEL, 2, "complete"),
        ]
        assert len({r.run_id for r in records}) == 4
        for record in records:
            folder = world.root / "raw" / record.run_id
            assert read_run_record(folder) == record
            assert len(read_attempts(folder)) == 4
            assert (folder / TOOL_LOG_FILE).exists()
        assert all(made.stopped for made in world.servers)
        # Every run loads its model afresh and warms it up before its first task (decision of 2026-11-08).
        assert world.model_server.calls == [
            f"has {MODEL}",
            f"has {OTHER_MODEL}",
            f"prepare {MODEL} {OLLAMA_CONTEXT}",
            f"warm up {MODEL} {OLLAMA_CONTEXT}",
            f"prepare {MODEL} {OLLAMA_CONTEXT}",
            f"warm up {MODEL} {OLLAMA_CONTEXT}",
            f"unload {MODEL}",
            f"prepare {OTHER_MODEL} {OLLAMA_CONTEXT}",
            f"warm up {OTHER_MODEL} {OLLAMA_CONTEXT}",
            f"prepare {OTHER_MODEL} {OLLAMA_CONTEXT}",
            f"warm up {OTHER_MODEL} {OLLAMA_CONTEXT}",
            f"unload {OTHER_MODEL}",
            "closed",
        ]
        assert {record.warm_up_seconds for record in records} == {1.25}

    async def test_a_run_of_another_model_than_the_default_checks_against_its_own(
        self, tmp_path: Path
    ) -> None:
        # The settings name qwen3.5:9b; a run of the other model must expect the other model from its agent.
        world = World(tmp_path, passing_script())
        (record,) = await world.evaluate(models=[OTHER_MODEL], runs=1)
        assert (record.status, record.aborted_reason) == ("complete", None)

    async def test_an_agent_left_on_the_default_model_aborts_a_run_of_the_other(self, tmp_path: Path) -> None:
        world = World(tmp_path, passing_script())
        world.agent_model = MODEL
        (record,) = await world.evaluate(models=[OTHER_MODEL], runs=1)
        assert record.aborted_reason == (
            "the retrieval agent runs 'qwen3.5:9b', not the run's model 'ministral-3:8b'"
        )

    async def test_the_run_record_says_what_ran(self, tmp_path: Path) -> None:
        world = World(tmp_path, passing_script())
        (record,) = await world.evaluate(models=[MODEL], runs=1)
        assert record.model == ModelBuild(
            provider="ollama",
            model=MODEL,
            digest="d63f" * 16,
            server_version="0.40.1",
            options=chat_model_options(build_chat_model(settings(llm_model=MODEL))),
        )
        assert record.code == COMMIT
        assert record.hashes.tool_descriptions == A_HASH
        assert record.reference_date == REFERENCE_DATE
        assert (record.shuffle_seed, record.task_order) == (1, ids_shuffled(1))
        assert record.settings["harness_timeout_s"] == 480.0
        assert "python" in record.libraries
        assert record.split == "dev"

    async def test_the_task_order_follows_the_shuffle_seed(self, tmp_path: Path) -> None:
        world = World(tmp_path, passing_script())
        (record,) = await world.evaluate(models=[MODEL], runs=1, shuffle_seed=7)
        assert record.task_order == ids_shuffled(7)
        assert [a.task_id for a in read_attempts(world.root / "raw" / record.run_id)] == ids_shuffled(7)

    async def test_the_servers_get_the_runs_model(self, tmp_path: Path) -> None:
        world = World(tmp_path, passing_script())
        seen: list[str] = []

        def start(config: Settings, folder: Path) -> FakeServers:
            seen.append(config.llm_model)
            return world.start_servers(config, folder)

        await evaluate(
            settings(),
            split="dev",
            models=[OTHER_MODEL],
            runs=1,
            root=world.root,
            services=Services(**{**world.services().__dict__, "servers": start}),
            say=lambda line: None,
        )
        assert seen == [OTHER_MODEL]

    async def test_an_aborted_run_stops_the_evaluation(self, tmp_path: Path) -> None:
        world = World(tmp_path, passing_script())
        world.dead = "the MCP server exited with code 1 during the run; its output is in mcp_server.log"
        records = await world.evaluate(models=[MODEL, OTHER_MODEL], runs=2)
        assert [(r.run_index, r.status, r.aborted_reason) for r in records] == [(1, "aborted", world.dead)]
        assert world.model_server.calls[-2:] == [f"unload {MODEL}", "closed"]

    async def test_a_run_for_the_report_goes_under_runs_and_needs_a_clean_checkout(
        self, tmp_path: Path
    ) -> None:
        world = World(tmp_path, passing_script())
        (record,) = await world.evaluate(models=[MODEL], runs=1, for_report=True)
        assert (world.root / "runs" / record.run_id / "run.json").exists()
        world.code = CodeVersion(commit=COMMIT.commit, dirty=True)
        with pytest.raises(EvaluationError, match="the checkout has changes"):
            await world.evaluate(models=[MODEL], runs=1, for_report=True)
        world.uncommitted = False
        (scratch,) = await world.evaluate(models=[MODEL], runs=1)  # a scratch run may be dirty
        assert scratch.code.dirty
        assert (world.root / "raw" / scratch.run_id / "run.json").exists()

    @pytest.mark.parametrize(
        ("options", "said"),
        [
            pytest.param({"models": ["gpt-5"]}, "not under evaluation: gpt-5", id="another model"),
            pytest.param({"models": []}, "at least one model and one run", id="no model"),
            pytest.param({"models": [MODEL], "runs": 0}, "at least one model and one run", id="no run"),
            pytest.param({"models": [MODEL, MODEL]}, "a model is named twice", id="a model twice"),
            pytest.param(
                {"models": [MODEL], "split": "test"},
                "the test split runs only with --for-report",
                id="test split",
            ),
            pytest.param(
                {"models": [MODEL], "split": "test", "for_report": True, "runs": 1},
                "the test split is measured in 3 runs per model",
                id="test split, one run",
            ),
            pytest.param(
                {"models": [MODEL], "split": "test", "for_report": True, "shuffle_seed": 7},
                "in the order of seed 1",
                id="test split, another order",
            ),
        ],
    )
    async def test_what_cannot_be_evaluated_is_refused_before_anything_starts(
        self, tmp_path: Path, options: dict[str, Any], said: str
    ) -> None:
        world = World(tmp_path, passing_script())
        with pytest.raises(EvaluationError, match=said):
            await world.evaluate(**options)
        assert not world.root.exists()
        assert world.model_server.calls == []

    async def test_a_model_the_server_lacks_is_refused_before_any_run(self, tmp_path: Path) -> None:
        world = World(tmp_path, passing_script(), MODEL)
        with pytest.raises(EvaluationError, match=f"does not have {OTHER_MODEL}: pull it first"):
            await world.evaluate(models=[MODEL, OTHER_MODEL], runs=1)
        assert not world.root.exists()
        assert world.servers == []

    async def test_a_run_for_the_report_waits_until_the_earlier_ones_are_committed(
        self, tmp_path: Path
    ) -> None:
        # A run that is not on record could be dropped unseen: "report every run".
        world = World(tmp_path, passing_script())
        world.uncommitted = True
        with pytest.raises(EvaluationError, match="not committed: commit them first"):
            await world.evaluate(models=[MODEL], runs=1, for_report=True)
        assert world.servers == []
        (scratch,) = await world.evaluate(models=[MODEL], runs=1)
        assert scratch.status == "complete"

    @pytest.mark.parametrize(
        ("constant", "named"),
        [
            ("FROZEN_TEST_SPLIT", "changed since: the test split[.]"),
            ("FROZEN_EVALUATION_CODE", "changed since: the evaluation code[.]"),
            ("FROZEN_GAZETTEER", "changed since: the gazetteer[.]"),
            ("FROZEN_EVALUATION_RULES", "changed since: docs/evaluation.md[.]"),
        ],
    )
    async def test_a_run_for_the_report_needs_the_frozen_files_as_they_were_frozen(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, constant: str, named: str
    ) -> None:
        monkeypatch.setattr(run_eval, constant, "0" * 64)
        world = World(tmp_path, passing_script())
        with pytest.raises(EvaluationError, match=rf"frozen at tag golden-test-v2 and {named}"):
            await world.evaluate(models=[MODEL], runs=1, for_report=True)
        (scratch,) = await world.evaluate(models=[MODEL], runs=1)
        assert scratch.status == "complete"

    async def test_several_changed_frozen_files_are_all_named(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(run_eval, "FROZEN_GAZETTEER", "0" * 64)
        monkeypatch.setattr(run_eval, "FROZEN_EVALUATION_RULES", "0" * 64)
        world = World(tmp_path, passing_script())
        with pytest.raises(EvaluationError, match=r"changed since: the gazetteer, docs/evaluation.md[.]"):
            await world.evaluate(models=[MODEL], runs=1, for_report=True)

    async def test_another_ollama_version_is_named_on_the_screen(self, tmp_path: Path) -> None:
        world = World(tmp_path, passing_script())
        world.model_server.server_version = "0.41.0"
        (record,) = await world.evaluate(models=[MODEL], runs=1)
        assert record.model.server_version == "0.41.0"
        assert any(line.startswith("warning: Ollama 0.41.0:") for line in world.lines)

    async def test_a_run_never_writes_into_a_folder_that_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(run_eval, "new_run_id", lambda model, index, started: RUN_ID)
        world = World(tmp_path, passing_script())
        (world.root / "raw" / RUN_ID).mkdir(parents=True)
        with pytest.raises(EvaluationError, match="exists already"):
            await world.evaluate(models=[MODEL], runs=1)
        assert world.servers == []

    async def test_a_provider_that_is_not_there_yet_is_refused_in_words(self, tmp_path: Path) -> None:
        world = World(tmp_path, passing_script())
        with pytest.raises(
            EvaluationError, match="the Mistral provider arrives with the paid evaluation runs"
        ):
            await evaluate(
                settings(llm_provider="mistral"),
                split="dev",
                models=[MODEL],
                root=world.root,
                services=world.services(),
            )

    async def test_a_checkout_that_changes_between_runs_stops_the_evaluation(self, tmp_path: Path) -> None:
        world = World(tmp_path, passing_script())
        reads = iter([COMMIT, COMMIT, COMMIT, CodeVersion(commit="f" * 40, dirty=False)])
        services = Services(**{**world.services().__dict__, "code": lambda: next(reads)})
        with pytest.raises(EvaluationError, match="the checkout changed since this evaluation started"):
            await evaluate(
                settings(),
                split="dev",
                models=[MODEL],
                runs=2,
                root=world.root,
                services=services,
                say=lambda line: None,
            )
        assert len(world.servers) == 1

    async def test_a_model_that_cannot_be_unloaded_changes_nothing_about_the_runs(
        self, tmp_path: Path
    ) -> None:
        world = World(tmp_path, passing_script())
        world.model_server.unload_fails = "Ollama cannot be reached"
        records = await world.evaluate(models=[MODEL, OTHER_MODEL], runs=1)
        assert [r.status for r in records] == ["complete", "complete"]
        assert f"warning: {MODEL} may still be loaded: Ollama cannot be reached" in world.lines

    async def test_a_model_that_cannot_be_loaded_is_reported_with_the_servers_stopped(
        self, tmp_path: Path
    ) -> None:
        world = World(tmp_path, passing_script())
        world.model_server.prepare_fails = "Ollama still holds another model"
        with pytest.raises(ModelServerError, match="still holds another model"):
            await world.evaluate(models=[MODEL], runs=1)
        assert world.servers[0].stopped
        assert world.model_server.calls[-2:] == [f"unload {MODEL}", "closed"]

    async def test_a_warm_up_that_fails_stops_the_run_before_its_record_and_its_tasks(
        self, tmp_path: Path
    ) -> None:
        world = World(tmp_path, passing_script())
        world.model_server.warm_up_fails = "Ollama answered /api/chat with 500: runner stopped"
        with pytest.raises(ModelServerError, match="runner stopped"):
            await world.evaluate(models=[MODEL], runs=1)
        assert world.servers[0].stopped
        assert world.model_server.calls[-3:] == [
            f"warm up {MODEL} {OLLAMA_CONTEXT}",
            f"unload {MODEL}",
            "closed",
        ]
        (folder,) = (world.root / "raw").iterdir()  # the run's folder was made: its servers started there
        assert not (folder / "run.json").exists()  # no record says `running` for a run that never ran

    async def test_a_run_whose_clients_cannot_be_made_leaves_no_record_saying_running(
        self, tmp_path: Path
    ) -> None:
        world = World(tmp_path, passing_script())

        def broken(config: Settings) -> AbstractAsyncContextManager[ScriptedOrchestrator]:
            raise RuntimeError("a bug")

        services = Services(**{**world.services().__dict__, "orchestrator": broken})
        with pytest.raises(RuntimeError, match="a bug"):
            await evaluate(
                settings(),
                split="dev",
                models=[MODEL],
                runs=1,
                root=world.root,
                services=services,
                say=lambda line: None,
            )
        (folder,) = (world.root / "raw").iterdir()
        assert not (folder / "run.json").exists()

    async def test_a_split_without_tasks_is_refused(self, tmp_path: Path) -> None:
        golden = tmp_path / "golden"
        golden.mkdir()
        (golden / "dev.jsonl").write_text("", encoding="utf-8")
        (golden / "test.jsonl").write_text(
            DEV_TASKS[3].model_copy(update={"split": "test"}).model_dump_json() + "\n"
        )
        world = World(tmp_path, passing_script())
        services = Services(**{**world.services().__dict__, "golden_dir": golden})
        with pytest.raises(EvaluationError, match="the dev split has no tasks"):
            await evaluate(settings(), split="dev", models=[MODEL], root=world.root, services=services)


def test_a_closed_pipe_does_not_end_a_run(monkeypatch: pytest.MonkeyPatch) -> None:
    # `eo-agent eval ... | head`: the run takes hours, and its records are what matters.
    def closed(*args: object, **kwargs: object) -> None:
        raise BrokenPipeError

    monkeypatch.setattr("builtins.print", closed)
    run_eval._print("a line of progress")


class TestTheRealServices:
    async def test_they_are_built_from_the_settings_without_any_request(self) -> None:
        config = settings(ollama_url="http://127.0.0.1:11500", stac_api_url="https://catalogue.example/v1")
        assert Services().servers is RunServers
        assert isinstance(run_eval.live_catalog(config), StacCatalog)
        async with run_eval.live_model_server(config) as model_server:
            assert isinstance(model_server, OllamaServer)
            assert model_server.url == "http://127.0.0.1:11500"
        async with run_eval.live_orchestrator(config) as orchestrator:
            assert isinstance(orchestrator, Orchestrator)


def test_the_chat_model_options_are_read_from_the_model_itself() -> None:
    assert chat_model_options(build_chat_model(settings())) == {
        "num_ctx": OLLAMA_CONTEXT,
        "temperature": 0.0,
        "seed": 42,
        "reasoning": False,
        "num_predict": 1024,
        "keep_alive": "30m",
    }
