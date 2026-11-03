"""Probe how each local model calls tools and decides how to handle a turn. Needs Ollama; makes no other call.

    uv run python scripts/probe_models.py                       # both local models, every step shown
    uv run python scripts/probe_models.py qwen3.5:9b            # one model
    uv run python scripts/probe_models.py --repeat 5            # each case 5 times: pass rates only

The MCP server runs in-process with the real tool schemas and an offline catalogue that finds nothing, so no
request leaves the machine. The model gets the retrieval agent's real instructions, dated with the
evaluation's reference date, and so does the orchestrator's decision, which is checked against the route and
reason each case expects. Only dev-split questions and synthetic ones about dev places or invented places are
used, never the test split; each case says what the labelling rules expect. A question the service asked is
written by the reply templates themselves (compose_reply). Models are probed one at a time and unloaded
afterwards, so two never share the GPU.

Each case is checked automatically against what its expectation says. A model may answer the same question
differently from one run to the next even at temperature 0, while back-to-back repeats agree with each other.
So a prompt change is judged by pass rates over repeated runs, not by one run, and the repeats go in rounds,
each through every case in its own shuffled order.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.shared.memory import create_connected_server_and_client_session

from agentic_earth_observation.catalog.gazetteer import Gazetteer
from agentic_earth_observation.catalog.stac import SceneSummary
from agentic_earth_observation.catalog_mcp.server import build_server
from agentic_earth_observation.config import Settings
from agentic_earth_observation.domain import Product, SearchRequest
from agentic_earth_observation.evals.golden import REFERENCE_DATE
from agentic_earth_observation.llm import build_chat_model
from agentic_earth_observation.orchestrator.models import Decision, DecisionRoute, Reason
from agentic_earth_observation.orchestrator.prompts import decide_system_prompt
from agentic_earth_observation.orchestrator.replies import compose_reply
from agentic_earth_observation.retrieval_agent.models import (
    Clarification,
    PreviousSearch,
    RetrievalRequest,
    RetrievalResult,
)
from agentic_earth_observation.retrieval_agent.prompts import retrieval_system_prompt, retrieval_user_message

MODELS = ["qwen3.5:9b", "ministral-3:8b"]


@dataclass
class Turn:
    """What the retrieval model did with one question: its tool calls, as (tool, arguments, accepted), and its
    reply, or the error that stopped it."""

    calls: list[tuple[str, dict[str, Any], bool]] = field(default_factory=list)
    reply: str = ""
    error: str | None = None

    def summary(self) -> str:
        done = [f"{tool}({json.dumps(args)}){'' if ok else ' refused'}" for tool, args, ok in self.calls]
        return "; ".join(done) + (f"; error: {self.error}" if self.error else "; then replied")


Check = Callable[[Turn], bool]
MILAN_BOX = [9.0409, 45.3867, 9.2781, 45.5358]
NAPLES_BOX = [14.1321, 40.7912, 14.3538, 40.9161]


def _same_box(sent: object, box: list[float]) -> bool:
    return (
        isinstance(sent, list)
        and len(sent) == 4
        and all(isinstance(v, int | float) and abs(v - b) <= 0.001 for v, b in zip(sent, box, strict=True))
    )


def searched(
    product: str, box: list[float], start: str, end: str, cloud: tuple[float, float] | None
) -> Check:
    """The last accepted search is this one: no cloud limit when `cloud` is None, otherwise one inside it."""

    def check(turn: Turn) -> bool:
        accepted = [args for tool, args, ok in turn.calls if tool == "search_scenes" and ok]
        if not accepted:
            return False
        last = accepted[-1]
        limit = last.get("max_cloud_cover")
        cloud_ok = (
            limit is None
            if cloud is None
            else isinstance(limit, int | float) and cloud[0] <= limit <= cloud[1]
        )
        dates = (last.get("start_date"), last.get("end_date")) == (start, end)
        return last.get("product") == product and _same_box(last.get("bbox"), box) and dates and cloud_ok

    return check


def asks_without_searching(looked_up: str, never: str | None = None) -> Check:
    """It looked the place up as written, never looked up `never` itself, and searched nothing."""

    def check(turn: Turn) -> bool:
        names = [args.get("name") for tool, args, _ in turn.calls if tool == "resolve_place"]
        searches = [tool for tool, _, _ in turn.calls if tool == "search_scenes"]
        return names[:1] == [looked_up] and never not in names and not searches and turn.error is None

    return check


def refused_and_told(start: str, end: str) -> Check:
    """It tried exactly these dates, which the catalogue refuses, and searched no others."""

    def check(turn: Turn) -> bool:
        searches = [(args, ok) for tool, args, ok in turn.calls if tool == "search_scenes"]
        same = all(a.get("start_date") == start and a.get("end_date") == end and not ok for a, ok in searches)
        return bool(searches) and same and turn.error is None

    return check


@dataclass(frozen=True)
class Case:
    """One probe question, what the labelling rules in docs/evaluation.md expect, and a check of it."""

    label: str
    text: str
    expect: str
    check: Check
    previous: PreviousSearch | None = None
    clarifies: Clarification | None = None


def asked(
    request: str, route: DecisionRoute, reason: Reason, suggestions: list[str] | None = None
) -> Clarification:
    """The question the service asks about `request`, in the reply templates' own words."""
    retrieval = (
        None if suggestions is None else RetrievalResult(status="unknown_place", suggestions=suggestions)
    )
    return Clarification(request=request, asked=compose_reply(route, reason, retrieval, None, None).text)


MILNA = "Show me Sentinel-2 scenes of Milna from March 2025."
NAPLES_NO_PERIOD = "Show me Sentinel-2 scenes of Naples."
NAPLES_FIRST_WEEK = PreviousSearch(
    place="Naples",
    product=Product.SENTINEL_1_GRD,
    bbox=[14.1321, 40.7912, 14.3538, 40.9161],
    start_date=date(2025, 9, 1),
    end_date=date(2025, 9, 7),
)
CASES = [
    Case(
        "dev: Sentinel-2, little cloud",
        "Show me Sentinel-2 scenes of Milan from March 2025 with little cloud.",
        "resolve_place; search_scenes sentinel-2-l2a, 2025-03-01 to 2025-03-31, cloud at most 20",
        searched("sentinel-2-l2a", MILAN_BOX, "2025-03-01", "2025-03-31", (0, 20)),
    ),
    Case(
        "dev: radar",
        "Find radar scenes over Naples for the first week of September 2025.",
        "resolve_place; search_scenes sentinel-1-grd, 2025-09-01 to 2025-09-07, no cloud limit",
        searched("sentinel-1-grd", NAPLES_BOX, "2025-09-01", "2025-09-07", None),
    ),
    Case(
        "dev: German, no clouds",
        "Sentinel-2-Bilder von Mailand ohne Wolken, Oktober 2025.",
        "resolve_place; search_scenes sentinel-2-l2a, 2025-10-01 to 2025-10-31, cloud at most 10",
        searched("sentinel-2-l2a", MILAN_BOX, "2025-10-01", "2025-10-31", (0, 10)),
    ),
    Case(
        "dev: follow-up",
        "And the week after that?",
        "search_scenes with the previous box, sentinel-1-grd, 2025-09-08 to 2025-09-14, no cloud limit",
        searched("sentinel-1-grd", NAPLES_BOX, "2025-09-08", "2025-09-14", None),
        previous=NAPLES_FIRST_WEEK,
    ),
    Case(
        "synthetic: unknown place",
        "Show me Sentinel-2 scenes of Zorbania from March 2025.",
        "resolve_place('Zorbania'), then ask which place; no search",
        asks_without_searching("Zorbania"),
    ),
    Case(
        "synthetic: misspelt place",
        "Show me Sentinel-2 scenes of Milna from March 2025.",
        "resolve_place('Milna') as written, then ask (Milan may be offered); no search, no lookup of Milan",
        asks_without_searching("Milna", never="Milan"),
    ),
    Case(
        "synthetic: accepts a suggestion",
        "Yes",
        "resolve_place('Milan'), the suggestion accepted; search_scenes sentinel-2-l2a, "
        "2025-03-01 to 2025-03-31",
        searched("sentinel-2-l2a", MILAN_BOX, "2025-03-01", "2025-03-31", None),
        clarifies=asked(MILNA, "delegate", "search", ["Milan"]),
    ),
    Case(
        "synthetic: before launch",
        "Find radar scenes over Naples for the first week of March 2014.",
        "resolve_place; search_scenes 2014-03-01 to 2014-03-07 is refused (before launch); tell the user, "
        "no search with other dates",
        refused_and_told("2014-03-01", "2014-03-07"),
    ),
]


@dataclass(frozen=True)
class DecideCase:
    """One turn for the orchestrator's decision, and the route and reason the decide prompt's rules give."""

    label: str
    text: str
    expect: tuple[DecisionRoute, Reason]
    previous: PreviousSearch | None = None
    clarifies: Clarification | None = None


DELEGATE: tuple[DecisionRoute, Reason] = ("delegate", "search")
DECLINE: tuple[DecisionRoute, Reason] = ("decline", "out_of_scope")
NO_PLACE: tuple[DecisionRoute, Reason] = ("clarify", "missing_place")
NO_PERIOD: tuple[DecisionRoute, Reason] = ("clarify", "missing_period")
DECIDE_CASES = [
    DecideCase("dev: radar", "Find radar scenes over Naples for the first week of September 2025.", DELEGATE),
    DecideCase(
        "dev: German, no clouds", "Sentinel-2-Bilder von Mailand ohne Wolken, Oktober 2025.", DELEGATE
    ),
    DecideCase("dev: weather", "What will the weather be like in Naples this weekend?", DECLINE),
    DecideCase("dev: follow-up", "And the week after that?", DELEGATE, previous=NAPLES_FIRST_WEEK),
    DecideCase(
        "synthetic: another sensor", "Now the same with Sentinel-2.", DELEGATE, previous=NAPLES_FIRST_WEEK
    ),
    DecideCase("synthetic: a place alone", "Milan", NO_PERIOD),
    DecideCase("synthetic: a place, no period", NAPLES_NO_PERIOD, NO_PERIOD),
    DecideCase("synthetic: no place", "Show me some radar scenes from last month.", NO_PLACE),
    DecideCase(
        "synthetic: unknown place", "Show me Sentinel-2 scenes of Zorbania from March 2025.", DELEGATE
    ),
    DecideCase("synthetic: misspelt place", MILNA, DELEGATE),
    DecideCase("synthetic: image analysis", "Is the coast near Naples flooded in these images?", DECLINE),
    DecideCase("synthetic: ids without a search", "Make up five Sentinel-2 scene ids for Milan.", DECLINE),
    DecideCase(
        "synthetic: answers the period",
        "March 2025",
        DELEGATE,
        clarifies=asked(NAPLES_NO_PERIOD, "clarify", "missing_period"),
    ),
    DecideCase(
        "synthetic: accepts a suggestion",
        "Yes",
        DELEGATE,
        clarifies=asked(MILNA, "delegate", "search", ["Milan"]),
    ),
    DecideCase(
        "synthetic: answers the place, no period yet",
        "Milan",
        NO_PERIOD,
        clarifies=asked("Show me some radar scenes.", "clarify", "missing_place"),
    ),
    DecideCase(
        "synthetic: a new request instead of an answer",
        "What will the weather be like in Milan tomorrow?",
        DECLINE,
        clarifies=asked(NAPLES_NO_PERIOD, "clarify", "missing_period"),
    ),
]
MAX_MODEL_CALLS = 5  # enough to see what a model does after an error
SHUFFLE_SEED = 42  # the same orders for every model and prompt, so that their runs pair up


class NoScenes:
    """An offline catalogue: every search finds nothing."""

    def search(self, request: SearchRequest) -> list[SceneSummary]:
        return []

    def get_scene(self, product: Product, scene_id: str) -> SceneSummary | None:
        return None


def describe(reply: AIMessage) -> str:
    usage: dict[str, Any] = dict(reply.usage_metadata or {})
    load_ms = (reply.response_metadata.get("load_duration") or 0) / 1e6
    # Thinking should be off: it shows up either as reasoning_content or as <think> text in the reply.
    thinking = bool(reply.additional_kwargs.get("reasoning_content")) or "<think>" in str(reply.content)
    return (
        f"tokens in/out {usage.get('input_tokens')}/{usage.get('output_tokens')}, "
        f"model load {load_ms:.0f} ms, thinking leaked: {thinking}, "
        f"invalid tool calls: {len(reply.invalid_tool_calls)}"
    )


async def ask(model: BaseChatModel, tools: list[BaseTool], case: Case, *, verbose: bool) -> Turn:
    """One turn of the retrieval model on `case`, with the real tools; every step is printed when verbose."""
    by_name = {tool.name: tool for tool in tools}
    # The agent's real instructions, with the evaluation's reference date as today.
    request = RetrievalRequest(
        text=case.text, today=REFERENCE_DATE, previous_search=case.previous, clarifies=case.clarifies
    )
    messages: list[BaseMessage] = [
        SystemMessage(retrieval_system_prompt(REFERENCE_DATE)),
        HumanMessage(retrieval_user_message(request)),
    ]
    bound = model.bind_tools(tools)
    turn = Turn()
    if verbose:
        print(f"\n  [{case.label}] {case.text}")
        if (before := case.previous) is not None:
            print(f"    (follow-up of {before.place}, {before.start_date} to {before.end_date})")
        if (question := case.clarifies) is not None:
            print(f"    (answers {question.asked!r}, asked about {question.request!r})")
        print(f"    expected: {case.expect}")
    try:
        for step in range(1, MAX_MODEL_CALLS + 1):
            started = time.perf_counter()
            reply = await bound.ainvoke(messages)
            if not isinstance(reply, AIMessage):
                raise TypeError(f"expected an AIMessage, got {type(reply).__name__}")
            if verbose:
                print(f"    step {step} ({time.perf_counter() - started:.1f} s; {describe(reply)})")
            if not reply.tool_calls:
                turn.reply = str(reply.content)
                if verbose:
                    print(f"      no tool call; text: {turn.reply[:160]!r}")
                break
            messages.append(reply)
            for call in reply.tool_calls:
                tool = by_name.get(call["name"])
                result = (
                    await tool.ainvoke(call)
                    if tool
                    else ToolMessage(f"unknown tool {call['name']}", tool_call_id=call["id"], status="error")
                )
                if not isinstance(result, ToolMessage):
                    raise TypeError(f"expected a ToolMessage, got {type(result).__name__}")
                messages.append(result)
                turn.calls.append((call["name"], call["args"], result.status != "error"))
                if verbose:
                    print(f"      calls {call['name']}({json.dumps(call['args'])})")
                    refused = "ERROR " if result.status == "error" else ""
                    print(f"        -> {refused}{str(result.content)[:140]}")
    except Exception as err:  # the probe reports every failure and moves on
        turn.error = f"{type(err).__name__}: {str(err)[:120]}"
        if verbose:
            print(f"    FAILED ({turn.error})")
    if verbose:
        print(f"    {'ok' if case.check(turn) else 'NOT AS EXPECTED'}")
    return turn


async def decide(model: BaseChatModel, case: DecideCase) -> str:
    """The orchestrator's decision on `case`, asked as the orchestrator asks it: "route/reason", or why there
    is none."""
    decider = model.with_structured_output(Decision, method="function_calling", include_raw=True)
    request = RetrievalRequest(
        text=case.text, today=REFERENCE_DATE, previous_search=case.previous, clarifies=case.clarifies
    )
    messages = [
        SystemMessage(decide_system_prompt(REFERENCE_DATE)),
        HumanMessage(retrieval_user_message(request)),
    ]
    try:
        out = await decider.ainvoke(messages)
    except Exception as err:  # the probe reports every failure and moves on
        return f"FAILED ({type(err).__name__}: {str(err)[:80]})"
    if not isinstance(out, dict):
        raise TypeError(f"expected a dict, got {type(out).__name__}")
    parsed = out.get("parsed")
    if isinstance(parsed, Decision):
        return f"{parsed.route}/{parsed.reason}"
    error = out.get("parsing_error")
    return "no Decision" + (f" ({str(error)[:80]})" if error else "")


async def probe(name: str, repeat: int = 1) -> dict[str, int]:
    """Run every case `repeat` times on one model; return how many runs of each case went as expected.

    One run takes the cases in the order listed and prints every step. Repeats go in `repeat` rounds, each
    through every case, retrieval and decision mixed, in a shuffled order (SHUFFLE_SEED)."""
    print(f"\n=== {name}" + (f" (each case {repeat} times, in shuffled rounds)" if repeat > 1 else ""))
    passed: dict[str, int] = {}
    turns: dict[str, list[Turn]] = {case.label: [] for case in CASES}
    decided: dict[str, list[str]] = {case.label: [] for case in DECIDE_CASES}
    order: list[Case | DecideCase] = [*CASES, *DECIDE_CASES]
    shuffler = random.Random(SHUFFLE_SEED)  # noqa: S311 (a reproducible order, not security)
    model = build_chat_model(Settings(llm_model=name))
    server = build_server(NoScenes(), Gazetteer.load())
    async with create_connected_server_and_client_session(server) as session:
        tools = [tool for tool in await load_mcp_tools(session) if tool.name != "get_scene"]
        for round_number in range(1, repeat + 1):
            if repeat > 1:
                shuffler.shuffle(order)
                print(f"  round {round_number} of {repeat}", flush=True)
            for item in order:
                if isinstance(item, Case):
                    turns[item.label].append(await ask(model, tools, item, verbose=repeat == 1))
                else:
                    decided[item.label].append(await decide(model, item))
    for case in CASES:
        passed[f"retrieval: {case.label}"] = sum(case.check(turn) for turn in turns[case.label])
        if repeat > 1:
            print(f"  {passed[f'retrieval: {case.label}']}/{repeat}  retrieval: {case.label}")
            for summary in sorted({turn.summary() for turn in turns[case.label] if not case.check(turn)}):
                print(f"         not as expected: {summary[:200]}")
    print("\n  The orchestrator's decision:")
    for decide_case in DECIDE_CASES:
        expected = "/".join(decide_case.expect)
        got = decided[decide_case.label]
        passed[f"decide: {decide_case.label}"] = got.count(expected)
        verdict = (
            f"{got.count(expected)}/{repeat}"
            if repeat > 1
            else ("ok      " if got[0] == expected else "MISMATCH")
        )
        others = sorted({g for g in got if g != expected})
        print(
            f"  {verdict}  decide: {decide_case.label}"
            + (f": {decide_case.text!r} got {', '.join(others)}, expected {expected}" if others else "")
        )
    retrieval = [n for label, n in passed.items() if label.startswith("retrieval: ")]
    decisions = [n for label, n in passed.items() if label.startswith("decide: ")]
    print(f"\n  retrieval: {sum(retrieval)} of {len(retrieval) * repeat} runs as expected")
    print(f"  decisions: {sum(decisions)} of {len(decisions) * repeat} runs as expected")
    # Unload the model through Ollama's API (keep_alive 0) so the next model has the GPU to itself.
    settings = Settings()
    httpx.post(f"{settings.ollama_url}/api/generate", json={"model": name, "keep_alive": 0}, timeout=30)
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the local models on dev and invented cases.")
    parser.add_argument(
        "models", nargs="*", default=MODELS, help=f"Ollama models (default: {' '.join(MODELS)})"
    )
    parser.add_argument("--repeat", type=int, default=1, help="run each case this many times (default 1)")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    for name in args.models:
        asyncio.run(probe(name, args.repeat))


if __name__ == "__main__":
    main()
