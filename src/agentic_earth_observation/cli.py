"""The `eo-agent` command.

    make serve-mcp           # terminal 1: the catalogue MCP server
    make serve-agent         # terminal 2: the retrieval agent over A2A (Ollama must be running)
    uv run eo-agent chat     # terminal 3: talk to the assistant

    uv run eo-agent eval --split dev --models qwen3.5:9b --runs 1   # the evaluation (starts its own servers)

Each line you type in the chat is one turn of the same conversation, so a follow-up continues the previous
search. After each answer a status line shows how the turn was handled, with its trace id, which files the
retrieval agent's tool calls in the MCP server's log. "quit", "exit" or Ctrl+D ends the chat.

The evaluation runs every task of a split through the system, each model the given number of times, and
writes one folder per run under results/ (evals/run_eval.py). Ollama must be running; nothing else.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
from collections.abc import Callable, Coroutine, Sequence
from datetime import date
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel

from agentic_earth_observation.config import ROOT, Settings
from agentic_earth_observation.evals.golden import SPLITS, GoldenError, golden_hash, load_golden
from agentic_earth_observation.evals.model_server import ModelServerError
from agentic_earth_observation.evals.records import OUTPUT_FOLDER, REPORTED_RUNS, ProvenanceError, RunRecord
from agentic_earth_observation.evals.report import ReportError, build_report, collect_runs, render_markdown
from agentic_earth_observation.evals.run_eval import (
    DEFAULT_RUNS,
    DEFAULT_SHUFFLE_SEED,
    MODELS_UNDER_EVALUATION,
    EvaluationError,
    evaluate,
)
from agentic_earth_observation.evals.servers import ServerError
from agentic_earth_observation.llm import build_chat_model
from agentic_earth_observation.orchestrator.client import (
    A2ARetrievalClient,
    RetrievalAgentError,
    RetrievalClient,
)
from agentic_earth_observation.orchestrator.graph import Orchestrator, TurnRefused
from agentic_earth_observation.orchestrator.models import TurnOutput

QUIT = {"quit", "exit"}
INTERRUPTED = 130  # the exit code of a command stopped by Ctrl+C
CANNOT_EVALUATE = (EvaluationError, GoldenError, ProvenanceError, ServerError, ModelServerError)


def main(argv: Sequence[str] | None = None, *, settings: Settings | None = None) -> int:
    """Run the command. `settings` lets tests leave the developer's `.env` out."""
    parser = argparse.ArgumentParser(prog="eo-agent", description="The Earth Observation assistant.")
    commands = parser.add_subparsers(dest="command", required=True)
    chat_parser = commands.add_parser(
        "chat", help="talk to the assistant (the MCP server and the retrieval agent must be running)"
    )
    chat_parser.add_argument(
        "--today", type=date.fromisoformat, help="the date the agents take as today (YYYY-MM-DD)"
    )
    eval_parser = commands.add_parser(
        "eval", help="run the evaluation over a split of the golden set (starts the servers itself)"
    )
    eval_parser.add_argument("--split", choices=SPLITS, required=True, help="which split of the golden set")
    eval_parser.add_argument(
        "--models",
        default=",".join(MODELS_UNDER_EVALUATION),
        help=f"the models to run, separated by commas (default: {','.join(MODELS_UNDER_EVALUATION)})",
    )
    eval_parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="runs per model (default: 3)")
    eval_parser.add_argument(
        "--for-report",
        action="store_true",
        help="write under results/runs for the report (needs a clean checkout) instead of results/raw",
    )
    eval_parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=DEFAULT_SHUFFLE_SEED,
        help="the seed of the task order (default: 1)",
    )
    report_parser = commands.add_parser("report", help="the report on a split's runs, as markdown")
    report_parser.add_argument("--split", choices=SPLITS, required=True, help="which split of the golden set")
    report_parser.add_argument(
        "--runs-dir",
        type=Path,
        default=ROOT / OUTPUT_FOLDER / REPORTED_RUNS,
        help=f"the folder of the runs' folders (default: {OUTPUT_FOLDER}/{REPORTED_RUNS})",
    )
    report_parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="count runs from a checkout with changes, for scratch use (the report then says so)",
    )
    report_parser.add_argument(
        "--output", type=Path, help="write the report to this file instead of printing it"
    )
    args = parser.parse_args(argv)
    if args.command == "report":  # it reads records only: no settings, no server, no model
        return _report(args)
    settings = settings or Settings()
    if args.command == "eval":
        return _evaluate(settings, args)
    today: date = args.today or settings.today()
    try:
        asyncio.run(_chat_with_the_running_servers(settings, today))
    except KeyboardInterrupt:  # during a turn; at the prompt it simply ends the chat
        return INTERRUPTED
    return 0


def _report(args: argparse.Namespace) -> int:
    """Write the report on a split's runs. 0 when it was written, also when no run counts yet."""
    try:
        tasks = [task for task in load_golden() if task.split == args.split]
        listed = collect_runs(
            args.runs_dir,
            args.split,
            [task.id for task in tasks],
            golden=golden_hash(),
            allow_dirty=args.allow_dirty,
        )
        text = render_markdown(build_report(args.split, listed, tasks))
    except (ReportError, GoldenError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    if args.output is None:
        sys.stdout.write(text)
        return 0
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    except OSError as err:
        print(f"error: the report cannot be written to {args.output}: {err.strerror}", file=sys.stderr)
        return 1
    print(f"report written to {args.output}")
    return 0


def _evaluate(settings: Settings, args: argparse.Namespace) -> int:
    """Run the evaluation and say how each run ended. 0 when every run completed."""
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    try:
        records = asyncio.run(
            _until_stopped(
                evaluate(
                    settings,
                    split=args.split,
                    models=models,
                    runs=args.runs,
                    for_report=args.for_report,
                    shuffle_seed=args.shuffle_seed,
                )
            )
        )
    except CANNOT_EVALUATE as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("interrupted: what each run finished is kept in its folder under results/", file=sys.stderr)
        return INTERRUPTED
    for record in records:
        print(run_line(record))
    return 0 if all(record.status == "complete" for record in records) else 1


async def _until_stopped[T](main: Coroutine[Any, Any, T]) -> T:
    """Run `main`, which SIGTERM and SIGHUP (a `kill`, a closed terminal) end as Ctrl+C does: asyncio handles
    only Ctrl+C by itself, and otherwise the run's servers would stay behind and its record say `running`.

    A signal being ignored stays ignored: that is what `nohup` asks for, for a run that takes all night.
    """
    task = asyncio.current_task()
    assert task is not None  # noqa: S101 - this is a coroutine, so it runs in a task
    loop = asyncio.get_running_loop()
    for stop in (signal.SIGTERM, signal.SIGHUP):
        if signal.getsignal(stop) != signal.SIG_IGN:
            loop.add_signal_handler(stop, task.cancel)
    return await main


def run_line(record: RunRecord) -> str:
    ended = record.status if record.aborted_reason is None else f"aborted: {record.aborted_reason}"
    tasks = record.tasks
    counts = f"{tasks.passed} passed, {tasks.failed} failed, {tasks.errored} errored, {tasks.lost} lost"
    return f"{record.run_id}: {ended} ({counts})"


async def _chat_with_the_running_servers(settings: Settings, today: date) -> None:
    async with A2ARetrievalClient(settings) as client:
        await chat(build_chat_model(settings), client, settings, today, ask=input, say=print)


async def chat(
    model: BaseChatModel,
    client: RetrievalClient,
    settings: Settings,
    today: date,
    *,
    ask: Callable[[str], str],
    say: Callable[[str], None],
) -> None:
    """The conversation: `ask` reads a line (raising EOFError at the end), `say` shows a line."""
    orchestrator = Orchestrator(
        model, client, settings.llm_model, decision_deadline_s=settings.decision_deadline_s
    )
    conversation = f"chat-{uuid4().hex[:8]}"
    say(f"Earth Observation assistant ({settings.llm_model}; today is {today}; agent at {settings.a2a_url}).")
    say('Ask for satellite scenes of a place and a period. "quit" or Ctrl+D ends the chat.')
    turn = 0
    while True:
        try:
            text = _read_line(ask).strip()
        except (EOFError, KeyboardInterrupt):
            say("")
            return
        if text.lower() in QUIT:
            return
        if not text:
            continue
        turn += 1
        trace_id = f"{conversation}-{turn}"
        started = time.perf_counter()
        try:
            out = await orchestrator.run_turn(conversation, text, trace_id, today)
        except TurnRefused as err:  # a line the assistant cannot take, such as one far too long
            say(f"  ({err})")
            continue
        except RetrievalAgentError as err:  # a bug in the retrieval agent: shown, and its log has the details
            say(f"  (the retrieval agent failed: {err})")
            continue
        say(out.answer)
        say(f"  [{status_line(out)} · {time.perf_counter() - started:.1f} s]")


def _read_line(ask: Callable[[str], str]) -> str:
    """The next line, with Ctrl+C raising KeyboardInterrupt while waiting for it.

    `input` blocks the event loop, so asyncio's own handler could only note the Ctrl+C, and the chat would
    stay until the next line was typed and then end with a traceback.
    """
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        return ask("> ")
    finally:
        signal.signal(signal.SIGINT, previous)


def status_line(out: TurnOutput) -> str:
    """How the turn was handled, for the person checking the system."""
    parts = [f"route {out.route}", f"status {out.status}", f"trace {out.trace_id}"]
    if out.failure is not None:
        detail = f": {out.failure_detail}" if out.failure_detail else ""
        parts.append(f"model failure {out.failure}{detail}")
    if out.infra_error is not None:
        parts.append(f"infrastructure: {out.infra_error}")
    return " · ".join(parts)
