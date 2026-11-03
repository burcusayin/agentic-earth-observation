"""The `eo-agent` command.

    make serve-mcp           # terminal 1: the catalogue MCP server
    make serve-agent         # terminal 2: the retrieval agent over A2A (Ollama must be running)
    uv run eo-agent chat     # terminal 3: talk to the assistant

Each line you type is one turn of the same conversation, so a follow-up continues the previous search. After
each answer a status line shows how the turn was handled, with its trace id, which files the retrieval agent's
tool calls in the MCP server's log. "quit", "exit" or Ctrl+D ends the chat.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import Callable, Sequence
from datetime import date
from uuid import uuid4

from langchain_core.language_models import BaseChatModel

from agentic_earth_observation.config import Settings
from agentic_earth_observation.llm import build_chat_model
from agentic_earth_observation.orchestrator.client import (
    A2ARetrievalClient,
    RetrievalAgentError,
    RetrievalClient,
)
from agentic_earth_observation.orchestrator.graph import Orchestrator
from agentic_earth_observation.orchestrator.models import TurnOutput

QUIT = {"quit", "exit"}


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
    args = parser.parse_args(argv)
    settings = settings or Settings()
    today: date = args.today or settings.today()
    asyncio.run(_chat_with_the_running_servers(settings, today))
    return 0


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
    orchestrator = Orchestrator(model, client, settings.llm_model)
    conversation = f"chat-{uuid4().hex[:8]}"
    say(f"Earth Observation assistant ({settings.llm_model}; today is {today}; agent at {settings.a2a_url}).")
    say('Ask for satellite scenes of a place and a period. "quit" or Ctrl+D ends the chat.')
    turn = 0
    while True:
        try:
            text = ask("> ").strip()
        except EOFError:
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
        except ValueError as err:  # a line the assistant cannot take, such as one far too long
            say(f"  ({err})")
            continue
        except RetrievalAgentError as err:  # a bug in the retrieval agent: shown, and its log has the details
            say(f"  (the retrieval agent failed: {err})")
            continue
        say(out.answer)
        say(f"  [{status_line(out)} · {time.perf_counter() - started:.1f} s]")


def status_line(out: TurnOutput) -> str:
    """How the turn was handled, for the person checking the system."""
    parts = [f"route {out.route}", f"status {out.status}", f"trace {out.trace_id}"]
    if out.failure is not None:
        detail = f": {out.failure_detail}" if out.failure_detail else ""
        parts.append(f"model failure {out.failure}{detail}")
    if out.infra_error is not None:
        parts.append(f"infrastructure: {out.infra_error}")
    return " · ".join(parts)
