"""Ask the retrieval agent one question from the command line: a smoke test of the whole chain.

    make serve-mcp                                                 # in one terminal; Ollama must be running
    uv run python -m agentic_earth_observation.retrieval_agent "<a question>"

It prints the turn's trace id, the result, and the tool calls that the MCP server recorded under that trace
id. Settings come from EO_* variables or `.env`, as for the server; "today" is EO_REFERENCE_DATE or the real
date. The exit status is 1 when the turn failed for a reason outside the model's control.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from uuid import uuid4

from pydantic import ValidationError

from agentic_earth_observation.catalog_mcp.tool_log import read_tool_calls
from agentic_earth_observation.config import Settings
from agentic_earth_observation.llm import build_chat_model
from agentic_earth_observation.retrieval_agent.agent import run_retrieval
from agentic_earth_observation.retrieval_agent.models import RetrievalRequest


def main(argv: Sequence[str] | None = None, *, settings: Settings | None = None) -> int:
    """Run one turn and print what happened. `settings` lets tests leave the developer's `.env` out."""
    parser = argparse.ArgumentParser(
        prog="python -m agentic_earth_observation.retrieval_agent",
        description="Ask the retrieval agent one question, through the running MCP server and the model.",
    )
    parser.add_argument("question", help="the request, in the user's own words")
    args = parser.parse_args(argv)
    settings = settings or Settings()
    try:
        request = RetrievalRequest(text=args.question, today=settings.today())
    except ValidationError as err:
        parser.error(f"the question cannot be sent: {err.errors()[0]['msg']}")
    trace_id = f"smoke-{uuid4().hex[:12]}"
    result = asyncio.run(run_retrieval(request, trace_id, build_chat_model(settings), settings))
    print(f"trace id: {trace_id}")
    print(result.model_dump_json(indent=2))
    _print_tool_calls(settings, trace_id)
    return 0 if result.infra_error is None else 1


def _print_tool_calls(settings: Settings, trace_id: str) -> None:
    if settings.tool_log is None:  # only possible from code: the environment cannot turn the log off
        print("the tool-call log is off")
        return
    try:
        calls = read_tool_calls(settings.tool_log, trace_id)
    except FileNotFoundError:
        print(f"no tool-call log at {settings.tool_log}: does the MCP server use the same EO_TOOL_LOG?")
        return
    print(f"tool calls recorded under this trace id in {settings.tool_log}:")
    for number, call in enumerate(calls, start=1):
        if not call.ok:
            outcome = f"failed ({call.error_kind})"
        elif call.tool == "search_scenes":  # the only agent tool that returns scenes
            outcome = f"ok, {len(call.returned_ids)} scene id(s)"
        else:
            outcome = "ok"
        print(f"  {number}. {call.tool} {json.dumps(call.arguments)}: {outcome}")
    if not calls:
        print("  none")


if __name__ == "__main__":
    raise SystemExit(main())
