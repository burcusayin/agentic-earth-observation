"""What one turn of an evaluation run counts as, and whether a task passed. Pure functions only.

After a turn the runner holds two accounts of it: the orchestrator's output (a TurnOutput) and the MCP
server's own records of the turn's tool calls. The first rule that applies decides what the turn makes of its
attempt (docs/decisions.md, 2026-11-02, 2026-11-05 and 2026-11-06):

1. the turn raised an error: `errored`. This one is the runner's, as there is no output to judge;
2. the output reports a failure of the infrastructure: `infrastructure`;
3. the log has a call that the catalogue failed: `infrastructure`, although the agent reports only `error`;
4. the output and the log disagree: `errored`, a bug in our own system, never scored;
5. otherwise the turn is scored, on the trace that `turn_trace` builds.

Infrastructure is decided before the two accounts are compared: after such a failure the agent's result is a
placeholder that reports none of the calls the agent made.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from agentic_earth_observation.catalog_mcp.tool_log import ToolCallRecord
from agentic_earth_observation.evals.records import AttemptOutcome
from agentic_earth_observation.evals.scoring import ToolCall, TurnScore, TurnTrace
from agentic_earth_observation.orchestrator.models import TurnOutput
from agentic_earth_observation.retrieval_agent.models import PreviousSearch

SEARCH = "search_scenes"
PLACE_LOOKUP = "resolve_place"


def turn_outcome(output: TurnOutput, records: Sequence[ToolCallRecord]) -> tuple[AttemptOutcome, str | None]:
    """What a turn that returned makes of its attempt, and why: rules 2 to 5, the first that applies.

    `records` are the server's records of this turn's calls, in the order the calls were sent, as
    `read_tool_calls` gives them. "scored" comes without a reason, the other two always with one.
    """
    _of_this_turn(output, records)
    if (reason := _infrastructure(output, records)) is not None:
        return "infrastructure", reason
    if (reason := _disagreement(output, records)) is not None:
        return "errored", reason
    return "scored", None


def turn_trace(output: TurnOutput, records: Sequence[ToolCallRecord]) -> TurnTrace:
    """The turn as the scoring reads it.

    The route, the status and the cited ids are the orchestrator's. The calls are the server's records only,
    never what an agent reports, in the order they were sent and with the refused ones included.
    """
    _of_this_turn(output, records)
    calls = [
        ToolCall(name=record.tool, arguments=record.arguments, ok=record.ok, returned_ids=record.returned_ids)
        for record in records
    ]
    return TurnTrace(route=output.route, calls=calls, status=output.status, cited_ids=output.cited_ids)


def task_passed(scores: Sequence[TurnScore], outputs: Sequence[TurnOutput]) -> bool:
    """Whether a scored attempt passed its task: every turn's score passes and no turn has a model failure.

    A model failure (running out of steps, a malformed tool call, output that cannot be parsed) fails the
    task whatever the trace scores (decision of 2026-11-02). Pass every turn of the task, in order.
    """
    if not scores or len(scores) != len(outputs):
        raise ValueError(
            f"a task is judged on all of its turns: pass one score and one output per turn, "
            f"not {len(scores)} and {len(outputs)}"
        )
    if any(output.infra_error is not None for output in outputs):
        raise ValueError("a turn lost to the infrastructure is tried again, never scored: do not judge it")
    return all(score.passed for score in scores) and all(output.failure is None for output in outputs)


def _of_this_turn(output: TurnOutput, records: Sequence[ToolCallRecord]) -> None:
    """Refuse records filed under another trace id: the caller read another turn's calls."""
    for record in records:
        if record.trace_id != output.trace_id:
            raise ValueError(
                f"turn {output.trace_id} was given a tool call with the trace id {record.trace_id!r}: "
                "read its calls with read_tool_calls(log, the turn's trace id)"
            )


def _infrastructure(output: TurnOutput, records: Sequence[ToolCallRecord]) -> str | None:
    """Why the turn is lost to the infrastructure, if it is (rules 2 and 3)."""
    if output.infra_error is not None:
        return output.infra_error or "the infrastructure failed, and the turn does not say how"
    for record in records:
        if record.error_kind == "catalogue":
            return f"the catalogue failed or did not answer in time, in a {record.tool} call"
    return None


def _disagreement(output: TurnOutput, records: Sequence[ToolCallRecord]) -> str | None:
    """How the orchestrator's output contradicts the server's records of the turn, if it does (rule 4).

    Checked only for a turn that reports no infrastructure failure. A refused call (`invalid_arguments`) is
    the model's doing and never a disagreement.
    """
    result = output.retrieval
    if result is None:
        if records:
            return f"the turn was not delegated, yet the log has {len(records)} tool call(s) for it"
        return None
    for record in records:
        if record.error_kind == "other":
            return f"a {record.tool} call failed and the server cannot say why (error kind 'other')"
        if record.error_kind == "cancelled":
            return f"a {record.tool} call was cancelled, yet the turn reports no infrastructure failure"
    searches = [record for record in records if record.tool == SEARCH and record.ok]
    # A result carries a search exactly when its status is `ok` or `empty`: RetrievalResult checks it.
    if result.search is not None:
        if not searches:
            return f"the agent reports a search ({result.status}), yet the log has no successful {SEARCH}"
        last = searches[-1]
        if [scene.id for scene in result.scenes] != last.returned_ids:
            return f"the agent's scenes are not the ones the log's last successful {SEARCH} returned"
        if not _same_search(result.search, last.arguments):
            return f"the agent's search is not the log's last successful {SEARCH}: the arguments differ"
        return None
    if searches:
        return f"the agent reports {result.status}, yet the log has a successful {SEARCH}"
    if result.status == "unknown_place" and not any(r.tool == PLACE_LOOKUP and r.ok for r in records):
        return f"the agent reports an unknown place, yet the log has no answered {PLACE_LOOKUP}"
    if result.status == "error" and all(record.ok for record in records):
        return "the agent reports a tool error, yet the log has no failed call"
    return None


def _same_search(search: PreviousSearch, arguments: dict[str, Any]) -> bool:
    """Whether the agent's search has the logged call's arguments.

    The logged arguments are read as the agent reads its own (`result_from_turn`), so 20 and 20.0 are the same
    cloud limit. The place name is the agent's own addition, which no call carries, so it is not compared.
    """
    try:
        logged = PreviousSearch.model_validate_json(json.dumps({**arguments, "place": search.place}))
    except (TypeError, ValueError):  # not a search at all; pydantic's ValidationError is a ValueError
        return False
    return logged == search
