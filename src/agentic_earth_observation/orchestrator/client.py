"""The orchestrator's client of the retrieval agent, over A2A.

    async with A2ARetrievalClient(settings) as client:
        result = await client.retrieve(request, trace_id)

Each call sends one RetrievalRequest, with the turn's trace id, and returns the agent's RetrievalResult. How
each outcome comes back (decision of 2026-11-03):
- the agent completed the turn: its result, which may itself report infrastructure (`infra_error`);
- the agent cannot be reached, does not answer within EO_A2A_TIMEOUT_S, or its connection breaks: a result
  with `infra_error`, as `run_retrieval` reports its own infrastructure;
- the agent rejected the request or failed: RetrievalAgentError, because either is a bug and must never be
  excluded from the scores as infrastructure;
- anything else (an HTTP or protocol error, an answer that breaks the contract) is raised as it is.

A call is bounded by the HTTP timeout (EO_A2A_TIMEOUT_S). Over a real connection it can also be cancelled.
With the agent in the client's own process, as in the unit tests, the SDK turns a cancellation into a
cancelled task, which is a RetrievalAgentError here (decision of 2026-11-06, correcting 2026-11-03).
"""

from __future__ import annotations

from types import TracebackType
from typing import Protocol, Self
from uuid import uuid4

import httpx
from a2a.client import Client, ClientConfig, create_client
from a2a.types.a2a_pb2 import Message, Part, Role, SendMessageRequest, StreamResponse, TaskState
from pydantic import ValidationError

from agentic_earth_observation.catalog_mcp.tool_log import TRACE_ID, TRACE_ID_RULE
from agentic_earth_observation.config import Settings
from agentic_earth_observation.failures import NETWORK_FAILURES
from agentic_earth_observation.retrieval_agent.models import (
    A2A_MEDIA_TYPE,
    RESULT_ARTIFACT,
    TRACE_KEY,
    RetrievalRequest,
    RetrievalResult,
)


class RetrievalAgentError(RuntimeError):
    """The retrieval agent rejected the request, failed, or answered outside the contract: a bug."""


class RetrievalClient(Protocol):
    """How the orchestrator delegates one turn: the A2A client below, or a fake in tests."""

    async def retrieve(self, request: RetrievalRequest, trace_id: str) -> RetrievalResult: ...


class A2ARetrievalClient:
    """The retrieval agent at EO_A2A_URL. `http` lets tests reach an in-process server."""

    def __init__(self, settings: Settings, *, http: httpx.AsyncClient | None = None) -> None:
        self.url = settings.a2a_url
        # httpx's default timeout is 5 s; a turn may take up to EO_TURN_DEADLINE_S, which is shorter than this
        self.http = http or httpx.AsyncClient(timeout=httpx.Timeout(settings.a2a_timeout_s))
        self._client: Client | None = None  # made from the agent card on first use

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()  # closes `http` too
        else:
            await self.http.aclose()

    async def retrieve(self, request: RetrievalRequest, trace_id: str) -> RetrievalResult:
        if not TRACE_ID.fullmatch(trace_id):
            raise ValueError(f"invalid trace id {trace_id!r}: use {TRACE_ID_RULE}")
        sent = SendMessageRequest(
            message=Message(
                role=Role.ROLE_USER,
                message_id=str(uuid4()),
                parts=[Part(text=request.model_dump_json(), media_type=A2A_MEDIA_TYPE)],
            ),
            metadata={TRACE_KEY: trace_id},
        )
        try:
            client = await self._connected()
            answers = [answer async for answer in client.send_message(sent)]
        except Exception as err:
            if (failure := _network_failure(err)) is None:
                raise
            return RetrievalResult(status="not_searched", infra_error=_describe(failure))
        return _result(answers, trace_id)

    async def _connected(self) -> Client:
        """The SDK client, made from the agent card. A failed fetch is tried again on the next call."""
        if self._client is None:
            config = ClientConfig(httpx_client=self.http, streaming=False)
            self._client = await create_client(self.url, client_config=config)
        return self._client


def _network_failure(err: BaseException) -> BaseException | None:
    """The connection failure behind an error, if there is one: the SDK raises its own error with it as the
    cause."""
    for candidate in (err, err.__cause__):
        if isinstance(candidate, NETWORK_FAILURES):
            return candidate
    return None


def _describe(failure: BaseException) -> str:
    message = " ".join(str(failure).split())
    text = f"A2A {type(failure).__name__}: {message}" if message else f"A2A {type(failure).__name__}"
    return text[:300]


def _result(answers: list[StreamResponse], trace_id: str) -> RetrievalResult:
    """The result in the agent's answer, which must be one completed task with one JSON result."""
    if len(answers) != 1 or not answers[0].HasField("task"):
        raise RetrievalAgentError(f"the retrieval agent answered turn {trace_id} without a task")
    task = answers[0].task
    if task.status.state != TaskState.TASK_STATE_COMPLETED:
        said = " ".join(part.text for part in task.status.message.parts) or "no reason given"
        state = str(TaskState.Name(task.status.state))
        raise RetrievalAgentError(f"the retrieval agent ended turn {trace_id} as {state}: {said}"[:1000])
    parts = [part for artifact in task.artifacts for part in artifact.parts]
    if (
        [artifact.name for artifact in task.artifacts] != [RESULT_ARTIFACT]
        or len(parts) != 1
        or not parts[0].HasField("text")
        or parts[0].media_type != A2A_MEDIA_TYPE
    ):
        raise RetrievalAgentError(f"the retrieval agent's answer to turn {trace_id} is not one JSON result")
    try:
        return RetrievalResult.model_validate_json(parts[0].text)
    except ValidationError as err:
        raise RetrievalAgentError(
            f"the retrieval agent's answer to turn {trace_id} is not a valid RetrievalResult"
        ) from err
