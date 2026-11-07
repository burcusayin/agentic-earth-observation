"""A2A server for the retrieval agent: one A2A task per delegated turn, on 127.0.0.1 only.

    make serve-agent        # http://127.0.0.1:8002; the MCP server and Ollama must be running

The orchestrator sends a RetrievalRequest as JSON text, with the turn's trace id in the request metadata, and
gets back a task whose artifact is the RetrievalResult as JSON text. JSON text, not a data part: a data part
carries every number as a double, so `"limit": 3` would arrive as 3.0 and the strict models refuse it.

Each task ends in one of three states (decision of 2026-11-03):
- completed: `run_retrieval` returned a result, an infrastructure failure included (`infra_error`);
- rejected: the request could not be read (no valid trace id, not one JSON part, an invalid RetrievalRequest);
- failed: a bug or a configuration error. The client sees only the error's type; the details go to the log,
  because the SDK would otherwise send the error's own text to the client.

Requests with a Host other than 127.0.0.1 or localhost are refused (400), and so is any request that carries
an Origin header (403): a web page must not be able to start an agent run.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib.metadata import version
from typing import Any

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types.a2a_pb2 import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Message,
    Part,
    Task,
    TaskState,
    TaskStatus,
)
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT, TransportProtocol
from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from agentic_earth_observation.catalog_mcp.tool_log import TRACE_ID, TRACE_ID_RULE
from agentic_earth_observation.config import LOCAL_HOSTS, Settings, local_port
from agentic_earth_observation.llm import build_chat_model
from agentic_earth_observation.retrieval_agent.agent import run_retrieval
from agentic_earth_observation.retrieval_agent.models import (
    A2A_MEDIA_TYPE,
    RESULT_ARTIFACT,
    TRACE_KEY,
    RetrievalRequest,
    RetrievalResult,
)

logger = logging.getLogger(__name__)

JSON = A2A_MEDIA_TYPE
RPC_PATH = "/a2a"
Retrieve = Callable[[RetrievalRequest, str, BaseChatModel, Settings], Awaitable[RetrievalResult]]


class RetrievalExecutor(AgentExecutor):
    """Runs one retrieval turn for each A2A task. `run` is `run_retrieval`, or a stand-in in tests."""

    def __init__(self, model: BaseChatModel, settings: Settings, run: Retrieve = run_retrieval) -> None:
        self.model = model
        self.settings = settings
        self.run = run

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id, context_id = context.task_id or "", context.context_id or ""
        # The SDK refuses any status update before the task itself; with it, every answer is a task.
        submitted = TaskStatus(state=TaskState.TASK_STATE_SUBMITTED)
        await event_queue.enqueue_event(Task(id=task_id, context_id=context_id, status=submitted))
        updater = TaskUpdater(event_queue, task_id, context_id)
        try:
            trace_id = read_trace_id(context.metadata)
            request = read_request(context.message)
        except ValueError as err:
            await updater.reject(updater.new_agent_message([Part(text=str(err))]))
            return
        try:
            result = await self.run(request, trace_id, self.model, self.settings)
        except Exception as err:
            logger.exception("the retrieval turn %s failed", trace_id)
            reason = f"the retrieval agent failed ({type(err).__name__}); its log has the details"
            await updater.failed(updater.new_agent_message([Part(text=reason)]))
            return
        await updater.add_artifact(
            [Part(text=result.model_dump_json(), media_type=JSON)], name=RESULT_ARTIFACT
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # The SDK calls this while the event queue is still open, then cancels `execute`; it closes the task
        # as cancelled itself if this did not. The orchestrator never cancels a turn.
        await TaskUpdater(event_queue, context.task_id or "", context.context_id or "").cancel()


def read_trace_id(metadata: dict[str, Any]) -> str:
    """The turn's trace id from the request metadata, which the client sets; refused unless it is valid."""
    value = metadata.get(TRACE_KEY)
    if not isinstance(value, str) or not TRACE_ID.fullmatch(value):
        raise ValueError(f"the request metadata needs a {TRACE_KEY!r}: {TRACE_ID_RULE}")
    return value


def read_request(message: Message | None) -> RetrievalRequest:
    """The RetrievalRequest in the message: exactly one text part, as JSON."""
    if message is None or len(message.parts) != 1:
        raise ValueError("send exactly one part: the RetrievalRequest as JSON text")
    part = message.parts[0]
    if not part.HasField("text") or part.media_type != JSON:
        raise ValueError(f"send the RetrievalRequest as a text part with media type {JSON}")
    try:
        return RetrievalRequest.model_validate_json(part.text)
    except ValidationError as err:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'request'}: {e['msg']}" for e in err.errors())
        raise ValueError(f"invalid RetrievalRequest: {problems}"[:1000]) from None


def agent_card(settings: Settings) -> AgentCard:
    """What the agent offers, served at /.well-known/agent-card.json."""
    return AgentCard(
        name="Satellite scene retrieval agent",
        description=(
            "Finds Sentinel-1 and Sentinel-2 scenes for a place and a period in a public STAC catalogue, "
            "through MCP tools. Scene metadata only: no images."
        ),
        version=version("agentic-earth-observation"),
        supported_interfaces=[
            AgentInterface(
                url=settings.a2a_url.rstrip("/") + RPC_PATH,
                protocol_binding=TransportProtocol.JSONRPC.value,
                protocol_version=PROTOCOL_VERSION_CURRENT,
            )
        ],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=[JSON],
        default_output_modes=[JSON],
        skills=[
            AgentSkill(
                id="find_scenes",
                name="Find scenes",
                description=(
                    "One search turn. Send a RetrievalRequest as JSON text with the trace id in the request "
                    f"metadata ({TRACE_KEY!r}); the completed task's artifact is a RetrievalResult as JSON "
                    "text."
                ),
                tags=["earth-observation", "sentinel", "stac"],
                input_modes=[JSON],
                output_modes=[JSON],
            )
        ],
    )


class RejectOrigin:
    """Refuses any request that carries an Origin header: browsers send one, our own clients do not."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and any(name == b"origin" for name, _ in scope["headers"]):
            await PlainTextResponse("requests from web pages are not accepted", status_code=403)(
                scope, receive, send
            )
            return
        await self.app(scope, receive, send)


def build_app(executor: AgentExecutor, settings: Settings) -> Starlette:
    """The A2A app: the agent card and the JSON-RPC endpoint, behind the Host and Origin checks."""
    card = agent_card(settings)
    handler = DefaultRequestHandlerV2(
        agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=card
    )

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        yield
        await handler.aclose()  # so that no SDK task outlives the server

    return Starlette(
        routes=[*create_agent_card_routes(card), *create_jsonrpc_routes(handler, rpc_url=RPC_PATH)],
        middleware=[
            Middleware(TrustedHostMiddleware, allowed_hosts=list(LOCAL_HOSTS)),
            Middleware(RejectOrigin),
        ],
        lifespan=lifespan,
    )


def main() -> None:
    """Serve the retrieval agent over A2A on 127.0.0.1, at the port in EO_A2A_URL."""
    settings = Settings()
    port = local_port(settings.a2a_url, "EO_A2A_URL")
    executor = RetrievalExecutor(build_chat_model(settings), settings)
    uvicorn.run(build_app(executor, settings), host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
