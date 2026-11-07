"""Which failures are the infrastructure's: one rule for the retrieval agent and the orchestrator.

Infrastructure is a failure outside the model's control and outside our code: a timeout, a connection that
cannot be made or breaks, the MCP client giving up waiting or losing its connection, and a busy model server.
It is excluded from the scores, so anything else must be raised as a bug or a configuration error, never
counted here (decisions of 2026-11-02 and 2026-11-03).
"""

from __future__ import annotations

import httpx
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED

from agentic_earth_observation.llm import model_server_failed

# A connection that times out, cannot be made, or breaks. HTTP errors are not here: a server's answer is not a
# failure of the network, and our own servers answering with an error have a bug or a wrong setting.
NETWORK_FAILURES = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)
# The MCP client's error code when a request gets no answer in time.
MCP_TIMED_OUT = httpx.codes.REQUEST_TIMEOUT


def is_infrastructure(err: BaseException) -> bool:
    """Whether an error is a failure outside the model's control and outside our code.

    An HTTP error from our own servers, a wrong MCP URL (which the MCP client reports as "Session terminated")
    or a model the model server does not have is a bug or a configuration error, so it is not.
    """
    if isinstance(err, (TimeoutError, ConnectionError, *NETWORK_FAILURES)):
        return True
    if isinstance(err, McpError):
        return err.error.code in (MCP_TIMED_OUT, CONNECTION_CLOSED)
    return model_server_failed(err)


def leaves(err: BaseException) -> list[BaseException]:
    """The errors inside nested exception groups, or the error itself."""
    if isinstance(err, BaseExceptionGroup):
        return [leaf for inner in err.exceptions for leaf in leaves(inner)]
    return [err]


def describe(err: BaseException) -> str:
    """The error's type and message on one line, for a result; some errors have no message."""
    message = " ".join(str(err).split())
    return f"{type(err).__name__}: {message}"[:300] if message else type(err).__name__
