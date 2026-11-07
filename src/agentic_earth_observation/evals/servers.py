"""The MCP server and the retrieval agent of one evaluation run, started, watched and stopped by the runner.

    async with RunServers(settings, folder) as servers:
        servers.tool_descriptions       # for the run record
        servers.check()                 # before every attempt, and before the run is called complete
        await servers.restart_agent()   # after an attempt lost to the infrastructure, or a bug

Every run gets its own pair (decision of 2026-11-05): the retrieval agent's model and the file of the
tool-call log are fixed when the servers start. Both are child processes of the runner, started from this
checkout, with the run's settings stated in their environment and their output in the run's folder.

What running the real servers showed, and the rule that follows from each (decisions of 2026-11-06 and -07):
- a new server looks ready while an older one answers on its port, and exits a moment later. So nothing may
  answer on a port before a server is started there, and a server is ready only while its process runs;
- a server that has died looks like a failure of the infrastructure to every turn after it. So `check` tells
  the runner, which ends the run instead of trying each remaining task three times;
- the retrieval agent goes on with a turn its client gave up on, and does not stop on SIGTERM while a turn is
  in flight. So a stop waits a few seconds and then kills, and `restart_agent` gives the next attempt a new
  agent after any attempt that was not scored.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import TracebackType
from typing import Self

import httpx
from a2a.client import A2ACardResolver
from langchain_mcp_adapters.sessions import StreamableHttpConnection, create_session
from mcp.shared._httpx_utils import McpHttpClientFactory
from mcp.types import Tool
from pydantic import SecretStr

from agentic_earth_observation.catalog_mcp import server as catalog_server
from agentic_earth_observation.catalog_mcp.server import tool_descriptions_hash
from agentic_earth_observation.config import ENV_PREFIX, ROOT, Settings, is_local_url, local_port
from agentic_earth_observation.evals.records import TOOL_LOG_FILE
from agentic_earth_observation.failures import describe, leaves
from agentic_earth_observation.retrieval_agent import a2a_server
from agentic_earth_observation.retrieval_agent.agent import local_mcp_client

MCP = "MCP server"
AGENT = "retrieval agent"
MCP_LOG = "mcp_server.log"
AGENT_LOG = "retrieval_agent.log"
MCP_COMMAND = (sys.executable, "-m", catalog_server.__name__)
AGENT_COMMAND = (sys.executable, "-m", a2a_server.__name__)
# Both servers were ready within a second when measured. The limit is for the first start after a reboot, or a
# machine short of memory, where failing would cost the rest of a long evaluation.
READY_LIMIT_S = 60.0
# An idle server stops in a fraction of a second. The agent with a turn in flight does not stop at all.
STOP_GRACE_S = 5.0
CHECK_TIMEOUT_S = 5.0  # one readiness check
POLL_S = 0.05


class ServerError(Exception):
    """The run's servers cannot be started, or are not running. The message says what to do."""


class ServerDied(ServerError):
    """A server the run started has exited: the run cannot go on. The message says which, and how it ended."""


def port_open(port: int) -> bool:
    """Whether something accepts connections on `port` of this machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(CHECK_TIMEOUT_S)
        return probe.connect_ex((catalog_server.HOST, port)) == 0


async def served_tools(settings: Settings, *, http: McpHttpClientFactory | None = None) -> list[Tool]:
    """The tools of the MCP server at EO_MCP_URL, asked for as the retrieval agent asks when a turn starts.

    `http` lets tests reach an in-process server.
    """
    connection: StreamableHttpConnection = {
        "transport": "streamable_http",
        "url": settings.mcp_url,
        "timeout": CHECK_TIMEOUT_S,
        "sse_read_timeout": CHECK_TIMEOUT_S,
        "session_kwargs": {"read_timeout_seconds": timedelta(seconds=CHECK_TIMEOUT_S)},
        "httpx_client_factory": http or (local_mcp_client if is_local_url(settings.mcp_url) else None),
    }
    async with create_session(connection) as session:
        await session.initialize()
        return list((await session.list_tools()).tools)


async def agent_answers(settings: Settings, *, http: httpx.AsyncClient | None = None) -> None:
    """Read the retrieval agent's card at EO_A2A_URL, as the orchestrator's client does before its first turn.

    Raises if there is no valid card. `http` lets tests reach an in-process server; it is closed afterwards.
    """
    local = is_local_url(settings.a2a_url)
    async with http or httpx.AsyncClient(timeout=CHECK_TIMEOUT_S, trust_env=not local) as client:
        await A2ACardResolver(client, settings.a2a_url).get_agent_card()


@dataclass(frozen=True)
class Probe:
    """How the run looks at its servers over the network. Tests replace it: they open no socket."""

    port_open: Callable[[int], bool] = port_open
    served_tools: Callable[[Settings], Awaitable[list[Tool]]] = served_tools
    agent_answers: Callable[[Settings], Awaitable[None]] = agent_answers


def server_environment(
    settings: Settings, tool_log: Path, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The environment of a server the run starts: the runner's own (`base`), with the run's settings stated.

    Every setting that has a value becomes its EO_* variable, and a variable beats `.env`, so both servers run
    with the runner's settings whatever the shell or `.env` says. `tool_log` replaces the configured log file.
    A secret is never copied out of the settings: its variable stays as the runner's environment has it, and a
    key kept in `.env` is read there by the server, as the runner read it.
    """
    environment = dict(os.environ if base is None else base)
    values = settings.model_dump() | {"tool_log": tool_log}
    for name, value in values.items():
        if value is not None and not isinstance(value, SecretStr):
            environment[f"{ENV_PREFIX}{name}".upper()] = str(value)
    return environment


@dataclass(frozen=True)
class _Server:
    name: str  # as the messages call it
    log: str  # the file in the run's folder that holds its output
    process: subprocess.Popen[bytes]

    def ended(self) -> str | None:
        """How the server's process ended, or None while it runs."""
        code = self.process.poll()
        if code is None:
            return None
        if code < 0:  # ended by a signal
            try:
                return f"was killed by {signal.Signals(-code).name}"
            except ValueError:
                return f"was killed by signal {-code}"
        return f"exited with code {code}"

    def output(self) -> str:
        return f"output is in {self.log} in the run's folder"


class RunServers:
    """The two servers of one run, whose files are in `folder`.

    The keyword arguments are for tests: stand-in commands, a probe that opens no socket, and shorter waits.
    """

    def __init__(
        self,
        settings: Settings,
        folder: Path,
        *,
        probe: Probe | None = None,
        mcp_command: Sequence[str] = MCP_COMMAND,
        agent_command: Sequence[str] = AGENT_COMMAND,
        ready_limit_s: float = READY_LIMIT_S,
        stop_grace_s: float = STOP_GRACE_S,
    ) -> None:
        self.settings = settings
        self.folder = folder.resolve()  # the servers are given the log's full path
        self.tool_log = self.folder / TOOL_LOG_FILE
        self._probe = probe or Probe()
        self._commands = {MCP: list(mcp_command), AGENT: list(agent_command)}
        self._ready_limit_s = ready_limit_s
        self._stop_grace_s = stop_grace_s
        self._mcp: _Server | None = None
        self._agent: _Server | None = None
        self._tool_descriptions: str | None = None

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None
    ) -> None:
        self.stop()

    @property
    def tool_descriptions(self) -> str:
        """SHA-256 of the tools as the run's MCP server describes them to the retrieval agent."""
        if self._tool_descriptions is None:
            raise ServerError("the servers have not been started: the tools are read from the running server")
        return self._tool_descriptions

    async def start(self) -> None:
        """Start the MCP server, then the retrieval agent, and return once both are ready.

        Raises ServerError, with nothing left running, when a port already answers, the run's folder holds an
        earlier run's tool-call log, or a server exits or does not get ready in time.
        """
        if self._mcp is not None or self._agent is not None:
            raise ServerError("the run's servers are already started")
        mcp_port, agent_port = _local_ports(self.settings)
        if self.tool_log.exists():
            raise ServerError(
                f"the folder {self.folder.name} already holds a tool-call log ({TOOL_LOG_FILE}): "
                "every run gets its own folder, so that its log holds its own calls only"
            )
        self._refuse_if_answering(MCP, mcp_port)
        self._refuse_if_answering(AGENT, agent_port)
        self.folder.mkdir(parents=True, exist_ok=True)
        try:
            self._mcp = self._spawn(MCP, MCP_LOG)
            tools = await self._until_ready(
                self._mcp, mcp_port, lambda: self._probe.served_tools(self.settings)
            )
            self._tool_descriptions = tool_descriptions_hash(tools)
            await self._start_agent(agent_port)
        except BaseException:  # Ctrl+C and a cancelled run included: leave nothing running
            self.stop()
            raise

    def check(self) -> None:
        """Raise ServerDied if one of the run's servers has exited."""
        for server in self._running():
            if (ended := server.ended()) is not None:
                raise ServerDied(f"the {server.name} {ended} during the run; its {server.output()}")

    async def restart_agent(self) -> None:
        """Replace the retrieval agent by a new one. The MCP server, and with it the tool-call log, goes on.

        A server found dead is not replaced: that is ServerDied, as in `check`.
        """
        self.check()
        _, agent = self._running()
        # In a thread: the wait may take the whole grace period, and the run's event loop has work of its own.
        await asyncio.to_thread(_stop, agent.process, self._stop_grace_s)
        self._agent = None
        await self._start_agent(_local_ports(self.settings)[1])

    def stop(self) -> None:
        """Stop both servers, the agent first: SIGTERM, then SIGKILL for one that still runs after the grace
        period. Harmless when nothing runs."""
        servers = [server for server in (self._agent, self._mcp) if server is not None]
        self._agent = self._mcp = None
        try:
            for server in servers:
                _stop(server.process, self._stop_grace_s)
        except BaseException:  # a second Ctrl+C during the wait: do not wait, and leave nothing running
            for server in servers:
                server.process.kill()
            raise

    def _running(self) -> tuple[_Server, _Server]:
        if self._mcp is None or self._agent is None:
            raise ServerError(
                "the run's servers are not running: start them with `async with RunServers(...)`"
            )
        return self._mcp, self._agent

    async def _start_agent(self, port: int) -> None:
        self._refuse_if_answering(AGENT, port)
        self._agent = self._spawn(AGENT, AGENT_LOG)
        await self._until_ready(self._agent, port, lambda: self._probe.agent_answers(self.settings))

    def _refuse_if_answering(self, name: str, port: int) -> None:
        if self._probe.port_open(port):
            raise ServerError(
                f"something already answers on port {port}, where the run starts its {name}: stop it first "
                "(a `make serve-mcp` or `make serve-agent` left running, or a server of a killed run)"
            )

    def _spawn(self, name: str, log: str) -> _Server:
        """Start a server's process. Its output is added to its log file, which a restart continues."""
        with (self.folder / log).open("ab") as output:
            process = subprocess.Popen(  # noqa: S603 - our own module, run by this Python, without a shell
                self._commands[name],
                cwd=ROOT,
                env=server_environment(self.settings, self.tool_log),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                # Its own session: Ctrl+C in the terminal reaches the runner alone, which then stops both.
                start_new_session=True,
            )
        return _Server(name, log, process)

    async def _until_ready[T](self, server: _Server, port: int, check: Callable[[], Awaitable[T]]) -> T:
        """Wait until `check` passes on the server's port, and return its answer."""
        give_up = time.monotonic() + self._ready_limit_s
        problem = "its port did not open"
        while True:
            if (ended := server.ended()) is not None:
                raise ServerError(f"the {server.name} {ended} before it was ready; its {server.output()}")
            if self._probe.port_open(port):
                try:
                    answer = await check()
                except Exception as err:
                    task = asyncio.current_task()
                    if task is not None and task.cancelling():  # anyio can turn a Ctrl+C into another error
                        raise
                    problem = f"its last check failed with {describe(leaves(err)[0])}"
                else:
                    break
            if time.monotonic() >= give_up:
                raise ServerError(
                    f"the {server.name} was not ready after {self._ready_limit_s:g} s ({problem}); "
                    f"its {server.output()}"
                )
            await asyncio.sleep(POLL_S)
        if (ended := server.ended()) is not None:  # so the answer came from another program
            raise ServerError(
                f"the {server.name} {ended}, yet something answered on port {port}: another server holds the "
                f"port; stop it first. The {server.name}'s {server.output()}"
            )
        return answer


def _local_ports(settings: Settings) -> tuple[int, int]:
    """The ports of the MCP server and of the retrieval agent, which a run can start on this machine only."""
    try:
        return local_port(settings.mcp_url, "EO_MCP_URL"), local_port(settings.a2a_url, "EO_A2A_URL")
    except ValueError as err:
        raise ServerError(str(err)) from err


def _stop(process: subprocess.Popen[bytes], grace_s: float) -> None:
    """Ask the process to stop, and kill it if it still runs after `grace_s` seconds."""
    process.terminate()  # nothing happens if it has ended already
    try:
        process.wait(grace_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
