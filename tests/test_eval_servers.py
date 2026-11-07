"""Tests for the servers of an evaluation run: how the runner starts, watches and stops the MCP server and the
retrieval agent.

The unit tests start real child processes, but small stand-ins that open no socket: a stand-in "listens" by
writing its process id to a file, and the tests' probe calls a port open while that process runs. The two
checks that speak HTTP are tested against the real servers in-process. One test, marked `integration`, starts
the real servers on their ports; it needs no model and makes no catalogue search.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any

import httpx
import pytest
from mcp.shared._httpx_utils import McpHttpClientFactory
from mcp.types import Tool
from pydantic import SecretStr
from starlette.applications import Starlette

from agentic_earth_observation.catalog.gazetteer import Gazetteer
from agentic_earth_observation.catalog_mcp.server import build_server, tool_descriptions_hash
from agentic_earth_observation.config import ROOT, Settings
from agentic_earth_observation.evals.records import TOOL_LOG_FILE
from agentic_earth_observation.evals.servers import (
    AGENT_LOG,
    MCP_LOG,
    Probe,
    RunServers,
    ServerDied,
    ServerError,
    agent_answers,
    port_open,
    served_tools,
    server_environment,
)
from agentic_earth_observation.retrieval_agent.a2a_server import RetrievalExecutor, build_app
from tests.fakes import FakeCatalog, ScriptedModel

pytestmark = pytest.mark.anyio

MCP_PORT = 8001
AGENT_PORT = 8002
TOOLS = [
    Tool(name="resolve_place", description="Find a known place by name.", inputSchema={"type": "object"}),
    Tool(name="search_scenes", description="Search satellite scenes.", inputSchema={"type": "object"}),
]

# A stand-in server: `python stand_in.py <name> <behaviour> <port file> <events file>`.
#   serve       claims its port, and stops when asked (SIGTERM);
#   stubborn    claims its port, and ignores SIGTERM;
#   deaf        runs, and never claims its port;
#   crash       exits with code 3 at once;
#   serve-once  serves the first time it is started, and crashes every time after.
# It never outlives a test by more than a minute, whatever the test does.
STAND_IN = """\
import os, signal, sys, time
from pathlib import Path

name, behaviour, port_file, events = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])


def note(what):
    with events.open("a") as out:
        out.write(f"{name} {what}\\n")


def stop(signum, frame):
    note("stopped")
    sys.exit(0)


print(
    f"{name} says: model={os.environ.get('EO_LLM_MODEL')} log={os.environ.get('EO_TOOL_LOG')} "
    f"cwd={os.getcwd()} own_session={os.getsid(0) == os.getpid()}",
    flush=True,
)
lives = events.read_text().count(f"{name} started") if events.exists() else 0
if behaviour == "crash" or (behaviour == "serve-once" and lives):
    print(f"{name} cannot start", file=sys.stderr, flush=True)
    sys.exit(3)
signal.signal(signal.SIGTERM, signal.SIG_IGN if behaviour == "stubborn" else stop)
if behaviour != "deaf":
    draft = port_file.with_suffix(".draft")
    draft.write_text(str(os.getpid()))
    draft.replace(port_file)
    note("started")
time.sleep(60)
sys.exit(4)
"""


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def settings(**changes: Any) -> Settings:  # noqa: ANN401 - any setting, validated by Settings itself
    return Settings(_env_file=None, **changes)  # type: ignore[call-arg]  # a pydantic-settings init option


def running(pid: int) -> bool:
    """Whether a child of this process is still running. One that has exited is not, collected or not."""
    try:
        return os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT | os.WNOHANG) is None
    except ChildProcessError:  # its parent has collected it already
        return False


def gone(pid: int, within_s: float = 5.0) -> bool:
    """Whether a child ends within the time, without collecting it."""
    give_up = time.monotonic() + within_s
    while running(pid):
        if time.monotonic() >= give_up:
            return False
        time.sleep(0.01)
    return True


def end(pid: int, how: signal.Signals = signal.SIGKILL) -> None:
    """Send a child the signal and wait until it has exited, without collecting it: that is the runner's."""
    os.kill(pid, how)
    os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)


class StandIns:
    """The stand-in servers of one test, and the probe that looks at them instead of the network."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.run_folder = folder / "run"
        self.script = folder / "stand_in.py"
        self.script.write_text(STAND_IN, encoding="utf-8")
        self.events_file = folder / "events"
        self.busy: set[int] = set()  # ports that another program holds
        self.tool_failures: list[Exception] = []  # what the next checks of the MCP server raise, in order
        self.during_tool_check: Callable[[], None] | None = None
        self.checks: list[str] = []
        self.made: list[RunServers] = []

    def servers(
        self,
        mcp: str = "serve",
        agent: str = "serve",
        *,
        config: Settings | None = None,
        ready_limit_s: float = 10.0,
        stop_grace_s: float = 5.0,
    ) -> RunServers:
        made = RunServers(
            config or settings(),
            self.run_folder,
            probe=Probe(
                port_open=self.port_open, served_tools=self.served_tools, agent_answers=self.agent_answers
            ),
            mcp_command=self.command("mcp", mcp, MCP_PORT),
            agent_command=self.command("agent", agent, AGENT_PORT),
            ready_limit_s=ready_limit_s,
            stop_grace_s=stop_grace_s,
        )
        self.made.append(made)
        return made

    def command(self, name: str, behaviour: str, port: int) -> list[str]:
        return [
            sys.executable,
            str(self.script),
            name,
            behaviour,
            str(self.port_file(port)),
            str(self.events_file),
        ]

    def port_file(self, port: int) -> Path:
        return self.folder / f"port-{port}"

    def pid(self, port: int) -> int:
        """The process id of the stand-in that claimed `port`."""
        return int(self.port_file(port).read_text())

    def port_open(self, port: int) -> bool:
        if port in self.busy:
            return True
        return self.port_file(port).exists() and running(self.pid(port))

    async def served_tools(self, config: Settings) -> list[Tool]:
        self.checks.append("tools")
        if self.tool_failures:
            raise self.tool_failures.pop(0)
        if self.during_tool_check is not None:
            self.during_tool_check()
        return TOOLS

    async def agent_answers(self, config: Settings) -> None:
        self.checks.append("card")

    def events(self) -> list[str]:
        return self.events_file.read_text().splitlines() if self.events_file.exists() else []

    def log(self, name: str) -> str:
        return (self.run_folder / name).read_text(encoding="utf-8")


@pytest.fixture
def stand_ins(tmp_path: Path) -> Iterator[StandIns]:
    made = StandIns(tmp_path)
    yield made
    for servers in made.made:
        servers.stop()


def rebuilt(environment: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    """The settings a server reads from `environment`, with no `.env` beside it."""
    for name, value in environment.items():
        if name.startswith("EO_"):
            monkeypatch.setenv(name, value)
    return settings()


class TestEnvironment:
    @pytest.mark.parametrize(
        "changes",
        [
            pytest.param({}, id="the defaults"),
            pytest.param({"llm_model": "ministral-3:8b"}, id="another model"),
            pytest.param({"reference_date": date(2025, 12, 15)}, id="a reference date"),
            pytest.param(
                {"turn_deadline_s": 200.0, "mcp_timeout_s": 100.0, "stac_max_retries": 0, "num_predict": 512},
                id="other limits",
            ),
            pytest.param(
                {"ollama_url": "http://127.0.0.1:11500", "a2a_url": "http://localhost:9100"}, id="other URLs"
            ),
        ],
    )
    def test_a_server_reads_exactly_the_runs_settings_from_it(
        self, changes: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = settings(**changes)
        log = tmp_path / TOOL_LOG_FILE
        environment = server_environment(config, log, base={})
        assert rebuilt(environment, monkeypatch) == config.model_copy(update={"tool_log": log})

    def test_the_runs_settings_beat_the_shells(self, tmp_path: Path) -> None:
        shell = {"EO_LLM_MODEL": "another-model", "EO_TOOL_LOG": "/elsewhere/calls.jsonl"}
        environment = server_environment(settings(), tmp_path / TOOL_LOG_FILE, base=shell)
        assert environment["EO_LLM_MODEL"] == "qwen3.5:9b"
        assert environment["EO_TOOL_LOG"] == str(tmp_path / TOOL_LOG_FILE)

    def test_the_rest_of_the_environment_is_kept(self, tmp_path: Path) -> None:
        environment = server_environment(settings(), tmp_path / TOOL_LOG_FILE, base={"PATH": "/usr/bin"})
        assert environment["PATH"] == "/usr/bin"

    def test_the_runners_own_environment_is_the_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SOMETHING_OF_THE_SHELL", "kept")
        assert server_environment(settings(), tmp_path / TOOL_LOG_FILE)["SOMETHING_OF_THE_SHELL"] == "kept"

    def test_a_secret_is_never_copied_from_the_settings(self, tmp_path: Path) -> None:
        config = settings(mistral_api_key=SecretStr("only-in-the-settings"))
        environment = server_environment(config, tmp_path / TOOL_LOG_FILE, base={})
        assert "EO_MISTRAL_API_KEY" not in environment
        assert not any("only-in-the-settings" in value or "*" in value for value in environment.values())

    def test_a_key_in_the_shell_reaches_the_server_as_it_is(self, tmp_path: Path) -> None:
        config = settings(mistral_api_key=SecretStr("only-in-the-settings"))
        shell = {"EO_MISTRAL_API_KEY": "from-the-shell"}
        environment = server_environment(config, tmp_path / TOOL_LOG_FILE, base=shell)
        assert environment["EO_MISTRAL_API_KEY"] == "from-the-shell"

    def test_a_setting_without_a_value_is_not_stated(self, tmp_path: Path) -> None:
        environment = server_environment(settings(), tmp_path / TOOL_LOG_FILE, base={})
        assert "EO_REFERENCE_DATE" not in environment
        assert "None" not in environment.values()


class TestStart:
    async def test_both_servers_start_with_the_runs_settings_and_write_to_its_folder(
        self, stand_ins: StandIns, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(
            stand_ins.folder
        )  # the servers must start from the checkout wherever the runner runs
        config = settings(llm_model="ministral-3:8b")
        async with stand_ins.servers(config=config):
            assert running(stand_ins.pid(MCP_PORT))
            assert running(stand_ins.pid(AGENT_PORT))
        log = stand_ins.run_folder / TOOL_LOG_FILE
        said = f"says: model=ministral-3:8b log={log} cwd={ROOT} own_session=True\n"
        assert stand_ins.log(MCP_LOG) == f"mcp {said}"
        assert stand_ins.log(AGENT_LOG) == f"agent {said}"

    async def test_the_agent_starts_once_the_mcp_server_is_ready(self, stand_ins: StandIns) -> None:
        async with stand_ins.servers():
            assert stand_ins.events() == ["mcp started", "agent started"]
            assert stand_ins.checks == ["tools", "card"]

    async def test_the_tools_it_records_are_the_ones_the_mcp_server_serves(self, stand_ins: StandIns) -> None:
        servers = stand_ins.servers()
        with pytest.raises(ServerError, match="have not been started"):
            _ = servers.tool_descriptions
        async with servers:
            assert servers.tool_descriptions == tool_descriptions_hash(TOOLS)

    async def test_a_relative_folder_gives_the_servers_an_absolute_log(
        self, stand_ins: StandIns, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(stand_ins.folder)
        stand_ins.run_folder = Path("run")
        async with stand_ins.servers():
            pass
        assert f"log={stand_ins.folder / 'run' / TOOL_LOG_FILE} " in stand_ins.log(MCP_LOG)

    @pytest.mark.parametrize(("port", "server"), [(MCP_PORT, "MCP server"), (AGENT_PORT, "retrieval agent")])
    async def test_a_port_that_already_answers_is_refused_and_nothing_is_started(
        self, stand_ins: StandIns, port: int, server: str
    ) -> None:
        stand_ins.busy.add(port)
        with pytest.raises(ServerError) as refused:
            await stand_ins.servers().start()
        assert f"something already answers on port {port}, where the run starts its {server}" in str(
            refused.value
        )
        assert "stop it first" in str(refused.value)
        assert stand_ins.events() == []
        assert not stand_ins.run_folder.exists()

    @pytest.mark.parametrize(
        ("changes", "said"),
        [
            pytest.param(
                {"mcp_url": "http://127.0.0.1:9001/mcp"},
                "EO_MCP_URL must be http://127.0.0.1:8001",
                id="port",
            ),
            pytest.param(
                {"mcp_url": "http://example.org:8001/mcp"},
                "EO_MCP_URL must be http://127.0.0.1:8001",
                id="host",
            ),
            pytest.param(
                {"mcp_url": "https://127.0.0.1:8001/mcp"},
                "EO_MCP_URL must be http://127.0.0.1:8001",
                id="https",
            ),
            pytest.param(
                {"mcp_url": "http://127.0.0.1:port/mcp"},
                "EO_MCP_URL must be http://127.0.0.1:8001",
                id="no port",
            ),
            pytest.param(
                {"a2a_url": "http://example.org:8002"},
                "EO_A2A_URL must be http://127.0.0.1:<port>",
                id="agent",
            ),
        ],
    )
    async def test_a_url_the_run_cannot_serve_is_refused(
        self, stand_ins: StandIns, changes: dict[str, str], said: str
    ) -> None:
        with pytest.raises(ServerError, match=re.escape(said)):
            await stand_ins.servers(config=settings(**changes)).start()
        assert stand_ins.events() == []

    async def test_localhost_is_this_machine_too(self, stand_ins: StandIns) -> None:
        config = settings(mcp_url="http://localhost:8001/mcp", a2a_url="http://localhost:8002")
        async with stand_ins.servers(config=config):
            assert stand_ins.events() == ["mcp started", "agent started"]

    async def test_a_folder_that_holds_a_tool_log_is_refused(self, stand_ins: StandIns) -> None:
        stand_ins.run_folder.mkdir()
        (stand_ins.run_folder / TOOL_LOG_FILE).write_text("", encoding="utf-8")
        with pytest.raises(
            ServerError, match=r"already holds a tool-call log .*: every run gets its own folder"
        ):
            await stand_ins.servers().start()
        assert stand_ins.events() == []

    async def test_an_agent_that_exits_at_once_is_reported_and_the_mcp_server_stopped(
        self, stand_ins: StandIns
    ) -> None:
        with pytest.raises(ServerError) as failed:
            await stand_ins.servers(agent="crash").start()
        assert str(failed.value) == (
            "the retrieval agent exited with code 3 before it was ready; "
            "its output is in retrieval_agent.log in the run's folder"
        )
        assert not isinstance(failed.value, ServerDied)
        assert stand_ins.events() == ["mcp started", "mcp stopped"]
        assert "agent cannot start" in stand_ins.log(AGENT_LOG)

    async def test_an_mcp_server_that_exits_at_once_is_reported_and_no_agent_started(
        self, stand_ins: StandIns
    ) -> None:
        with pytest.raises(ServerError, match="the MCP server exited with code 3 before it was ready"):
            await stand_ins.servers(mcp="crash").start()
        assert stand_ins.events() == []
        assert not (stand_ins.run_folder / AGENT_LOG).exists()

    async def test_a_server_that_never_opens_its_port_is_given_up_on(self, stand_ins: StandIns) -> None:
        started = time.monotonic()
        with pytest.raises(ServerError) as failed:
            await stand_ins.servers(agent="deaf", ready_limit_s=0.3).start()
        assert 0.3 <= time.monotonic() - started < 5
        assert str(failed.value) == (
            "the retrieval agent was not ready after 0.3 s (its port did not open); "
            "its output is in retrieval_agent.log in the run's folder"
        )
        events = stand_ins.events()  # the deaf agent may be stopped before it can say so
        assert (events[0], events[-1]) == ("mcp started", "mcp stopped")
        assert "agent started" not in events

    async def test_a_check_that_fails_is_tried_again(self, stand_ins: StandIns) -> None:
        stand_ins.tool_failures = [
            httpx.ConnectError("not yet"),
            ExceptionGroup("wrapped", [OSError("nor now")]),
        ]
        async with stand_ins.servers():
            assert stand_ins.checks == ["tools", "tools", "tools", "card"]

    async def test_a_server_whose_check_keeps_failing_is_given_up_on_with_the_last_failure(
        self, stand_ins: StandIns
    ) -> None:
        stand_ins.tool_failures = [ExceptionGroup("wrapped", [httpx.ConnectError("refused")])] * 10_000
        with pytest.raises(ServerError) as failed:
            await stand_ins.servers(ready_limit_s=0.3).start()
        assert str(failed.value) == (
            "the MCP server was not ready after 0.3 s (its last check failed with ConnectError: refused); "
            "its output is in mcp_server.log in the run's folder"
        )
        assert stand_ins.events() == ["mcp started", "mcp stopped"]

    async def test_a_server_that_died_while_another_answered_for_it_is_not_ready(
        self, stand_ins: StandIns
    ) -> None:
        # What the prototype showed: an older server on the port answers the check, and the new one exits.
        stand_ins.during_tool_check = lambda: end(stand_ins.pid(MCP_PORT))
        with pytest.raises(ServerError) as failed:
            await stand_ins.servers().start()
        assert str(failed.value) == (
            "the MCP server was killed by SIGKILL, yet something answered on port 8001: another server holds "
            "the port; stop it first. The MCP server's output is in mcp_server.log in the run's folder"
        )
        assert stand_ins.events() == ["mcp started"]

    async def test_starting_twice_is_refused(self, stand_ins: StandIns) -> None:
        async with stand_ins.servers() as servers:
            with pytest.raises(ServerError, match="already started"):
                await servers.start()
            assert stand_ins.events() == ["mcp started", "agent started"]
            servers.check()


class TestCheck:
    async def test_servers_that_run_pass(self, stand_ins: StandIns) -> None:
        async with stand_ins.servers() as servers:
            servers.check()

    @pytest.mark.parametrize(
        ("port", "said"),
        [
            (MCP_PORT, f"the MCP server was killed by SIGKILL during the run; its output is in {MCP_LOG}"),
            (
                AGENT_PORT,
                f"the retrieval agent was killed by SIGKILL during the run; its output is in {AGENT_LOG}",
            ),
        ],
    )
    async def test_a_server_that_was_killed_is_reported(
        self, stand_ins: StandIns, port: int, said: str
    ) -> None:
        async with stand_ins.servers() as servers:
            end(stand_ins.pid(port))
            with pytest.raises(ServerDied) as died:
                servers.check()
        assert str(died.value) == f"{said} in the run's folder"

    async def test_a_server_that_exited_is_reported_with_its_code(self, stand_ins: StandIns) -> None:
        async with stand_ins.servers() as servers:
            end(stand_ins.pid(AGENT_PORT), signal.SIGTERM)  # the stand-in then exits with code 0
            with pytest.raises(ServerDied, match="the retrieval agent exited with code 0 during the run"):
                servers.check()

    async def test_the_report_names_no_path_of_this_machine(self, stand_ins: StandIns) -> None:
        # The runner writes the reason into the run record, which may be committed.
        async with stand_ins.servers() as servers:
            end(stand_ins.pid(MCP_PORT))
            with pytest.raises(ServerDied) as died:
                servers.check()
        assert str(stand_ins.folder) not in str(died.value)
        assert "/" not in str(died.value)

    async def test_servers_that_are_not_running_cannot_be_checked(self, stand_ins: StandIns) -> None:
        servers = stand_ins.servers()
        with pytest.raises(ServerError, match="are not running"):
            servers.check()
        async with servers:
            pass
        with pytest.raises(ServerError, match="are not running") as stopped:
            servers.check()
        assert not isinstance(stopped.value, ServerDied)


class TestStop:
    async def test_leaving_the_run_stops_both_the_agent_first(self, stand_ins: StandIns) -> None:
        async with stand_ins.servers():
            mcp, agent = stand_ins.pid(MCP_PORT), stand_ins.pid(AGENT_PORT)
        assert not running(mcp)
        assert not running(agent)
        assert stand_ins.events() == ["mcp started", "agent started", "agent stopped", "mcp stopped"]

    async def test_an_error_in_the_run_still_stops_them(self, stand_ins: StandIns) -> None:
        with pytest.raises(RuntimeError, match="a bug in the run"):
            async with stand_ins.servers():
                raise RuntimeError("a bug in the run")
        assert stand_ins.events()[2:] == ["agent stopped", "mcp stopped"]

    async def test_a_server_that_stops_when_asked_is_not_waited_for(self, stand_ins: StandIns) -> None:
        async with stand_ins.servers(stop_grace_s=30.0):
            started = time.monotonic()
        assert time.monotonic() - started < 10

    async def test_a_server_that_ignores_the_signal_is_killed_after_the_grace_period(
        self, stand_ins: StandIns
    ) -> None:
        servers = stand_ins.servers(agent="stubborn", stop_grace_s=0.4)
        await servers.start()
        agent = stand_ins.pid(AGENT_PORT)
        started = time.monotonic()
        servers.stop()
        assert 0.4 <= time.monotonic() - started < 5
        assert not running(agent)
        assert stand_ins.events() == ["mcp started", "agent started", "mcp stopped"]

    async def test_stopping_twice_is_harmless(self, stand_ins: StandIns) -> None:
        servers = stand_ins.servers()
        await servers.start()
        servers.stop()
        servers.stop()
        assert stand_ins.events() == ["mcp started", "agent started", "agent stopped", "mcp stopped"]

    async def test_a_stop_that_is_interrupted_leaves_nothing_running(
        self, stand_ins: StandIns, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A second Ctrl+C arrives while the runner waits for a server to stop.
        servers = stand_ins.servers(mcp="stubborn", agent="stubborn")
        await servers.start()
        mcp, agent = stand_ins.pid(MCP_PORT), stand_ins.pid(AGENT_PORT)

        def interrupted(process: subprocess.Popen[bytes], timeout: float | None = None) -> int:
            raise KeyboardInterrupt

        with monkeypatch.context() as patched:
            patched.setattr(subprocess.Popen, "wait", interrupted)
            with pytest.raises(KeyboardInterrupt):
                servers.stop()
        assert gone(mcp)
        assert gone(agent)
        assert stand_ins.events() == ["mcp started", "agent started"]


class TestRestart:
    async def test_the_agent_is_replaced_and_the_mcp_server_goes_on(self, stand_ins: StandIns) -> None:
        async with stand_ins.servers() as servers:
            mcp, old = stand_ins.pid(MCP_PORT), stand_ins.pid(AGENT_PORT)
            described = servers.tool_descriptions
            await servers.restart_agent()
            new = stand_ins.pid(AGENT_PORT)
            assert new != old
            assert not running(old)
            assert running(new)
            assert stand_ins.pid(MCP_PORT) == mcp
            assert running(mcp)
            assert servers.tool_descriptions == described
            servers.check()
            assert stand_ins.events() == ["mcp started", "agent started", "agent stopped", "agent started"]
            assert stand_ins.checks == ["tools", "card", "card"]
        assert stand_ins.log(AGENT_LOG).count("agent says:") == 2  # one file for both
        assert stand_ins.events()[-2:] == ["agent stopped", "mcp stopped"]

    async def test_an_agent_that_ignores_the_signal_is_killed_and_replaced_while_the_run_goes_on(
        self, stand_ins: StandIns
    ) -> None:
        # The retrieval agent does not stop on SIGTERM while a turn is in flight (the prototype).
        ticks: list[float] = []

        async def tick() -> None:
            while True:
                await asyncio.sleep(0.01)
                ticks.append(time.monotonic())

        async with stand_ins.servers(agent="stubborn", stop_grace_s=1.0) as servers:
            old = stand_ins.pid(AGENT_PORT)
            ticking = asyncio.ensure_future(tick())
            await asyncio.sleep(0.03)  # the ticking is under way
            began = time.monotonic()
            await servers.restart_agent()
            ended = time.monotonic()
            ticking.cancel()
            assert not running(old)
            assert running(stand_ins.pid(AGENT_PORT))
            assert stand_ins.pid(AGENT_PORT) != old
            servers.stop()
        moments = [began, *(tick for tick in ticks if began < tick < ended), ended]
        longest_pause = max(later - earlier for earlier, later in pairwise(moments))
        assert longest_pause < 0.5  # the second of waiting for the old agent did not hold up the event loop

    @pytest.mark.parametrize(("port", "server"), [(MCP_PORT, "MCP server"), (AGENT_PORT, "retrieval agent")])
    async def test_a_server_found_dead_is_reported_and_nothing_restarted(
        self, stand_ins: StandIns, port: int, server: str
    ) -> None:
        async with stand_ins.servers() as servers:
            end(stand_ins.pid(port))
            with pytest.raises(ServerDied, match=f"the {server} was killed by SIGKILL during the run"):
                await servers.restart_agent()
            assert stand_ins.events() == ["mcp started", "agent started"]

    async def test_a_new_agent_that_does_not_start_is_reported(self, stand_ins: StandIns) -> None:
        async with stand_ins.servers(agent="serve-once") as servers:
            with pytest.raises(
                ServerError, match="the retrieval agent exited with code 3 before it was ready"
            ):
                await servers.restart_agent()
        assert stand_ins.events() == ["mcp started", "agent started", "agent stopped", "mcp stopped"]

    async def test_a_port_taken_in_the_meantime_is_refused(self, stand_ins: StandIns) -> None:
        async with stand_ins.servers() as servers:
            stand_ins.busy.add(AGENT_PORT)
            with pytest.raises(ServerError, match="something already answers on port 8002"):
                await servers.restart_agent()
            assert stand_ins.events() == ["mcp started", "agent started", "agent stopped"]


@asynccontextmanager
async def catalogue() -> AsyncIterator[McpHttpClientFactory]:
    """The real MCP server in-process; yields the HTTP client factory that reaches it."""
    server = build_server(FakeCatalog(), Gazetteer.load())
    app = server.streamable_http_app()

    def client(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), headers=headers, timeout=timeout, auth=auth
        )

    async with server.session_manager.run():
        yield client


class TestProbe:
    async def test_the_tools_are_the_ones_the_mcp_server_serves(self) -> None:
        in_process = await build_server(FakeCatalog(), Gazetteer.load()).list_tools()
        async with catalogue() as http:
            tools = await served_tools(settings(), http=http)
        assert sorted(tool.name for tool in tools) == [
            "get_scene",
            "list_products",
            "resolve_place",
            "search_scenes",
        ]
        assert tool_descriptions_hash(tools) == tool_descriptions_hash(in_process)

    async def test_the_agents_card_is_read_as_its_client_reads_it(self) -> None:
        config = settings()
        app = build_app(RetrievalExecutor(ScriptedModel(messages=iter([])), config), config)
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app))
        await agent_answers(config, http=http)
        assert http.is_closed

    async def test_a_server_without_the_agents_card_does_not_answer_as_the_agent(self) -> None:
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=Starlette()))
        with pytest.raises(Exception, match="HTTP 404"):
            await agent_answers(settings(), http=http)


@pytest.mark.integration
async def test_the_real_servers_start_answer_and_stop(tmp_path: Path) -> None:
    config = settings()
    folder = tmp_path / "run"
    in_process = await build_server(FakeCatalog(), Gazetteer.load()).list_tools()
    assert not port_open(MCP_PORT)
    assert not port_open(AGENT_PORT)
    async with RunServers(config, folder) as servers:
        assert port_open(MCP_PORT)
        assert port_open(AGENT_PORT)
        assert servers.tool_descriptions == tool_descriptions_hash(in_process)
        assert (folder / TOOL_LOG_FILE).exists()  # the MCP server opened the run's log, not the default one
        servers.check()
        with pytest.raises(ServerError, match="something already answers on port 8001"):
            await RunServers(config, tmp_path / "second").start()
        await servers.restart_agent()
        servers.check()
        await agent_answers(config)
        assert len(await served_tools(config)) == 4
    assert not port_open(MCP_PORT)
    assert not port_open(AGENT_PORT)
    assert "Uvicorn running on http://127.0.0.1:8001" in (folder / MCP_LOG).read_text(encoding="utf-8")
    assert (folder / AGENT_LOG).read_text(encoding="utf-8").count(
        "Uvicorn running on http://127.0.0.1:8002"
    ) == 2
