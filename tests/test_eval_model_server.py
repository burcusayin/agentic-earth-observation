"""Tests for the model server of a run: Ollama asked over its HTTP API, answered here by a scripted server
behind httpx's mock transport. The request and answer shapes are the ones Ollama 0.40.1 gave on 2026-11-07.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
import pytest

from agentic_earth_observation.evals import model_server
from agentic_earth_observation.evals.model_server import (
    LOAD_TIMEOUT_S,
    LoadedModel,
    ModelServerError,
    OllamaServer,
)
from agentic_earth_observation.llm import OLLAMA_KEEP_ALIVE, SEED

pytestmark = pytest.mark.anyio

URL = "http://127.0.0.1:11434"
QWEN: dict[str, Any] = {"model": "qwen3.5:9b", "digest": "d63f" * 16, "size": 7639229431}
MINISTRAL: dict[str, Any] = {"model": "ministral-3:8b", "digest": "87fd" * 16, "size": 6022252443}
CONTEXT = 16384


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class ScriptedOllama:
    """Ollama's HTTP API: the models it has, and the ones it holds loaded with a context length."""

    def __init__(self, *models: dict[str, Any], loaded: dict[str, int] | None = None) -> None:
        self.models = list(models)
        self.loaded = dict(loaded or {})  # model -> context length
        self.requests: list[tuple[str, str, dict[str, Any]]] = []  # method, path, body ({} for a GET)
        self.answers: dict[str, httpx.Response] = {}  # a scripted answer for a path, in place of the real one
        self.load_context = 4096  # what a load without a context gives
        self.leaves_memory_after: int | None = (
            0  # how many looks at /api/ps an unloaded model stays; None: stays
        )
        self.leaving: dict[str, int] = {}
        self.keeps_loaded = True
        self.ignores_context = False  # loads with load_context whatever the request asks
        self.still_held_at_load: list[bool] = []  # for each load: was the model still in memory then?

    def server(self) -> OllamaServer:
        return OllamaServer(URL, http=httpx.AsyncClient(transport=httpx.MockTransport(self)))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body: dict[str, Any] = json.loads(request.content) if request.content else {}
        self.requests.append((request.method, path, body))
        if path in self.answers:
            return self.answers[path]
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.1"})
        if path == "/api/tags":
            return httpx.Response(200, json={"models": self.models})
        if path == "/api/ps":
            held = [
                {**m, "context_length": self.loaded[m["model"]], "expires_at": "2026-11-07T01:23:18Z"}
                for m in self.models
                if m["model"] in self.loaded
            ]
            for name in list(self.leaving):
                self.leaving[name] -= 1
                if self.leaving[name] < 0:
                    del self.leaving[name]
                    self.loaded.pop(name, None)
            return httpx.Response(200, json={"models": held})
        if path == "/api/generate":
            if not any(m["model"] == body["model"] for m in self.models):
                return httpx.Response(404, json={"error": f"model '{body['model']}' not found"})
            if body.get("keep_alive") == 0:
                if self.leaves_memory_after == 0:
                    self.loaded.pop(body["model"], None)
                elif self.leaves_memory_after is not None:
                    self.leaving[body["model"]] = self.leaves_memory_after
                reason = "unload"
            else:
                self.still_held_at_load.append(body["model"] in self.loaded)
                if self.keeps_loaded:
                    asked = body.get("options", {}).get("num_ctx", self.load_context)
                    self.loaded[body["model"]] = self.load_context if self.ignores_context else asked
                reason = "load"
            return httpx.Response(
                200, json={"model": body["model"], "response": "", "done": True, "done_reason": reason}
            )
        if path == "/api/chat":
            if body.get("model") not in self.loaded:
                self.loaded[body["model"]] = body.get("options", {}).get("num_ctx", self.load_context)
            return httpx.Response(
                200,
                json={
                    "model": body["model"],
                    "message": {"role": "assistant", "content": "ok"},
                    "done": True,
                },
            )
        return httpx.Response(404, text="404 page not found")


async def test_the_version_is_the_servers() -> None:
    async with ScriptedOllama().server() as server:
        assert await server.version() == "0.40.1"


async def test_a_model_is_there_when_the_server_lists_it() -> None:
    async with ScriptedOllama(QWEN).server() as server:
        assert await server.has("qwen3.5:9b")
        assert not await server.has("ministral-3:8b")


async def test_preparing_loads_the_model_with_the_context_and_gives_its_build() -> None:
    ollama = ScriptedOllama(QWEN, MINISTRAL)
    async with ollama.server() as server:
        loaded = await server.prepare("qwen3.5:9b", CONTEXT)
    assert loaded == LoadedModel(model="qwen3.5:9b", digest=QWEN["digest"], context_length=CONTEXT)
    assert ollama.loaded == {"qwen3.5:9b": CONTEXT}
    (load,) = [body for method, path, body in ollama.requests if path == "/api/generate"]
    assert load == {
        "model": "qwen3.5:9b",
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"num_ctx": CONTEXT},
        "stream": False,
    }


async def test_preparing_unloads_every_other_model_first() -> None:
    ollama = ScriptedOllama(QWEN, MINISTRAL, loaded={"ministral-3:8b": 4096})
    async with ollama.server() as server:
        await server.prepare("qwen3.5:9b", CONTEXT)
    assert ollama.loaded == {"qwen3.5:9b": CONTEXT}
    generated = [body for method, path, body in ollama.requests if path == "/api/generate"]
    assert generated[0] == {"model": "ministral-3:8b", "keep_alive": 0, "stream": False}
    assert generated[1]["model"] == "qwen3.5:9b"


async def test_preparing_waits_until_the_model_itself_has_left_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_server, "POLL_S", 0.0)
    ollama = ScriptedOllama(QWEN, loaded={"qwen3.5:9b": CONTEXT})
    ollama.leaves_memory_after = 2
    async with ollama.server() as server:
        await server.prepare("qwen3.5:9b", CONTEXT)
    assert ollama.still_held_at_load == [False]  # loaded only once the old copy had left memory


async def test_preparing_waits_until_the_other_model_has_left_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    # The unload is answered at once; a model in use leaves when it is done.
    monkeypatch.setattr(model_server, "POLL_S", 0.0)
    ollama = ScriptedOllama(QWEN, MINISTRAL, loaded={"ministral-3:8b": 4096})
    ollama.leaves_memory_after = 2
    async with ollama.server() as server:
        await server.prepare("qwen3.5:9b", CONTEXT)
    assert ollama.loaded == {"qwen3.5:9b": CONTEXT}


async def test_a_model_that_does_not_leave_memory_stops_the_run_before_the_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_server, "POLL_S", 0.0)
    monkeypatch.setattr(model_server, "UNLOAD_WAIT_S", 0.05)
    ollama = ScriptedOllama(QWEN, MINISTRAL, loaded={"ministral-3:8b": 4096})
    ollama.leaves_memory_after = None
    async with ollama.server() as server:
        with pytest.raises(ModelServerError, match=r"still holds ministral-3:8b after 0\.05 s"):
            await server.prepare("qwen3.5:9b", CONTEXT)
    assert "qwen3.5:9b" not in ollama.loaded


def test_a_client_of_ollama_on_this_machine_ignores_proxy_settings() -> None:
    assert OllamaServer(URL)._http.trust_env is False
    assert OllamaServer("https://models.example")._http.trust_env is True


async def test_preparing_a_model_that_is_loaded_already_loads_it_afresh() -> None:
    # Ollama answers a prompt it has seen from a cache, so a model kept loaded from the run before would
    # answer that run's prompts in a fraction of the time: every run starts from a fresh load (2026-11-08).
    ollama = ScriptedOllama(QWEN, loaded={"qwen3.5:9b": CONTEXT})
    async with ollama.server() as server:
        await server.prepare("qwen3.5:9b", CONTEXT)
    generated = [body for method, path, body in ollama.requests if path == "/api/generate"]
    assert generated[0] == {"model": "qwen3.5:9b", "keep_alive": 0, "stream": False}
    assert generated[1]["options"] == {"num_ctx": CONTEXT}
    assert ollama.loaded == {"qwen3.5:9b": CONTEXT}


async def test_a_load_gets_time_for_a_model_of_several_gigabytes() -> None:
    ollama = ScriptedOllama(QWEN)
    seen: list[dict[str, float | None]] = []  # httpx passes the four limits as a dict

    class Timed(httpx.MockTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            seen.append(request.extensions["timeout"])
            return await super().handle_async_request(request)

    server = OllamaServer(URL, http=httpx.AsyncClient(transport=Timed(ollama)))
    async with server:
        await server.prepare("qwen3.5:9b", CONTEXT)
    (load,) = [
        t for (method, path, body), t in zip(ollama.requests, seen, strict=True) if path.endswith("generate")
    ]
    assert load["read"] == LOAD_TIMEOUT_S


async def test_a_model_the_server_does_not_keep_loaded_is_reported() -> None:
    ollama = ScriptedOllama(QWEN)
    ollama.keeps_loaded = False
    async with ollama.server() as server:
        with pytest.raises(ModelServerError, match=r"did not keep qwen3\.5:9b loaded"):
            await server.prepare("qwen3.5:9b", CONTEXT)


async def test_a_model_loaded_with_another_context_is_reported() -> None:
    ollama = ScriptedOllama(QWEN)
    ollama.ignores_context = True
    async with ollama.server() as server:
        with pytest.raises(ModelServerError, match="with a context of 4096, not 16384"):
            await server.prepare("qwen3.5:9b", CONTEXT)


async def test_unloading_asks_for_no_keep_alive() -> None:
    ollama = ScriptedOllama(QWEN, loaded={"qwen3.5:9b": CONTEXT})
    async with ollama.server() as server:
        await server.unload("qwen3.5:9b")
    assert ollama.loaded == {}
    assert ollama.requests == [
        ("POST", "/api/generate", {"model": "qwen3.5:9b", "keep_alive": 0, "stream": False})
    ]


async def test_a_model_the_server_does_not_have_is_reported_with_its_words() -> None:
    async with ScriptedOllama(QWEN).server() as server:
        with pytest.raises(
            ModelServerError, match=re.escape("""404: {"error":"model 'nope:1b' not found"}""")
        ):
            await server.unload("nope:1b")


async def test_a_server_that_cannot_be_reached_is_reported() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("All connection attempts failed", request=request)

    server = OllamaServer(URL, http=httpx.AsyncClient(transport=httpx.MockTransport(refuse)))
    async with server:
        with pytest.raises(ModelServerError, match="cannot be reached \\(ConnectError\\): is it running"):
            await server.version()


@pytest.mark.parametrize(
    ("answer", "said"),
    [
        pytest.param(httpx.Response(200, text="<html>"), "is not JSON", id="not JSON"),
        pytest.param(httpx.Response(200, json={"models": "none"}), "is not understood", id="another shape"),
    ],
)
async def test_an_answer_that_is_not_ollamas_is_reported(answer: httpx.Response, said: str) -> None:
    ollama = ScriptedOllama(QWEN)
    ollama.answers["/api/tags"] = answer
    async with ollama.server() as server:
        with pytest.raises(ModelServerError, match=f"/api/tags {said}"):
            await server.has("qwen3.5:9b")


async def test_the_url_may_end_with_a_slash() -> None:
    ollama = ScriptedOllama()
    server = OllamaServer(URL + "/", http=httpx.AsyncClient(transport=httpx.MockTransport(ollama)))
    async with server:
        await server.version()
    assert ollama.requests == [("GET", "/api/version", {})]


async def test_the_warm_up_is_one_short_call_on_unrelated_text_at_the_run_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter([100.0, 102.5])  # before the call, after it

    def tick() -> float:
        return next(clock)

    monkeypatch.setattr(model_server, "clock", tick)
    ollama = ScriptedOllama(QWEN, loaded={"qwen3.5:9b": CONTEXT})
    async with ollama.server() as server:
        seconds = await server.warm_up("qwen3.5:9b", CONTEXT)
    assert seconds == 2.5
    (chat,) = [body for method, path, body in ollama.requests if path == "/api/chat"]
    assert chat["model"] == "qwen3.5:9b"
    assert chat["stream"] is False
    assert chat["think"] is False
    assert chat["keep_alive"] == OLLAMA_KEEP_ALIVE
    assert chat["options"] == {"num_ctx": CONTEXT, "temperature": 0, "seed": SEED, "num_predict": 8}
    (message,) = chat["messages"]
    assert message == {"role": "user", "content": model_server.WARM_UP_TEXT}


def test_the_warm_up_text_names_nothing_of_the_golden_set() -> None:
    # It is cached by Ollama: no task may find its own words there. tests/test_no_leakage.py checks it against
    # the test split, since its scan of system code leaves evals/ out.
    words = set(model_server.WARM_UP_TEXT.lower().replace(",", " ").replace(".", " ").split())
    assert not words & {"sentinel", "scene", "scenes", "satellite", "cloud", "radar", "image", "images"}
    assert 200 < len(model_server.WARM_UP_TEXT) < 4000


async def test_a_warm_up_that_reloads_the_model_with_another_context_is_reported() -> None:
    ollama = ScriptedOllama(QWEN, loaded={"qwen3.5:9b": CONTEXT})
    ollama.answers["/api/ps"] = httpx.Response(200, json={"models": [{**QWEN, "context_length": 4096}]})
    async with ollama.server() as server:
        with pytest.raises(ModelServerError, match="with a context of 4096, not 16384"):
            await server.warm_up("qwen3.5:9b", CONTEXT)


async def test_a_warm_up_the_server_refuses_is_reported_with_its_words() -> None:
    ollama = ScriptedOllama(QWEN, loaded={"qwen3.5:9b": CONTEXT})
    ollama.answers["/api/chat"] = httpx.Response(500, json={"error": "model runner has unexpectedly stopped"})
    async with ollama.server() as server:
        with pytest.raises(
            ModelServerError, match=r"Ollama answered /api/chat with 500: .*unexpectedly stopped"
        ):
            await server.warm_up("qwen3.5:9b", CONTEXT)
