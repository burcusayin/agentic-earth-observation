"""The model server of a run: which build of the model runs, that it alone is loaded, and loaded afresh.

Ollama is asked over its HTTP API: its version, the models it has, the ones it holds in memory, and a load or
an unload through /api/generate with `keep_alive`. A bare load gives the model Ollama's default context, and
the agents' first request would then reload it with theirs, so the load states the context. Checked against
Ollama 0.40.1 on 2026-11-07.

Ollama answers a prompt it has seen before from a cache, in a fraction of the time, while still reporting
every prompt token. At temperature 0 a run repeats the prompts of the run before, so a model kept loaded
between runs would make the later runs look about twice as fast. Every run therefore starts from a fresh
load, then one warm-up call on unrelated text takes the model's first-call cost (measured on 2026-11-08:
about 1.3 s for ministral-3:8b, none for qwen3.5:9b) without caching anything a task will ask.
"""

from __future__ import annotations

import asyncio
import time
from types import TracebackType
from typing import Any, Protocol, Self

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from agentic_earth_observation.config import is_local_url
from agentic_earth_observation.llm import OLLAMA_KEEP_ALIVE, SEED

LOAD_TIMEOUT_S = 300.0  # loading a model of several gigabytes from disk
REQUEST_TIMEOUT_S = 30.0
UNLOAD_WAIT_S = 30.0  # the unload is asked for at once, but a model in use leaves memory when it is done
POLL_S = 0.2
clock = time.monotonic  # what the warm-up is timed with; a test can replace it without touching asyncio's
WARM_UP_TOKENS = 8  # the warm-up only needs the model to run once; its answer is not read
# About 400 tokens of text that names nothing of the golden set: what Ollama caches from it no task will ask.
WARM_UP_TEXT = " ".join(
    [
        "The orchard by the old stone wall had three rows of apple trees, and the gardener walked them "
        "each morning."
    ]
    * 20
)


class ModelServerError(Exception):
    """The model server cannot be used as the run needs it. The message says what to do."""


class LoadedModel(BaseModel):
    """A model as the server holds it in memory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    digest: str | None = None  # the build of the model's files; an API model has none
    context_length: int | None = None


class ModelServer(Protocol):
    """What a run needs from the server of the model under evaluation."""

    async def version(self) -> str | None: ...

    async def has(self, model: str) -> bool: ...

    async def prepare(self, model: str, context: int) -> LoadedModel:
        """Make `model` the only loaded model, freshly loaded with this context, and say which build it is."""
        ...

    async def warm_up(self, model: str, context: int) -> float:
        """Run the loaded model once on unrelated text, and say how many seconds that took."""
        ...

    async def unload(self, model: str) -> None: ...


class _Version(BaseModel):
    version: str


class _Model(BaseModel):
    model: str
    digest: str | None = None
    context_length: int | None = None


class _Models(BaseModel):
    models: list[_Model]


class _Generated(BaseModel):
    done: bool


class OllamaServer:
    """Ollama at `url`. `http` lets tests answer in place of the server; it is closed with the server."""

    def __init__(self, url: str, *, http: httpx.AsyncClient | None = None) -> None:
        self.url = url.rstrip("/")
        self._http = http or httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S, trust_env=not is_local_url(url))

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None
    ) -> None:
        await self._http.aclose()

    async def version(self) -> str | None:
        return (await self._ask("GET", "/api/version", _Version)).version

    async def has(self, model: str) -> bool:
        return any(m.model == model for m in (await self._ask("GET", "/api/tags", _Models)).models)

    async def prepare(self, model: str, context: int) -> LoadedModel:
        for held in await self._loaded():  # the model itself too: its prompt cache must not carry over
            await self.unload(held.model)
        await self._until_empty()
        await self._ask(
            "POST",
            "/api/generate",
            _Generated,
            json={
                "model": model,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": {"num_ctx": context},
                "stream": False,  # one JSON answer, whatever a future version streams for an empty prompt
            },
            timeout=LOAD_TIMEOUT_S,
        )
        return await self._held_as_loaded(model, context, "after loading it")

    async def warm_up(self, model: str, context: int) -> float:
        started = clock()
        await self._ask(
            "POST",
            "/api/chat",
            _Generated,
            json={
                "model": model,
                "messages": [{"role": "user", "content": WARM_UP_TEXT}],
                "stream": False,
                "think": False,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": {
                    "num_ctx": context,
                    "temperature": 0,
                    "seed": SEED,
                    "num_predict": WARM_UP_TOKENS,
                },
            },
            timeout=LOAD_TIMEOUT_S,
        )
        seconds = clock() - started
        await self._held_as_loaded(
            model, context, "after the warm-up"
        )  # another context would have reloaded it
        return seconds

    async def _held_as_loaded(self, model: str, context: int, when: str) -> LoadedModel:
        """The model as the server holds it, or ModelServerError when it does not, or with another context."""
        loaded = [m for m in await self._loaded() if m.model == model]
        if not loaded:
            raise ModelServerError(f"Ollama at {self.url} did not keep {model} loaded {when}")
        if loaded[0].context_length != context:
            raise ModelServerError(
                f"Ollama loaded {model} with a context of {loaded[0].context_length}, not {context}: "
                "the agents' requests would reload it"
            )
        return LoadedModel(model=model, digest=loaded[0].digest, context_length=loaded[0].context_length)

    async def unload(self, model: str) -> None:
        await self._ask(
            "POST", "/api/generate", _Generated, json={"model": model, "keep_alive": 0, "stream": False}
        )

    async def _loaded(self) -> list[_Model]:
        return (await self._ask("GET", "/api/ps", _Models)).models

    async def _until_empty(self) -> None:
        """Wait until the server holds no model: a run measures one model at a time, freshly loaded."""
        give_up = time.monotonic() + UNLOAD_WAIT_S
        while others := [m.model for m in await self._loaded()]:
            if time.monotonic() >= give_up:
                raise ModelServerError(
                    f"Ollama still holds {', '.join(others)} after {UNLOAD_WAIT_S:g} s: "
                    "is something else using it? Stop that first"
                )
            await asyncio.sleep(POLL_S)

    async def _ask[T: BaseModel](
        self,
        method: str,
        path: str,
        shape: type[T],
        **request: Any,  # noqa: ANN401 - httpx's own arguments
    ) -> T:
        """Ollama's answer to one request, validated as `shape`."""
        try:
            response = await self._http.request(method, self.url + path, **request)
            response.raise_for_status()
            return shape.model_validate(response.json())
        except httpx.HTTPStatusError as err:
            said = " ".join(err.response.text.split())[:300]
            raise ModelServerError(f"Ollama answered {path} with {err.response.status_code}: {said}") from err
        except httpx.HTTPError as err:
            raise ModelServerError(
                f"Ollama at {self.url} cannot be reached ({type(err).__name__}): is it running?"
            ) from err
        except ValidationError as err:  # before ValueError: pydantic's error is one
            raise ModelServerError(f"Ollama's answer to {path} is not understood: {err}") from err
        except ValueError as err:  # not JSON
            raise ModelServerError(f"Ollama's answer to {path} is not JSON: {err}") from err
