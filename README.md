# agentic-earth-observation

A small multi-agent Earth Observation assistant, and an evaluation of how well it uses its tools. You ask for
satellite scenes in plain language ("cloud-free Sentinel-2 images of Milan from October 2025"); an orchestrator
agent decides what to do, delegates the search to a retrieval agent, and answers with the scene ids the tools
returned. The main deliverable is the evaluation: how often the agents call the right tool with the right
arguments, decline what they cannot do, ask when a place is unknown, and never invent a scene.

This is a personal project, built to gain hands-on experience with multi-agent orchestration in LangGraph,
the Model Context Protocol (MCP), agent-to-agent delegation (A2A), and the evaluation of LLM tool use.

## Status

| Component | Where | Status |
|---|---|---|
| Validated domain types (box, dates, cloud cover, limits) | `src/agentic_earth_observation/domain.py` | Done |
| STAC catalogue client (Earth Search) | `src/agentic_earth_observation/catalog/stac.py` | Done |
| Gazetteer: place names to boxes, from OpenStreetMap | `src/agentic_earth_observation/catalog/gazetteer.py` | Done |
| MCP server with four tools | `src/agentic_earth_observation/catalog_mcp/server.py` | Done |
| Golden set (22 tasks) and turn scoring | `data/golden/`, `src/agentic_earth_observation/evals/` | Done; test split frozen at tag `golden-test-v1` |
| Retrieval agent (LangGraph), served over A2A | `src/agentic_earth_observation/retrieval_agent/` | Done |
| Orchestrator (LangGraph), an A2A client | `src/agentic_earth_observation/orchestrator/` | In progress: the A2A client is done |
| Evaluation runner and results | `src/agentic_earth_observation/evals/run_eval.py` | Planned |

## Architecture

```
user question
     |
orchestrator (LangGraph)            decides: delegate / decline / clarify
     |  A2A task
retrieval agent (LangGraph, A2A server)
     |  MCP tool calls: list_products, resolve_place, search_scenes
catalogue MCP server  -->  Earth Search STAC API (Sentinel-1 and Sentinel-2 scene metadata)
```

The three parts run as separate processes, so each boundary is a real protocol boundary rather than a function
call. Only scene metadata is used; no imagery is downloaded. Everything listens on `127.0.0.1` only.

## Evaluation

The golden set has 22 tasks: 4 for development and 18 for testing.
- **Categories:** single searches, follow-up questions, typos and other languages, relative dates, out-of-scope
  requests including a prompt injection, unknown or misspelt places, radar with a cloud limit, dates before a
  satellite launched, and a search with no results.
- **Frozen first:** the test split, the labelling rules, the scoring code and the gazetteer were frozen (git tag
  `golden-test-v1`) before any agent prompt was written. A test checks that the system code and scripts never
  mention test-split places, questions or dates.
- **Scoring:** each turn is scored on:
  - the route;
  - tool order;
  - each search argument against an accepted range;
  - grounding (every cited scene id came from a tool);
  - the final status.

  Results will be reported as counts with confidence intervals, and differences that a small test set cannot
  support will not be claimed.

The details are in [docs/evaluation.md](docs/evaluation.md), and the design decisions with their reasons in
[docs/decisions.md](docs/decisions.md).

## Getting started

Requires [uv](https://docs.astral.sh/uv/) and make; the MCP Inspector also needs Node, and the agents need
[Ollama](https://ollama.com) with the model in `EO_LLM_MODEL` pulled.

```bash
make setup        # .venv from uv.lock (uv's own Python 3.13) and the git hooks
make check        # lint, format check, mypy --strict, tests with coverage (no network, no models)
make serve-mcp    # the catalogue MCP server on http://127.0.0.1:8001/mcp
make inspector    # in a second terminal: try the tools by hand in the MCP Inspector
make serve-agent  # in a second terminal: the retrieval agent over A2A on http://127.0.0.1:8002
uv run eo-agent chat   # in a third terminal: talk to the assistant (both servers and Ollama running)
uv run python -m agentic_earth_observation.retrieval_agent "<a question>"   # one question, through MCP
uv run python -m agentic_earth_observation.evals.golden   # validate the golden set and print its hash
```

Any MCP client can use the running server. For example, to add it to Claude Code:
`claude mcp add --transport http eo-catalog http://127.0.0.1:8001/mcp`.

`.env.example` lists every setting with its default. The MCP server reads the catalogue, timeout and tool-log
settings; the retrieval agent reads the model and timeout settings as well. The Mistral provider is not wired up
yet.

## How this repository is developed

The project is developed with [Claude Code](https://claude.com/claude-code) as a pair programmer. The maintainer
designs, reviews and commits every change, and wrote core pieces such as the validators and the MCP tools. The
rules and safeguards are committed so they can be reviewed:

- **[`CLAUDE.md`](CLAUDE.md):** the project rules for Claude Code:
  - how to work: plan first, small steps, tests with every change;
  - code standards;
  - evaluation integrity: what is frozen, no leakage, report every run;
  - cost: local models first, ask before any paid call;
  - secrets and git.
- **[`.claude/settings.json`](.claude/settings.json):** Claude Code's permissions here.
  - Checks and tests run without asking.
  - Pushing, opening pull requests, changing dependencies, re-recording HTTP fixtures and live tests need
    approval.
  - Reading `.env` and key files, force-pushing, skipping hooks and merging are refused.
  - The deny rules match command patterns, so they are a safeguard, not a sandbox. The approval required for
    every push is the real control.
- **Commit checks ([`.pre-commit-config.yaml`](.pre-commit-config.yaml)):** formatting, lint, types, secret
  scanning (gitleaks) and a check for private files, on every commit and in CI.
- **Independent reviews:** before the test set was frozen, separate review passes examined the code, the
  evaluation and the repository. Their findings were fixed in dedicated commits.

## Roadmap

**Milestone 1, a working system and a first evaluation:**
- [x] Catalogue MCP server, validated and tested against recorded catalogue responses.
- [x] Golden set and scoring, frozen.
- [ ] Retrieval agent and orchestrator, connected over A2A.
- [ ] Evaluation of two local models (Ollama) and two Mistral API models, three runs each, with latency and cost
  per task.

**Follow-up milestones (planned):**
- **Evaluation rigour:**
  - A recorded local catalogue for fully offline runs.
  - Fault injection (timeouts, server errors, rate limits).
  - About 50 tasks.
  - Determinism statistics and an accuracy-versus-cost chart.
  - One targeted improvement, measured before and after.
- **Breadth and polish:**
  - The Copernicus Data Space catalogue as a second backend.
  - Local tracing with Arize Phoenix.
  - Property-based tests.
  - Claude Code skills for common workflows.
  - A demo mode that needs no model or key.
  - A guide to adding an MCP tool.

## Licence and data

The code is licensed under the [Apache License 2.0](LICENSE). The data needs credit, as set out in
[`NOTICE`](NOTICE):
- **`data/gazetteer.json`** is derived from OpenStreetMap: © OpenStreetMap contributors, under the
  [Open Database License](https://opendatacommons.org/licenses/odbl/1-0/).
- **The recorded test fixtures** contain modified Copernicus Sentinel data (2025), obtained through Earth
  Search.
