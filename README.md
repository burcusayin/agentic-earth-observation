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
| Golden set (54 tasks) and turn scoring | `data/golden/`, `src/agentic_earth_observation/evals/` | Done; test split frozen at tag `golden-test-v2` (50 tasks; the 18 of `golden-test-v1` and 32 added) |
| Retrieval agent (LangGraph), served over A2A | `src/agentic_earth_observation/retrieval_agent/` | Done |
| Orchestrator (LangGraph), an A2A client, and `eo-agent chat` | `src/agentic_earth_observation/orchestrator/`, `src/agentic_earth_observation/cli.py` | Done |
| Evaluation runner, and `eo-agent eval` | `src/agentic_earth_observation/evals/run_eval.py` | Done |
| The report, and `eo-agent report` | `src/agentic_earth_observation/evals/report.py`, `stats.py` | Done |
| Results on the test split: two local models, 3 runs each | [`docs/results.md`](docs/results.md), `results/` | Done (2026-11-08) |

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
call. The catalogue speaks STAC, the SpatioTemporal Asset Catalog API for satellite scene metadata. Only
scene metadata is used; no imagery is downloaded. Everything listens on `127.0.0.1` only.

## Evaluation

The golden set has 54 tasks: 4 for development and 50 for testing (18 until tag `golden-test-v2`; the results
below are on those 18).
- **Categories:** single searches, follow-up questions, typos and other languages, relative dates, out-of-scope
  requests including a prompt injection, unknown or misspelt places, radar with a cloud limit, dates before a
  satellite launched, and a search with no results.
- **Frozen first:** the test split, the labelling rules, the scoring code and the gazetteer were frozen (git tag
  `golden-test-v1`) before any agent prompt was written. A test checks that the code the agents run, the text
  they read and the scripts do not mention test-split places, questions or dates; what it cannot see is recorded in
  [docs/decisions.md](docs/decisions.md) (2026-11-07).
- **Grown before the first improvement:** after the results below, the test split grew from 18 to 50 tasks and
  was frozen again (git tag `golden-test-v2`), with the rules and the scoring unchanged, so that a change to the
  system is measured on tasks that had not been run when it was chosen. The new tasks were written and checked
  by separate sessions of an AI coding assistant and have not been read by a person
  ([docs/decisions.md](docs/decisions.md), 2026-11-12). No run on the 50 tasks exists yet.
- **Tuned on dev only:** a prompt change is kept only if it raises pass rates on dev and synthetic cases
  (`scripts/probe_models.py`, repeated runs on both local models) and lowers none. The numbers are in
  [docs/decisions.md](docs/decisions.md).
- **Scoring:** each turn is scored on:
  - the route;
  - tool order;
  - each search argument against an accepted range;
  - grounding (every cited scene id came from a tool);
  - the final status.

  Results are reported as counts with confidence intervals, and differences that a small test set cannot
  support are not claimed.

### Results

Both local models, three runs each over the 18 test tasks (2026-11-08, from a clean checkout):

| | qwen3.5:9b | ministral-3:8b |
|---|---|---|
| Passed tasks (95% interval) | 17 of 18 (15 to 18) | 15 of 18 (12 to 18) |
| Same result in all 3 runs | every task | every task |
| Answers citing a scene no tool returned | 0 | 0 |
| Seconds per task, median | 16.3 | 10.4 |

The models differ on 4 tasks (3 passed by qwen3.5:9b only, 1 by ministral-3:8b only; exact sign test p = 0.63):
with 18 tasks, the difference does not show that either is better. The task-by-task analysis and the limitations,
among them timings taken on a fanless laptop, are in [docs/results.md](docs/results.md); the full report is
[results/summary-v1.0.md](results/summary-v1.0.md), and every run's records are in `results/runs/`.

The details are in [docs/evaluation.md](docs/evaluation.md), and the design decisions with their reasons in
[docs/decisions.md](docs/decisions.md).

## Getting started

Requires [uv](https://docs.astral.sh/uv/) 0.12.23 or newer and make. Run it from a clone: the code reads
`data/` and writes `results/` in the repository, so it does not work as an installed package. Tested on macOS
and, for the checks, on Ubuntu.

The agents need [Ollama](https://ollama.com) (tested with 0.40.1) and a model. The two under evaluation take
about 8 GB of memory while loaded, on top of what the machine is doing; with less to spare, turns get slow enough
to time out.

```bash
ollama pull qwen3.5:9b       # 7.6 GB, the default in EO_LLM_MODEL
ollama pull ministral-3:8b   # 6.0 GB, the other model under evaluation
```

The MCP Inspector also needs Node.

```bash
make setup        # .venv from uv.lock (uv's own Python 3.13) and the git hooks
make check        # lint, format check, mypy --strict, tests with coverage (no network, no models)
make serve-mcp    # the catalogue MCP server on http://127.0.0.1:8001/mcp
make inspector    # in a second terminal: try the tools by hand in the MCP Inspector
make serve-agent  # in a second terminal: the retrieval agent over A2A on http://127.0.0.1:8002
uv run eo-agent chat   # in a third terminal: talk to the assistant (both servers and Ollama running)
uv run python -m agentic_earth_observation.retrieval_agent "<a question>"   # one question, through MCP
uv run python -m agentic_earth_observation.evals.golden   # validate the golden set and print its hash
make eval-smoke   # the evaluation on the dev split: starts its own servers; needs Ollama and Earth Search, ~2 min
```

The dev split is for trying things: `uv run eo-agent eval --split dev --models qwen3.5:9b --runs 1` writes one
folder per run under `results/raw/`, with its record, every attempt, the tool-call log and the server logs. The
test split runs only with `--for-report`, from a clean checkout, because every run of it is reported.

`uv run eo-agent report --split dev --runs-dir results/raw --allow-dirty` turns those folders into a report in
markdown: every run with why it counts or not, then for each model the passed tasks with 95% intervals (lost
tasks left out, and counted as failed), the comparison of the models, and the secondary measures. When the
runs span more than one setup, it also compares them: each model before and after a change, task by task, with the
session noise against the runs of v1.0 and how often the runs of a setup did the same. It prints task
ids, never a task's text. `make report` writes the report on the test split's runs to `results/summary.md`.
It counts the runs on the current test split only. The runs of v1.0 are on the 18 tasks of `golden-test-v1`, so
their report is kept as `results/summary-v1.0.md`; `make report` at tag `v1.0.1` makes it again.

In the chat, each line continues the conversation: a follow-up such as "and the week after that?" continues
the last search, and an answer to the assistant's own question (which place? which dates? did you mean …?)
completes the request it asked about.

Any MCP client can use the running server. For example, to add it to Claude Code:
`claude mcp add --transport http eo-catalog http://127.0.0.1:8001/mcp`.

`.env.example` lists every setting with its default. The MCP server reads the catalogue, timeout and tool-log
settings; the retrieval agent reads the model and timeout settings as well; the chat reads the model, the
retrieval agent's address and the A2A timeout. If port 8001 or 8002 is busy, `EO_MCP_URL` and `EO_A2A_URL` move
the servers. The Mistral provider is not wired up yet.

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
  - A hook formats every Python file Claude Code writes, with `ruff format`.
  - The deny rules match command patterns, so they are a safeguard, not a sandbox. The approval required for
    every push is the real control.
- **Commit checks ([`.pre-commit-config.yaml`](.pre-commit-config.yaml)):** formatting, lint, types, secret
  scanning (gitleaks) and a check for private files, on every commit. CI runs them too, scans the whole history
  with gitleaks, and checks the names of all tracked files; a list of private strings is checked on the
  maintainer's machine only.
- **Adversarial reviews:** before the test set was frozen, and again before the evaluation runner was finished,
  separate read-only Claude Code sessions reviewed the code, the evaluation and the repository, without being
  told the author's conclusions. Their findings were fixed, or recorded with a reason in
  [`docs/decisions.md`](docs/decisions.md). No other person has reviewed this code yet.

Bug reports and fixes are welcome: see [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`SECURITY.md`](SECURITY.md).

## Roadmap

**Milestone 1, a working system and a first evaluation:**
- [x] Catalogue MCP server, validated and tested against recorded catalogue responses.
- [x] Golden set and scoring, frozen.
- [x] Retrieval agent and orchestrator, connected over A2A.
- [x] Evaluation of two local models (Ollama), three runs each, with latency per task: see
  [docs/results.md](docs/results.md). The two Mistral API models first planned here moved to the follow-up
  milestones (decision of 2026-11-05).

**Follow-up milestones (planned):**
- **Evaluation rigour:**
  - Two Mistral API models next to the local ones, with cost per task.
  - A recorded local catalogue for fully offline runs.
  - Fault injection (timeouts, server errors, rate limits).
  - About 50 tasks: done, 50 tasks frozen at tag `golden-test-v2`; no run on them yet.
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

The code is licensed under the [Apache License 2.0](LICENSE). The data is not, and needs credit, as set out in
[`NOTICE`](NOTICE):
- **`data/gazetteer.json`** is derived from OpenStreetMap: © OpenStreetMap contributors, under the
  [Open Database License](https://opendatacommons.org/licenses/odbl/1-0/).
- **The run records in `results/runs/`** contain modified Copernicus Sentinel data (2024-2025), obtained
  through Earth Search during the runs, and search boxes from the gazetteer.
- **The recorded test fixtures** contain modified Copernicus Sentinel data (2025), obtained through Earth
  Search.
