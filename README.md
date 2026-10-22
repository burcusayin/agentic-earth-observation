# agentic-earth-observation

A small agentic Earth Observation assistant. You ask for satellite scenes in plain language
("cloud-free Sentinel-2 images of Rome from July 2025"); an orchestrator agent plans, delegates the search
to a retrieval sub-agent, and answers with the scene ids it found. The point of the project is the
evaluation: how often the agents call the right tool with the right arguments, and how fast.

This is a personal project, built to gain hands-on experience with multi-agent orchestration in LangGraph,
the Model Context Protocol (MCP), agent-to-agent delegation (A2A), and the evaluation of LLM tool use.

Status: skeleton. Only `evals/scoring.py` and its tests are implemented.

## What it demonstrates

| Capability | Where |
|---|---|
| Agent orchestration with LangGraph: planning, tool calling, state and memory, multiple agents | `orchestrator/graph.py`: plan, delegate, synthesise, with a checkpointer for follow-up questions |
| An MCP server, and MCP tools consumed by an agent | `catalog_mcp/server.py`: a STAC catalogue exposed as four MCP tools, used by the retrieval agent |
| A2A: agent cards and task delegation between independent agents | `retrieval_agent/a2a_server.py`: the sub-agent published over A2A; the orchestrator is its client |
| A specialised data-retrieval sub-agent | `retrieval_agent/agent.py` |
| Evaluation of tool-use accuracy, task success, robustness and latency | `evals/`: golden tasks, per-argument scoring, bootstrap confidence intervals, model comparison |

## Architecture

```
user question
     |
orchestrator (LangGraph)  -- plan: answer / delegate / decline
     |  A2A task
retrieval agent (LangGraph, served over A2A)
     |  MCP tool calls
catalog MCP server  -->  public STAC API (Sentinel-1, Sentinel-2 scene metadata)
```

Three processes, so each boundary is a real protocol boundary and not a function call.

## Design decisions

- **Data source:** scene metadata only, through a public STAC API. Default is Earth Search
  (`https://earth-search.aws.element84.com/v1`), which needs no account. The Copernicus Data Space (CDSE) STAC
  endpoint is the official Copernicus catalogue: if its search works without a token, switch `STAC_API_URL`
  to it. No imagery is downloaded.
- **Place names:** resolved from a small local gazetteer (`data/gazetteer.json`) so that evaluation is
  deterministic. The boxes in it are hand-typed placeholders and must be checked.
- **Models:** provider set in `.env`. At least one Mistral model is included in the comparison. Development
  runs against a local model; paid APIs are used only for evaluation runs.
- **Scope:** no web UI, no vector database, no imagery understanding. Each would add a lot of work without
  strengthening what the project sets out to show: orchestration, MCP, A2A and their evaluation.

## Roadmap

### Milestone 1: working system and first evaluation (in progress)

- **Catalogue MCP server:** validated tools to list products, resolve place names, search scenes and read
  one scene, over a public STAC API.
- **Retrieval agent:** a LangGraph agent that uses the MCP tools, with a step limit, published over A2A with an
  agent card.
- **Orchestrator:** a LangGraph graph that plans, delegates to the retrieval agent over A2A, declines
  out-of-scope requests, asks for clarification when a place is unknown, and remembers the previous search.
- **Golden set:** about 25 tasks across single searches, follow-up questions, robustness, out-of-scope requests
  and failure cases. It is split into a dev set and a test set, and the test set is frozen before any prompt
  is written.
- **Evaluation:** two local models (Ollama) and two Mistral API models, three runs each.
  - Scoring: tool choice and per-argument accuracy, task success against a direct STAC query, fabricated scene
    ids (must be zero), latency and cost per task.
  - Bootstrap confidence intervals.

### Follow-up milestones (planned)

- **Evaluation rigour:**
  - A recorded local catalogue, so evaluation runs are offline and fully reproducible.
  - Fault injection (timeouts, server errors, rate limits).
  - A golden set of about 50 tasks.
  - Determinism statistics and an accuracy-versus-cost chart.
  - One targeted improvement, measured before and after.
- **Breadth and polish:**
  - The Copernicus Data Space catalogue as a second backend.
  - Local tracing with Arize Phoenix.
  - Property-based tests.
  - Claude Code skills for common workflows.
  - A demo mode that needs no model and no API key.
  - A short guide to adding a new MCP tool.

## Getting started

Requires [uv](https://docs.astral.sh/uv/) and make.

```bash
make setup     # creates .venv from uv.lock and installs the git hooks
make check     # lint, format check, mypy --strict, tests with coverage
cp .env.example .env   # then set the model provider (and a key for paid models)
```
