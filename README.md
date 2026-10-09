# agentic-earth-observation

A small agentic Earth Observation assistant. You ask for satellite scenes in plain language
("cloud-free Sentinel-2 images of Rome from July 2025"); an orchestrator agent plans, delegates the search
to a retrieval sub-agent, and answers with the scene ids it found. The point of the project is the
evaluation: how often the agents call the right tool with the right arguments, and how fast.

Status: skeleton. Only `evals/scoring.py` and its tests are implemented.

## What it demonstrates

| Capability | Where |
|---|---|
| Agent orchestration with LangGraph: planning, tool calling, state and memory, multiple agents | `orchestrator/graph.py`: plan, delegate, synthesise, with a checkpointer for follow-up questions |
| An MCP server, and MCP tools consumed by an agent | `catalog_mcp/server.py`: a STAC catalogue exposed as four MCP tools, used by the retrieval agent |
| A2A: agent cards and task delegation between independent agents | `retrieval_agent/a2a_server.py`: the sub-agent published over A2A; the orchestrator is its client |
| A specialised data-retrieval sub-agent | `retrieval_agent/agent.py` |
| Evaluation of tool-use accuracy, task success, robustness and latency | `evals/`: golden tasks, per-argument scoring, bootstrap confidence intervals, model comparison |
| Documentation for extending the system | `docs/add_a_tool.md`: add a new MCP tool in ten minutes |

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

## Milestones

About six working days. Each milestone ends with something that runs.

**M1. Catalogue MCP server (day 1).** Implement the four tools in `catalog_mcp/server.py` with validation at
the boundary, plus unit tests against recorded STAC responses. Done when an MCP inspector can list and call
the tools.

**M2. Retrieval agent (day 2).** LangGraph agent with a model node, a tool node and a step limit; tools loaded
from the MCP server. Done when "Sentinel-2 over Rome, July 2025" returns scene ids from a script.

**M3. A2A and the orchestrator (day 3).** Publish the retrieval agent over A2A with an agent card; build the
orchestrator graph with plan, delegate and synthesise nodes and a checkpointer. Done when a two-turn
conversation ("same area, but November 2024") reuses the earlier place.

**M4. Evaluation harness (day 4).** Grow `data/golden.jsonl` from 5 to about 40 tasks across the categories
below; implement `evals/run_eval.py`. Record every tool call, latency and token usage per run.

| Category | What it tests | Target count |
|---|---|---|
| single_search | right collection, box, dates, cloud filter | 15 |
| memory | follow-up questions that depend on earlier turns | 6 |
| robustness | typos, other languages, vague dates, paraphrases of the same task | 10 |
| out_of_scope | should decline without calling a tool | 5 |
| tool_failure | the catalogue returns an error or nothing; the agent must say so | 4 |

Metrics: tool-selection accuracy, per-argument accuracy, task success (scene ids match a direct STAC query),
fabricated scene ids (must be zero), latency p50 and p95, cost per task.

**M5. Results (day 5).** Run two or three models, three runs each. Write `results/summary.md` with bootstrap
confidence intervals and an honest error analysis: which category fails, and why. Then make one targeted
improvement and re-measure, so the README can show a before and after.

**M6. Release (day 6).** `docs/add_a_tool.md`, a CI workflow (ruff, mypy, pytest), a licence and pinned
dependencies.

## Getting started

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env                  # then fill in the model provider and key
pytest
```

The agent, MCP and A2A libraries change quickly. Read the current documentation of each before writing M1 to
M3, and pin the versions once the first end-to-end run works.
