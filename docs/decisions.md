# Decisions

Dated record of the technical decisions that shape the project, with the reason for each. Add an entry when a
decision is made or reversed; do not rewrite old entries.

## 2026-10-05
- **Scene metadata only, through a public STAC API; no imagery is downloaded.** The project evaluates tool use,
  not image analysis.
- **Place names come from a small local gazetteer.** Evaluation stays deterministic.
- **No web UI, vector database or imagery understanding.** None is needed to practise orchestration, MCP, A2A
  and their evaluation, and each would add a lot of work.

## 2026-10-09
- **Tooling:**
  - Python 3.13, uv (with `uv.lock`), ruff, `mypy --strict`, pytest and pre-commit; the same `make check` runs
    locally and in CI.
  - Licence: Apache-2.0.
- **Pinned versions:** every runtime dependency has an upper bound. `mcp` stays below 2.0 because
  langchain-mcp-adapters 0.3 requires it. a2a-sdk 1.x uses protobuf types; code is written against the installed
  source.
- **Catalogue:**
  - Earth Search v1, with Sentinel-2 L2A from `sentinel-2-c1-l2a`, whose ids stay stable, and Sentinel-1 from
    `sentinel-1-grd`.
  - Cloud cover is filtered with the STAC `query` parameter, because Earth Search silently ignores CQL2
    `filter`, and every result is re-checked.
- **Product names in the tools** are `sentinel-2-l2a` and `sentinel-1-grd`, mapped explicitly to catalogue
  collection ids.
- **Models under evaluation:**
  - Local through Ollama: `qwen3.5:9b` and `ministral-3:8b`.
  - Paid through the Mistral API, for evaluation runs only: Ministral 3 8B and Mistral Small 4.
- **Ollama settings for every request:** `num_ctx=16384` (the default context truncates silently),
  `temperature=0`, a fixed `seed`, thinking off, and one model loaded at a time.
- **Agents are hand-written LangGraph `StateGraph`s** with an explicit step limit. Only tool errors are passed
  back to the model; other failures are infrastructure errors and are excluded from scores.
- **Decline, clarification and failure are detected from structured fields** (the orchestrator's route and the
  result status), not by an LLM judge.

## 2026-10-22
- **Milestone 1 scope:**
  - The ground truth for each task comes from a direct STAC query with the expected arguments, run alongside the
    agent; both sets of ids are stored in the run log.
  - About 20 golden tasks, split into dev and test; the test split is frozen before any prompt is written.
  - Failure cases come from natural situations (unknown place, no results, out of scope) rather than fault
    injection.
  - The rest is listed as follow-up work in the README roadmap.

## 2026-10-30
- **The gazetteer is built from OpenStreetMap** (Nominatim, ten places, names in several languages as aliases),
  replacing the hand-typed boxes. It is ODbL-licensed and credited in `NOTICE`.
- **The package is named `agentic_earth_observation`**, like the repository.
- **Tool arguments are strict.** Unknown argument names are refused rather than dropped; booleans and text are
  not accepted as numbers; dates are `YYYY-MM-DD` only.
- **Catalogue errors never reach a model verbatim.** The model sees a fixed message; the details are logged.
- **Evaluation, before the freeze**, following an independent review:
  - 22 tasks (4 dev, 18 test), with a "no results" task and a misspelt-place task added.
  - Every expected search was checked against Earth Search.
  - The labelling rules are written generically and contain no test answers. A test checks that system code
    never mentions test-split places or questions.
  - Box overlap threshold 0.98 (3-decimal rounding passes, 2-decimal fails). Exact dates. Cloud as accepted
    ranges.
  - The route is observed from behaviour. Status is checked on search turns. No scene may be cited when the
    right answer is a refusal, an error or "none found".
  - Malformed calls and running out of steps count as model failures.
  - Statistics are reported as counts with intervals and exact sign tests; small differences are not
    claimed.
- **What is frozen** at tag `golden-test-v1`: the test split, the gazetteer, the rules and scoring in
  `docs/evaluation.md`, and the evaluation code (hashed into every run record).
