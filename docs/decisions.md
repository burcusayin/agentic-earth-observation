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

## 2026-10-31
- **Ollama seed is 42** for every evaluation request (with `num_ctx=16384` and `temperature=0`, as decided on
  2026-10-09).
- **CI also runs the commit checks** (including the gitleaks secret scan), and the uv version is pinned in CI.
- **Commit messages are checked** for co-author lines and private strings by a commit-msg hook.
- **Findings from the model probe** (`scripts/probe_models.py`; dev questions only; both local models):
  - Both call tools natively, resolve the place first, copy the box exactly, and handle the radar question
    correctly.
  - qwen3.5:9b sent the optional cloud limit as text ("20") when the schema said "number or null". It
    repeated that even after an actionable error message, but sent a number once the schema advertised a
    plain optional number. **The cloud limit is therefore advertised as a plain number**, and validation is
    unchanged.
  - Wrong number types now get an error message that says how to fix the call.
  - **Routing uses structured output through tool calling (`function_calling`).** Ollama answers 501 for
    `json_schema` with qwen3.5:9b, and ministral-3:8b returned a self-contradictory decision with it.
- **The retrieval agent is given `resolve_place`, `search_scenes` and `list_products`, but not `get_scene`.**
  No task needs it and five forbid it, so offering it would only cost steps and prompt tokens.
- **After `resolve_place` reports an unknown place, the agent is not forced to stop.** The clarify tasks then
  measure whether the model asks or guesses.

## 2026-11-02
- **The retrieval prompt names `search_scenes` in its steps.** With the first full prompt, ministral-3:8b
  answered straight after `resolve_place` on both dev search questions and claimed a search it had not made: it
  read "when the search returns" as the place lookup's reply. "Once you have the bbox, call search_scenes" and
  "when search_scenes returns" fixed both. qwen3.5:9b's calls did not change. Checked on the dev questions, the
  dev follow-up and a made-up place only.
- **Text is never accepted for a list or a number, even when it would parse.** FastMCP parses such text as JSON
  before validating it, so a box sent as `"[...]"` was searched while the record kept the text, and the scoring
  would have failed a search that ran correctly. The server now refuses it with a message that says how to send
  the value, as already decided for numbers on 2026-10-30.
- **The tool-call record is ordered and complete.** It keeps the client's request id, because calls sent
  together can finish in any order. It also records cancelled calls and names unknown tools. The server checks
  at start-up that the log can be written; a later write failure reaches the model only as a fixed message,
  never with a local path, and never hides the tool's own error.
- **Each catalogue call has a hard deadline** (`EO_STAC_DEADLINE_S`, 90 s, retries included). A missed deadline
  is a catalogue error, so it counts as infrastructure. This replaces an estimate of the worst case, which could
  not cover slow reads or `Retry-After` waits. The STAC client now ignores `Retry-After`, which urllib3 would
  otherwise honour for hours. The timeouts nest: one STAC request < STAC deadline < MCP < retrieval turn < A2A <
  harness, and one model call < retrieval turn.
- **Settings are strict.**
  - An empty variable means the default, and a misspelt `EO_` name is refused, whether it comes from the
    environment or from `.env`.
  - Errors never show a value, so a misspelt key cannot leak a secret.
  - A relative tool-log path starts at the repository root, and the log cannot be turned off from the
    environment.
  - At least 4 model calls per turn: place, search, one retry and the reply.
- **What the run records hash.**
  - `prompts.py` is hashed as a whole, so the request format (`REQUEST:`, `PREVIOUS SEARCH:`) counts as well as
    the system prompt.
  - The tool descriptions are hashed from the tools as the agent receives them: name, description and input
    schema, including each parameter's description.
- **Model failures and infrastructure, clarified.** The 2026-10-09 entry says that "other failures are
  infrastructure errors". That means failures outside the model's control: the catalogue, the network or a
  server. The model's own failures fail the task, as decided on 2026-10-30: running out of steps, malformed tool
  calls and unparseable output. How a turn timeout counts will be decided with the agent.
- **The review read the test split.** One reviewer read the test split to check the prompt against the
  labelling rules. One finding came only from test questions and was deliberately not acted on; tuning stays on
  the dev split.
- **The leakage test is stricter.** It also checks the capitalised words inside test questions (so places with
  no gazetteer entry are covered) and the test-split dates, and it scans `scripts/` as well as `src/`.
- **The prompt now states two more labelling rules, and the PREVIOUS SEARCH line names the place.**
  - The rules: pass the place name as written; don't search when the place is unknown, even if a suggestion
    looks right; after an error, correct how the call is written, never what the user asked for.
  - A reviewer had predicted that models would move the dates after a "before launch" error, or correct a
    misspelt place themselves. The probe's synthetic cases on dev places showed neither with the two local
    models, so this is not a fix.
  - The rules are stated because the scoring checks them (`docs/evaluation.md`), so that no model, including
    the API models still to come, is scored on a rule it was not told.
  - Step 3 is unchanged: it carries the ministral-3:8b fix above.
  - After the change, all seven probe cases still gave the expected calls on both local models.
- **How the runner classifies each turn**, within the frozen rules of `docs/evaluation.md`.
  - **Status mapping**, from agent status to scored status:

    | Agent status | Scored status |
    |---|---|
    | `ok` | `ok` |
    | `empty` | `empty` |
    | `error` | `error` |
    | `unknown_place` | `clarified` |
    | `not_searched` | `declined` |
  - **Model failures fail the task**, whatever the trace scores. There are three: running out of steps, a
    malformed tool call (unparseable, or naming a tool that does not exist), and unparseable output. A refused
    argument is not a failure on its own: the model may correct it, and the last search is the one scored.
  - **Infrastructure:**
    - what counts: a catalogue error or a missed catalogue deadline in the turn; a server (MCP, A2A, Ollama or
      the Mistral API) not answering within its timeout; and a turn that runs past its deadline. The frozen
      rules list the model failures, and a timeout is not one of them, while the step limit already catches a
      model that loops.
    - what happens: the task is retried with a fresh trace id, at most twice (at temperature 0 a run repeats
      itself), excluded from the scores, and counted in the report next to each model's latency.
  - **Grounding holds by construction:** answers are built from templates and cite only ids the tools returned,
    so the write-up reports grounding as a property of the design, not of the model.
- **How one retrieval turn becomes a result** (`run_retrieval` and `result_from_turn` in `agent.py`).
  - **The status comes from the tools' results only,** never from the model's text.
    - The last successful search gives `ok` or `empty`, with its scenes and its arguments as sent.
    - Otherwise, a place that `resolve_place` did not know gives `unknown_place`, with the latest such call's
      suggestions, even if a later call found a place and nothing was searched.
    - Otherwise, any failed tool call gives `error`; otherwise `not_searched`.
    - A model failure is kept beside whatever the tools returned.
  - **What counts as infrastructure** refines the list in "How the runner classifies each turn" above:
    - the turn's deadline, when it is what ended the turn;
    - a timeout, a connection that cannot be made or is cut, the MCP client giving up waiting (each MCP
      request is bounded by `EO_MCP_TIMEOUT_S`, so an answer the client cannot read fails in time too), and a
      model server that says it is busy (Ollama's 503).
  - **Everything else is raised,** because infrastructure is excluded from the scores and a bug counted as
    infrastructure would shrink them unseen:
    - an HTTP error from our own MCP server, which is a bug in it;
    - a wrong MCP URL, which the MCP client reports as "Session terminated";
    - a model the model server does not have (404), and any other error.
  - **Catalogue errors reach the model as tool errors,** so the turn's status is `error`. The evaluation runner
    counts them as infrastructure from the log's `error_kind`, because the model and the agent see only a fixed
    message.
  - **Usage is counted per model call as it happens,** so a turn that fails part-way still reports the calls and
    tokens it spent. `seconds` is the whole turn, tool calls included.
  - **A search names the place whose box it used:** one that `resolve_place` found in the turn, or the previous
    search's place when a follow-up reuses its box. A box agreeing to 0.001° counts, so one copied to 3
    decimals does. When no tool returned the box (the model typed it), the place is left empty and the PREVIOUS
    SEARCH line leaves it out.
- **A reply that Ollama cannot read is the model's failure.**
  - Both local models use Ollama's own parser for tool calls. When it cannot read one (malformed XML for
    qwen3.5:9b; malformed JSON, or a tool it was not given, for ministral-3:8b), Ollama 0.40.1 answers HTTP 500,
    or puts the error inside the stream once output has begun. Checked in Ollama's source for that version.
  - A crash of the model server looks the same. These errors count as `malformed_tool_call` and fail the task,
    with Ollama's message kept in `failure_detail` for the error analysis.
  - The other choice, infrastructure, would retry and exclude the model's own failures and so raise its score
    unseen; a crash counted against the model is at least visible in that message.
- **ChatOllama's repair of list arguments belongs to the serving stack.** When a model writes a list or an object
  as text (`"[9.04, ...]"`), langchain-ollama parses it before the call reaches our server, which then never
  sees the text that it would refuse (decision above). langchain-mistralai does not do this. The repair is part
  of the local-against-API difference the evaluation measures, and the write-up states it.

## 2026-11-03
- **How the retrieval agent is served over A2A** (`retrieval_agent/a2a_server.py`, a2a-sdk 1.2.2). Checked by
  reading the SDK and by running it in-process.
  - **Requests and results travel as JSON text** (a text part with media type `application/json`), each validated
    with its model's `model_validate_json`. A data part carries every number as a double, so `"limit": 3`
    arrives as `3.0`, which the strict models refuse.
  - **The trace id travels in the request's metadata** (`trace_id`), and a request without a valid one is
    refused: the MCP server would otherwise file the turn's calls under no trace.
  - **Every answer is a task**, on the SDK's current request handler. Answering with a message instead leaves
    four asyncio tasks behind per request with that handler.
  - **What a task's final state means:**
    - completed: `run_retrieval` returned a result, an infrastructure failure (`infra_error`) included;
    - rejected: the request could not be read (no valid trace id, not exactly one JSON text part, an invalid
      `RetrievalRequest`), with the reason;
    - failed: a bug or a configuration error. The client is told only the error's type and the details go to
      the server log, because the SDK would otherwise send the error's own text to the client.
  - **Only this machine may start a turn:** the server binds to 127.0.0.1; a Host other than 127.0.0.1 or
    localhost is refused (400), and so is any request with an Origin header (403), so that a web page cannot
    start an agent run. The SDK's client sends no Origin.
- **How the orchestrator calls the retrieval agent** (`orchestrator/client.py`). Checked against the SDK's
  source and by running each failure, over a real socket where it matters.
  - A completed task gives the agent's `RetrievalResult`, which may itself carry an `infra_error`.
  - An agent that cannot be reached, does not answer within `EO_A2A_TIMEOUT_S`, or drops the connection gives
    a result with `infra_error` (for example `A2A ReadTimeout`), the same shape as the agent's own
    infrastructure failures. The SDK wraps these as its own errors with the httpx failure as the cause, which
    is what is checked.
  - A rejected or failed task raises `RetrievalAgentError`, with the agent's reason and the trace id: either is
    a bug, and must never be excluded from the scores as infrastructure. The evaluation runner will record such
    a task as errored, apart from infrastructure and model failures.
  - Anything else is raised as it is: an HTTP error (a refused request, a missing card), a JSON-RPC error, an
    answer that is not JSON, or an answer that breaks the contract (not one JSON result).
  - The request never asks for streaming, so there is always one answer to read, and each call is bounded by
    the HTTP timeout: the SDK swallows cancellation, so asyncio.timeout would not bound it.
- **How the orchestrator handles a turn** (`orchestrator/`). Checked against langgraph 1.2.14 and
  langchain-ollama 1.1 by running them.
  - **The model decides with a `Decision` tool call** (route and reason; a pairing that contradicts itself is
    invalid). It delegates every request for scenes that names a place, known or not, or continues the previous
    search; it clarifies only a missing place or period with no previous search; it declines anything else,
    treating the user's text as data.
  - **A reply without a valid Decision is the model's failure** (`parse_error`), with the reason kept: ChatOllama
    ignores `tool_choice`, so a model may answer in text, which comes back as no decision and no error. So is a
    reply Ollama cannot read. The turn then falls back to a declining reply without searching.
  - **A model server that cannot be reached or is busy is infrastructure**, by the same rule as the retrieval
    agent's, now shared in `failures.py`.
  - **The route is observed:** `delegate` exactly when the retrieval agent was called; otherwise the decision's
    route, or `decline` when there was no decision. The scored status follows the mapping of 2026-11-02.
  - **The reply is a template** (`replies.py`), never the model's text, and `TurnOutput` refuses an answer that
    cites a scene id the retrieval agent's tools did not return.
  - **The conversation memory holds only plain values and JSON text.** LangGraph's checkpointer restores our
    own classes only if each one is registered, warns otherwise, and returns an unregistered one, such as a
    nested enum, silently changed.
