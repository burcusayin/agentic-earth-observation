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
    so the report states grounding as a property of the design, not of the model.
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
  of the local-against-API difference the evaluation measures, and the report states it.

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
    the HTTP timeout: the SDK swallows cancellation, so asyncio.timeout would not bound it. (Corrected on
    2026-11-06: that holds only with the agent in the client's own process.)
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
- **The assistant can take an answer to its own question** (the maintainer's request, which goes beyond Milestone 1's plan).
  Found in the first end-to-end chat: each turn was decided alone, so "Yes" or a place name given in answer
  meant nothing.
  - The orchestrator remembers the question its last turn asked (a clarification, or an unknown place with
    its suggestions) together with the request so far. The next turn reaches both models as REQUEST, THE
    SERVICE ASKED and THE USER ANSWERED, and each prompt says how to read it: a request completed or corrected
    by the answer, with an accepted suggestion as the place the user wrote, unless the answer is a new request.
  - A turn that searches or declines settles the question; a failed turn (the model's or the infrastructure's)
    leaves it open; answers to repeated questions add up, to at most 4,000 characters.
  - Without an open question nothing changes. The frozen evaluation does not score answers to questions.
  - Designed and probed on dev places and invented ones only: the chat that found the gap used a place outside
    the dev split, so that transcript is not used for tuning.
- **The slow turns in the first end-to-end chat came from the machine, not the code.** After pauses, Ollama took
  about 80 s to process about 100 new prompt tokens while the swap was almost full (24.9 of 25.6 GB). Evaluation
  runs start with memory to spare, and their records keep Ollama's own timings, so a slow turn can be told
  apart from a slow model.
- **Both prompts were tuned on the dev split by pass rates** (`scripts/probe_models.py --repeat 5`). The probe's
  8 retrieval and 16 decision cases are dev questions and synthetic ones about dev or invented places, each
  checked automatically. A change was kept only if it raised a case's pass rate on a local model and lowered
  none. It was measured twice, with the repeats back to back and then in five shuffled rounds, and the two runs
  agreed case by case.
  - **Retrieval: the rule for answers moved out of the search steps** into its own paragraph after the argument
    rules. ministral-3:8b went from 25 to 35 of 40 runs: the dev radar question and the before-launch question
    went from 0 to 5 of 5. With the rule among the steps, it answered after `resolve_place` without searching,
    as on 2026-11-02. qwen3.5:9b stayed at 40 of 40.
  - **Decision: the rules became numbered steps, and the first that applies wins,** with a period defined and
    "a place name on its own has no period". qwen3.5:9b went from 70 to 80 of 80 runs, ministral-3:8b from 60
    to 70: both now ask for the period of a place named without one.
  - **Still failing on ministral-3:8b, with either prompt:** the dev German question is not delegated (the old
    prompt asked for its period, the new one declines it); a place given in answer to "which place?" is
    delegated without a period; and an accepted suggestion is looked up again under the misspelt name.
- **At temperature 0 with a fixed seed, a run repeats itself mostly, not always.** In the two tuning runs, every
  probe case gave the same result in all 10 of its runs, whether repeated back to back or in shuffled rounds.
  But an earlier comparison the same day, with the same messages, answered the dev radar question differently
  in two runs within one model load, and passed the before-launch question that failed all 10 runs here. The
  cause is not known. So the retry rule of 2026-11-02 holds with a caveat: a retried task may not repeat the
  first attempt exactly. The evaluation runs each task three times, which shows whether this matters for the
  scores.
- **Requests that try to override the rules are declined, and the probe now checks it.** Four synthetic cases,
  in invented wording about dev places: made-up scene ids, the prompt's own instructions, an override given as
  the answer to a question, and a valid search with an override added. The last is labelled a decline as a
  whole, as `docs/evaluation.md` says for out-of-scope requests; the labelling rules do not name mixed
  requests, so this was the maintainer's choice. With the committed prompts, all four went as expected in 5 of
  5 runs on both local models. Every other case gave the same result as in both tuning runs, apart from one
  qwen3.5:9b turn that ran past the 120 s model timeout while the swap was nearly full: an infrastructure
  failure, not the model's.
- **The orchestrator's decision has a deadline, and the timeouts nest around a whole user turn** (found in the
  review of PR #3).
  - ChatOllama streams every reply, so the model timeout (`EO_MODEL_TIMEOUT_S`, 120 s) bounds only how long
    the model server may stay silent, not a call. "One model call < retrieval turn" (2026-11-02) described a
    limit the code did not enforce. The retrieval turn was bounded by its own deadline; the decision was not.
  - The decision now has a deadline (`EO_DECISION_DEADLINE_S`, 150 s). A decision past it is infrastructure
    ("the decision ran past its deadline"), like a retrieval turn past its deadline.
  - A user turn is the decision, then the A2A call, so the settings now require decision deadline + A2A <
    harness, and the harness timeout rises from 360 to 480 s. The silence limit must stay below both
    deadlines.
- **`.env` is read from the repository, wherever a process starts.** It was read from the working folder, so a
  server or the chat started elsewhere would have run on the defaults without a warning, while the tool-call
  log was already anchored to the repository.
- **The STAC client honours Retry-After again, up to 10 s** (review of PR #3). On 2026-11-02 it ignored the
  header, so that a 429 asking for an hour could not block a worker. But then a rate-limited request was
  retried after 0, 1 and 2 s, whatever pause the public service had asked for. urllib3's own cap
  (`retry_after_max`, kept on every retry) now limits the pause to 10 s. Without the header the backoff is
  unchanged, and the 90 s STAC deadline still bounds the whole call.

## 2026-11-04
- **The review of PR #3.** An adversarial review raised ten findings; each was checked against the code, the
  installed libraries or a reproduction before anything changed.
  - **Fixed:** the decision's deadline, `.env` and Retry-After (entries of 2026-11-03); the chat's refusals
    (below); the trace-id rule now described once (`TRACE_ID_RULE`); one request built for both models; and a
    `noqa` given its reason.
  - **The chat ends on a bug inside a turn.** It took any `ValueError` for a refused line, and pydantic's
    `ValidationError` is one, so a grounding refusal or a graph bug was shown as a line it cannot take.
    `run_turn` now raises `TurnRefused` for the lines it refuses, and only those are caught.
  - **Not changed, with the evidence:**
    - A deadline arriving inside an exception group: anyio 4.15.1 never puts a `CancelledError` into a
      group, the MCP client catches its own stream errors, and a real bug raised while the deadline cancels a
      turn is raised by design, as its test states.
    - `read_tool_calls` reads the whole log for each lookup: a Milestone 1 run takes seconds. The evaluation
      runner will keep one log per run.
    - The MCP server lists its four tools on every call: the alternative is FastMCP's private tool manager.

## 2026-11-05
- **The evaluation runner starts the MCP server and the retrieval agent itself, one pair per run, and stops
  them at the end.**
  - The retrieval agent's model is fixed when its server starts, and so is the file of the tool-call log. Every
    run therefore needs both servers restarted with the right settings. Done by hand, a forgotten restart
    gives a run whose two agents use different models, or two runs in one log, and the run record would not
    show it.
  - Started by the runner from the same checkout, the servers are also what the record's commit and hashes
    describe.
  - The runner refuses to start while something answers on either server's port, and stops a run in which the
    retrieval agent reports another model than the run's.
- **A run is one model's pass over one split, with its own folder:** `run.json` (the run record),
  `attempts.jsonl` (one record per attempt of a task, written as each attempt ends, so a run cut short keeps
  what it finished) and `tool_calls.jsonl` (the MCP server's log of that run only).
- **A bug in our own system during a run makes the attempt `errored`.** This covers a retrieval agent that
  fails or rejects a request (2026-11-03) and any unexpected error in a turn.
  - The attempt is recorded with the error and is not retried. The task counts as failed, and is listed apart
    from model failures and infrastructure.
  - The run goes on, and ends as aborted at the third errored task: by then the system is broken, not a task.
  - A run with an errored task is marked in the report. After a fix the runs are repeated, and both are listed.
  - This is the "tolerate up to N errors" mode of the Inspect AI evaluation framework, whose default is to
    stop at the first error and resume later (https://inspect.aisi.org.uk/handling-errors.html). The runner
    has no resume, so stopping at once would lose the rest of an unattended run.
- **A delegated turn with no tool call in the log is scored as it is when the model called no tool.** It then
  fails the required tools, as a model's own failure should; counting every such turn as infrastructure would
  retry and exclude it. Only when the agent's result shows that a tool ran (`ok`, `empty`, `unknown_place` or
  `error`) and the log has no call for it do the agent and the log disagree, and that attempt is `errored`.
- **The two Mistral API models are run last, after the local runs.** Until they have run, the first
  release reports the local models, and the API runs move to the follow-up milestones.
- **What a run record's dirty flag covers** (`evals/records.py`).
  - Tracked files with changes, staged or not, and untracked files that are not ignored.
  - Not `results/`, which holds what runs write: otherwise the first run of an evaluation would make every later
    one dirty.
  - git is asked for untracked files explicitly, because a user's git configuration can hide them, and the
    checkout must be the top of its own repository, so that a folder inside another repository is never
    described by that repository's commit.
- **Run records keep named settings only,** never the API key or a path on the machine. A test fails when a
  setting is neither recorded nor left out for a stated reason.

## 2026-11-06
- **How the runner counts a turn** (`evals/outcome.py`). The first rule that applies decides:

  | # | What happened | The attempt is |
  |---|---|---|
  | 1 | the turn raised an error | `errored`: not retried, the task counts as failed |
  | 2 | the turn reports an infrastructure failure | `infrastructure`: the task is tried again, at most twice |
  | 3 | the log has a call that the catalogue failed | `infrastructure` |
  | 4 | the turn's output and the log disagree (next entry) | `errored` |
  | 5 | none of these | scored, and the next turn runs |

  - Infrastructure is decided before the output and the log are compared. The plan of 2026-11-05 put a bug
    first; but after an infrastructure failure the agent's result is a placeholder that reports none of the
    calls it made, so the comparison would take a timeout for a bug.
  - A catalogue failure counts wherever it is in the turn, even when a later search succeeded (2026-11-02).
- **What a turn's output and the log must agree on,** when the turn reports no infrastructure failure:

  | The output says | The log has |
  |---|---|
  | the turn was not delegated | no call |
  | `ok` or `empty` | a successful `search_scenes`, the last of which returned the result's scene ids in the same order and has the result's search arguments |
  | any other status | no successful `search_scenes` |
  | `unknown_place` | an answered `resolve_place` |
  | `error` | a failed call |
  | `not_searched` | nothing more is required: the turn is scored as it is (2026-11-05) |
  | anything | no failed call that the server cannot explain (`other`), and no cancelled call |

  - A refused call (`invalid_arguments`) is the model's doing, and never a disagreement.
  - The logged arguments are read as the agent reads its own, so 20 and 20.0 are the same cloud limit. The
    place name is the agent's addition, which no call carries, and is not compared.
  - `other` means a bug in the server or in a tool. A cancelled call comes from a client that gave up, which
    a turn reports as infrastructure; in a turn that reports none, it is unexplained.
  - Checked against whole turns through the real retrieval loop, the MCP server and its log, with a scripted
    model: no ordinary turn is taken for a disagreement, including two searches sent at once of which the
    first finishes last.
- **Tool calls recorded under none of the run's trace ids change no attempt.** They cannot be tied to a task,
  so excluding or failing a task for them would be a guess. The runner counts them in the run record, and the
  report marks such a run.
- **The run's servers** (`evals/servers.py`), from measurements of the real servers started as child processes
  (no model, no catalogue search):
  - **Nothing may answer on a server's port before the run starts it.** With an older server on the port, a
    new one looks ready within 0.01 s, because the older one answers, and exits a moment later. So the runner
    refuses to start while either port answers, and a server counts as ready only while its process runs.
    The runner checks that both still run before every attempt and before it calls a run complete.
  - **The servers run with the runner's settings:** every setting with a value is stated in their environment,
    where it beats `.env`. The API key is never copied out of the settings: the servers read it where the
    runner did.
  - **Ready means:** the MCP server answers `initialize` and `list_tools` (the run record's tool hash is made
    from this answer), and the retrieval agent's card can be read, each as the agents themselves do it.
  - **A stop is SIGTERM, 5 s, then SIGKILL,** the agent first. An idle server stops within a second; the
    retrieval agent with a turn in flight does not stop on SIGTERM at all (uvicorn waits for the turn).
  - **A server found dead ends the run as `aborted`,** with the server and how it ended as the reason, rather
    than restarting it and going on. To a turn, a dead server is an infrastructure failure, so without this
    every remaining task would be tried three times and excluded, and the run would end complete with almost
    nothing scored. A crashed server is a bug in our own system, as the third errored task is; the rest of one
    run of 18 tasks costs little to repeat, while a task scored after a crash of unknown cause can move the
    result by 5.6 points. The paid runs are the one place where this costs money: to be looked at again
    before any of them.
  - The reason kept in the run record names the server's log file in the run's folder, never a path of this
    machine.
- **The harness timeout** (to be built into the runner): a turn runs as a task, and the runner waits for it
  with `asyncio.wait(timeout=EO_HARNESS_TIMEOUT_S)`, which returns on time whatever the task does; never with
  `asyncio.timeout` or `wait_for`, which return late when a call swallows cancellation (6 s under a 1 s limit
  when measured). On a timeout the attempt is `infrastructure` ("the turn ran past the harness timeout"),
  and the retrieval agent is restarted before the next attempt: it goes on with the abandoned turn (the A2A
  SDK continues a disconnected client's task in the background) and holds the model meanwhile. A new agent is
  ready in under a second.
- **Correction of 2026-11-03, "the SDK swallows cancellation":** that holds only when the agent runs in the
  client's own process, as in the unit tests, where a cancelled call ends as a cancelled task and so as
  `RetrievalAgentError`. Over a real connection the client's call is cancelled as any other: `asyncio.timeout(2)`
  raised at 2.0 s, and the client stayed usable. The docstring of `orchestrator/client.py` says so now. A call
  is still bounded by the HTTP timeout; the harness timeout above is the second bound.

## 2026-11-07
An adversarial review of the runner, the agents and the evaluation, by separate read-only Claude Code sessions,
and the first real runs on the dev split. What came of them:

- **The retrieval agent is replaced after every attempt lost to the infrastructure, and after an errored one,**
  not only after a harness timeout (revising 2026-11-06). An A2A timeout leaves it at the abandoned turn just the
  same, and a new agent is ready in under a second.
- **A task is tried again only after a 30 s pause,** so that a short outage does not take all three attempts.
- **The third task lost to the infrastructure aborts the run,** in a row or not. A run ended `complete` with almost
  everything excluded, which is what 2026-11-06 refused for a dead server. The final record now holds how many
  tasks passed, failed, errored and were lost, and the command prints them.
- **An evaluation stops at its first aborted run.** The next runs would meet the same broken system; a person
  looks first. It is a choice about time, easy to reverse.
- **The test split runs only for the report:** from a clean checkout, in 3 runs per model and the one fixed order,
  and only when `results/runs/` holds nothing uncommitted, so that every such run stays on record. A run that was
  never committed can still be deleted by hand; these rules make that a deliberate act, not an accident.
- **The frozen files are checked mechanically:** tests and a run for the report compare them with hashes taken at
  tag `golden-test-v1`.
- **A checkout that changes during an evaluation aborts it.** The servers import the checkout at each start, the
  runner keeps the code it began with, and the record would name the wrong commit.
- **A catalogue that returns nothing aborts the run.** `score_turn` takes the run's own search as the truth about
  whether scenes exist, so an empty catalogue, or a client that drops everything, would pass every search as "none
  found", with the oracle agreeing. The golden set has one task without results; a second aborts.
- **What the oracle query records:** no precision for a turn that cites nothing (the share of nothing is
  undefined, and `retrieval_metrics` would answer 0 or 1); recall over what the tools returned, as
  `docs/evaluation.md` says, so a model failure after a good search is not recall 0; no metrics on an attempt that
  is not scored; and a catalogue refusing the expected search is a bug that aborts the run.
- **A record never names a path of this machine.** Error messages can, so `run.json` and `attempts.jsonl` are
  scrubbed as they are written, and a password in a recorded URL is left out. Server logs stay out of git.
- **The catalogue client goes to the configured catalogue only** (`catalog/stac.py`). pystac-client follows the links
  in a catalogue's answers with the method, headers and body they name, and our local servers trust a caller
  without an Origin header, so a hostile answer could have forged tool-log records or started a turn. A link to
  another origin and any redirect are refused, and fail the search.
- **A bug of ours no longer counts as a failure of the catalogue.** Building the search stands outside the error
  handling, and a 4xx (other than 408 and 429) is `CatalogRefused`, error kind `other`: errored, not excluded.
- **Scene ids have the shape of a Sentinel product id, and cloud cover is a real number.** The id is the one text
  a model and the answer repeat; now it cannot carry a sentence, so no other text from the catalogue reaches a
  model, and the retrieval prompt stays as it was tuned. Checked against the ids in the recorded responses and in
  real runs.
- **Ollama's own trouble is infrastructure** (revising 2026-11-02 for these cases): its messages for a runner that
  died, did not start, or a model too large for memory, read from the binary of version 0.40.1. On a machine
  short of memory these would otherwise have lowered a model's score. Every other 500 stays the model's failure,
  and a parser error stays one even when the text it quotes holds those words.
- **Clients of a server on this machine ignore proxy settings.** With `HTTP_PROXY` set, requests to 127.0.0.1 went
  to the proxy, the user's words included.
- **A reply may ask for at most 8 tool calls at once;** more fails the turn as `step_limit`, with none of them
  run. They would all start together, and each search reaches a public service. A turn needs two or three.
- **The search of a failed turn is not what a follow-up continues:** the user was told that the turn failed.
- **`EO_MCP_URL` names the MCP server's port,** as `EO_A2A_URL` does the agent's, so a stranger with port 8001 busy
  can move it.
- **Run records keep Ollama's own timings** (promised on 2026-11-03): model time and load time per turn. On the
  dev runs no turn spent more than 0.15 s loading, so loading the model before the first task does what it should.
- **The commit checks in CI were emptier than they read:** both the gitleaks hook and the private-files hook look at
  staged changes, and CI has none. CI now scans the whole history with gitleaks, and the private-files hook checks
  every tracked name when nothing is staged. The list of private strings exists only on the maintainer's machine.
- **Left as they are, with the reason:**
  - An MCP answer the client cannot read waits for the timeout and counts as infrastructure (2026-11-02). It takes
    a garbled answer from our own server, and no such thing has been seen.
  - A turn with a model failure and a catalogue failure counts as infrastructure: the order of 2026-11-06, because
    the catalogue failure may have caused the other.
  - The A2A server keeps finished tasks, and has no limit on a request's size. It is local, single-user, and
    restarted for every run.
  - A catalogue that drips an answer keeps a worker thread after the deadline: the tool still answers on time.
  - The leakage test skips `evals/` and `scripts/build_gazetteer.py` (which `docs/evaluation.md`, frozen, does not
    say), the first word of a sentence, and words in another case, other than gazetteer names. Widening it would
    give failures that could only be understood by reading the test split. Its failures now print a count.
  - Two latent defects in the frozen files: `golden.py` requires `search_scenes` for any expected search, so another
    accepted route could never pass, and `scoring.py` accepts any status on a turn with no search and no
    `outcomes`. A check that prints only counts finds no such turn in either split, so no score changes.
