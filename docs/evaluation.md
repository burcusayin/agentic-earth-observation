# Evaluation design

How the agents are evaluated: the golden set, the labelling rules, the scoring, and what is frozen. The golden
test split and this document are frozen (git tag `golden-test-v1`) before any agent prompt is written. The
MCP tool descriptions already exist; they are checked automatically for test-split content (see
"No leakage").

## What is measured

Given a question in plain language, does the system call the right tools with the right arguments, report what
it found, decline what it cannot do, ask when a place is unknown, and cite only scenes that a tool returned?

## Golden set

22 tasks in `data/golden/dev.jsonl` (4) and `data/golden/test.jsonl` (18), one JSON object per line.

| Category | Dev | Test | Purpose |
|---|---|---|---|
| `single_search` | 1 | 5 | One question, one search: product, place, dates, cloud limit, limit |
| `follow_up` | 1 | 2 | Two turns; the second changes one thing and keeps the rest |
| `robustness` | 1 | 3 | Typos in the wording, other languages, relative dates |
| `out_of_scope` | 1 | 2 | Requests the system must decline without searching, including a prompt injection |
| `clarify` | 0 | 3 | Places the gazetteer does not know, including a misspelt one: the system must ask |
| `edge` | 0 | 3 | Radar with a cloud limit, dates before a satellite launched, a search with no results |

- **Split.** Prompts and tool descriptions are tuned on the dev split only. The test split is only measured.
  Dev and test use different places (checked by a test).
- **Live check.** Every expected search was run against Earth Search on 2026-10-30; each task's notes record how
  many scenes it returned. Only the "no results" task returns none, by design.

## Labelling rules

These rules decide the expected answer of every task. The agents' prompts may state the same conventions, so
the system is scored on following stated rules, not on guessing ours.

- **Reference date.** "Today" is **2025-12-15**. Relative dates are resolved against it.
  - "Last month" is the previous calendar month.
  - "Last <season>" is the most recent complete meteorological season before today: spring is March to May,
    summer June to August, autumn September to November, winter December to February.
- **Periods.** A month is the whole calendar month. "The first week of <month>" is days 1 to 7. A range between
  two named days includes both of them. A single named day is a range of one day.
- **Sensor.** `sentinel-1-grd` when the request says radar, SAR, Sentinel-1 or "through clouds"; otherwise
  `sentinel-2-l2a` (including when only "images" or "satellite images" is said).
- **Clouds** (Sentinel-2 only). Wording in any language counts by its meaning.

  | Meaning | Accepted `max_cloud_cover` |
  |---|---|
  | no clouds: "cloud-free", "clear", "senza nuvole", "ohne Wolken" | any value from 0 to 10 |
  | few clouds: "mostly clear", "little cloud", "low cloud" | any value from 0 to 20 |
  | nothing said about clouds | the argument must be absent |

  On radar, cloud wording is ignored: search Sentinel-1 without a cloud limit, without asking. Saying that the
  limit does not apply is good practice.
- **Places.** Bounding boxes come from `resolve_place`, never from the model's memory. A place the gazetteer does
  not know, even an obvious misspelling of a known one, is clarified by asking the user (the suggestions may be
  offered); it is never searched.
- **Limit.** Scored only when the question asks for a number of scenes. Otherwise it must be left out or set to
  the default, 10.
- **Out of scope.** A request whose purpose the system cannot serve (weather, image analysis, downloads,
  producing scene ids without tools) is declined as a whole, without searching first.
- **Empty results.** When a search finds nothing, the system says so. It does not widen the dates or the cloud
  limit unless the user asks.
- **Follow-ups.** A follow-up keeps everything the user does not change, and may reuse the box from the earlier
  turn instead of resolving the place again. Switching to radar drops the cloud limit.

## Task format

```json
{
  "id": "dev-s2-milan-march",
  "split": "dev",
  "category": "single_search",
  "turns": [
    {
      "user": "Show me Sentinel-2 scenes of Milan from March 2025 with little cloud.",
      "expect": {
        "route": "delegate",
        "required_tools": ["resolve_place", "search_scenes"],
        "forbidden_tools": [],
        "search": {
          "product": "sentinel-2-l2a",
          "place": "milan",
          "start": "2025-03-01",
          "end": "2025-03-31",
          "cloud": [0, 20]
        }
      }
    }
  ],
  "notes": "Live check 2026-10-30: the expected search returns 4 scenes."
}
```

- **`route`** is how the turn was handled, observed from the system's behaviour rather than from what it says:
  - `delegate` when the orchestrator called the retrieval agent;
  - `decline` when it refused without delegating;
  - `clarify` when it asked a question without delegating.

  When several routes are correct, `route` lists them. For example, an unknown place may be checked with the
  retrieval agent first (`delegate`) and then clarified.
- **`required_tools`** must be called in this order, with other calls allowed in between. **`forbidden_tools`**
  must not be called. Tool calls are recorded at the MCP server, not reported by the agents.
- **`search`** is the expected `search_scenes` call. `place` is a gazetteer key. `cloud` is the accepted range,
  or `null` when the argument must be absent. `limit` is present only when the question states one.
- **`outcomes`**, when present, lists the acceptable final statuses:

  | Status | Meaning |
  |---|---|
  | `ok` | Scenes were found and reported |
  | `empty` | The search succeeded and found nothing, and the answer says so |
  | `error` | A tool or catalogue error was reported to the user |
  | `declined` | The system will not do it as asked; it may offer an alternative |
  | `clarified` | The system asked for missing information |

  Without `outcomes`, a search turn must end `ok` when the scored search found scenes and `empty` when it
  found none.

`agentic_earth_observation.evals.golden` validates every task:
- the category and split are known;
- every gazetteer key exists;
- the expected search passes the same validators as a real tool call;
- the dates are no later than the reference date;
- the ids are unique.

## Scoring

Per turn, in `agentic_earth_observation.evals.scoring`:

1. **Route:** one of the accepted routes.
2. **Tools:** `required_tools` appear in order, and no `forbidden_tools` are called.
3. **Search arguments:** the last successful `search_scenes` call of the turn is compared with `search`.
   - `product`: exact.
   - `bbox`: overlap of at least **0.98** (intersection over union) with the gazetteer box. A box copied from
     `resolve_place`, even rounded to 3 decimals, passes; a box from memory does not.
   - `start_date` and `end_date`: exact.
   - `max_cloud_cover`: a number inside the accepted range, or absent when `cloud` is `null`.
   - `limit`: as stated in the labelling rules.
   - Argument names the tool does not define are refused by the server and count as errors.
   - The first `search_scenes` call is scored too and reported separately, to show self-correction.
4. **Grounding:** every scene id in the answer was returned by a tool in the same conversation. A fabricated id
   fails the turn.
5. **Outcome:** the final status is acceptable. When the correct answer is a refusal, an error or "none found",
   the answer must cite no scene ids.

A turn passes when all five pass, and **a task passes when all its turns pass**.

Reported alongside:
- **Scene precision:** the share of cited scene ids that a direct catalogue query with the expected arguments
  also returns. That query uses the top of the accepted cloud range and the expected limit, or 50 when none is
  asked for. It shows how often the cited scenes are correct, not only real.
- **Recall:** for tasks that ask for a number of scenes, the share of the correct scenes that were returned.
- **Over-clarification and over-refusal:** clarify and decline rates on tasks where a search was expected.
- **Failure counts:** malformed tool calls, running out of steps and unparseable model output count as **model
  failures** and fail the task. Only failures outside the model's control are infrastructure errors: a server
  not answering, or a timeout of the catalogue. Those are retried, excluded from scores and counted in the
  report.

## Runs and statistics

- Every model runs every task 3 times, in a task order shuffled with a fixed seed. At temperature 0 the runs
  are nearly identical, so they show stability (the share of tasks with the same result in all runs), not
  three times the evidence.
- Results are reported as counts (for example 14/18 tasks), with 95% bootstrap confidence intervals that
  resample whole tasks.
- With 18 test tasks, one task is about 5.6 percentage points, and a single model's interval is roughly ±20
  points.
  - Differences between two models of less than about 25 to 35 points (5 or 6 tasks) can be noise.
  - Pairwise claims use an exact sign test (McNemar) on the tasks where the two models differ, with 6
    comparisons among 4 models.
  - Per-category numbers are descriptive only.
- Every run record stores:
  - the git commit and whether the working tree had changes;
  - the model id;
  - the hashes of the prompts, the tool descriptions, the golden set and the evaluation code;
  - the seed.

## What is frozen

At the git tag `golden-test-v1`:
- the test split;
- this document's labelling rules and scoring;
- `data/gazetteer.json`;
- the evaluation code (`evals/golden.py`, `evals/scoring.py`, including the reference date and the
  thresholds).

A fix after that needs a dated entry in `docs/decisions.md` and a new tag. `evaluation_code_hash()` in
`evals/golden.py` identifies the evaluation code in every run record.

## No leakage

Prompts, tool descriptions and examples never use the test split's places, place names in any language, or
questions. `tests/test_no_leakage.py` checks all system code under `src/` for them on every run.
