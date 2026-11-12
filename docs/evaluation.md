# Evaluation design

How the agents are evaluated: the golden set, the labelling rules, the scoring, how runs are compared, and what
is frozen. The golden test split and this document have been frozen twice:

- **`golden-test-v1`** (2026-10-31): 18 test tasks, frozen before any agent prompt was written. The MCP tool
  descriptions already existed; they are checked automatically for test-split content (see "No leakage"). The
  results of v1.0 are on these tasks.
- **`golden-test-v2`**: the same 18 tasks, unchanged, followed by 32 new ones, and the rules for comparing two
  setups. The new tasks were added on 2026-11-12, after the results of v1.0 had been seen and before any run
  on them. The labelling rules and the scoring are word for word those of v1.

## What is measured

Given a question in plain language, does the system call the right tools with the right arguments, report what
it found, decline what it cannot do, ask when a place is unknown, and cite only scenes that a tool returned?

## Golden set

54 tasks in `data/golden/dev.jsonl` (4) and `data/golden/test.jsonl` (50), one JSON object per line.

| Category | Dev | Test, v1 | Test, added in v2 | Test | Purpose |
|---|---|---|---|---|---|
| `single_search` | 1 | 5 | 4 | 9 | One question, one search: product, place, dates, cloud limit, limit |
| `follow_up` | 1 | 2 | 5 | 7 | Two turns; the second changes one thing and keeps the rest |
| `robustness` | 1 | 3 | 13 | 16 | Typos in the wording, other languages, relative dates |
| `out_of_scope` | 1 | 2 | 3 | 5 | Requests the system must decline without searching, including a prompt injection |
| `clarify` | 0 | 3 | 4 | 7 | Places the gazetteer does not know, including a misspelt one: the system must ask |
| `edge` | 0 | 3 | 3 | 6 | Radar with a cloud limit, dates before a satellite launched, a search with no results |
| Total | 4 | 18 | 32 | 50 | |

- **Split.** Prompts and tool descriptions are tuned on the dev split only. The test split is only measured.
  Dev and test use different places (checked by a test).
- **Live check.** Every expected search was run against Earth Search, those of v1 on 2026-10-30 and those added
  in v2 on 2026-11-12; each task's notes record how many scenes it returned. Only the "no results" task of v1
  returns none, by design.
- **The tasks added in v2** (decision of 2026-11-12 in `docs/decisions.md`):
  - Their ids start with `v2-`. Of the 13 robustness tasks, 8 ask for a relative date, 3 are in other languages
    and 2 have typos in the wording. A relative-date task's id is `v2-relative-<period>-…`, the period one of
    `last-month` (2 tasks), `last-spring` (1), `last-summer` (1), `last-autumn` (2) and `last-winter` (2).
  - They need no labelling rule that v1 did not have, and every expected search is over one of the eight places
    of the v1 test split.
  - The mix leans on the kinds of task that v1.0 failed. A pass rate on the 50 tasks is therefore not comparable
    with v1.0's on 18, and says nothing about typical requests.
  - They were written, labelled a second time and reviewed by separate sessions of an AI coding assistant, none
    of which wrote prompts for the system. No person has read them. The checks they went through, and what
    those found, are in the decision.

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
- With 50 test tasks, one task is 2 percentage points, and a single model's interval is roughly ±11 to ±14
  points.
  - Differences between two models of less than about 16 to 20 points (8 to 10 tasks) can be noise.
  - With the 18 tasks of v1, one task is about 5.6 points, a single model's interval is roughly ±20 points, and
    differences between two models of less than about 25 to 35 points (5 or 6 tasks) can be noise.
  - These figures take the tasks to be independent. The relative-date tasks that share a period are not, so the
    real intervals are somewhat wider.
  - Pairwise claims use an exact sign test (McNemar) on the tasks where the two models differ, with 6
    comparisons among 4 models.
  - Per-category numbers are descriptive only.
- Every run record stores:
  - the git commit and whether the working tree had changes;
  - the model id;
  - the hashes of the prompts, the tool descriptions, the golden set and the evaluation code;
  - the seed.

## Comparing two setups

A setup is a set of runs with the same code, prompts, tool descriptions and model options. When something is
changed and measured again, the two setups of one model are compared as below. At this tag `eo-agent report`
makes none of these comparisons yet, and counts only runs on the current golden set: the rules are fixed here
first, before that code and before any run on the 50 tasks.

- **A task's result** in a setup is the result that most of the model's runs in that setup had (3 runs, or more
  when an evaluation was started again), among the runs with a result: a run lost to the infrastructure has
  none, and an errored run counts as failed. A task without a majority is left out, and counted as left out.
  This is the rule already used between two models.
- **Before and after:** the tasks passed in both setups, failed in both, gained, lost and left out. Every gained
  or lost task is listed by id, with an exact two-sided sign test on gained against lost.
- **Seen and new:** the same on all 50 tasks, on the 18 of v1, whose results had been seen when the change was
  chosen, and on the 32 added in v2, which had not been run when the first change after this tag was chosen.
  For a later change all 50 have been seen, and the split into seen and new says nothing.
- **Relative dates:** for the relative-date tasks added in v2, the tasks passed per period, before and after. A
  period passes when all its tasks pass; a period with a task left out is left out, and counted. The sign test
  is over the five periods. It cannot go below p = 0.0625, so nothing is called significant on its strength.
- **Session noise:** the 18 tasks of v1 in the runs of v1.0 against the baseline runs on the 50 tasks, which use
  the same prompts, compared as under "Before and after". A difference there is not the effect of a change to
  the prompts, the tool descriptions or the agents, which are the same on both sides; it shows what the session,
  the task order among 50 tasks and the live catalogue can move.
- **Determinism:** over the turns that every run of a model reached, the share where all runs made the same tool
  calls (names, arguments, order), and the share where they also cited the same scenes. Within a setup, and
  between the two setups of the session-noise comparison.

The noise figure for two different models (see "Runs and statistics") does not apply to one model before and
after: those results are paired, task by task, and are judged by the tasks that changed.

## What is frozen

At the git tag `golden-test-v2`:
- the test split: the 18 tasks of `golden-test-v1`, byte for byte, followed by the 32 added;
- this document's labelling rules and scoring, unchanged since `golden-test-v1`, and its rules for comparing
  two setups;
- `data/gazetteer.json`, unchanged since `golden-test-v1`;
- the evaluation code (`evals/golden.py`, `evals/scoring.py`, including the reference date and the
  thresholds), unchanged since `golden-test-v1`.

A fix after that needs a dated entry in `docs/decisions.md` and a new tag. `evaluation_code_hash()` in
`evals/golden.py` identifies the evaluation code in every run record.

## No leakage

Prompts, tool descriptions and examples never use the test split's places, place names in any language, or
questions. `tests/test_no_leakage.py` checks all system code under `src/` for them on every run.
