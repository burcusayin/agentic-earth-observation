# Evaluation design

How the agents are evaluated: the golden set, the labelling rules, the scoring, and what is frozen. This document is written before any agent prompt exists, so the test cannot be shaped around the system.

## What is measured

Given a question in plain language, does the system call the right tools with the right arguments, decline what it cannot do, ask when a place is unknown, and cite only scenes that a tool returned?

## Golden set

About 20 tasks in `data/golden/dev.jsonl` and `data/golden/test.jsonl`, one JSON object per line.

| Category | Count | Purpose |
|---|---|---|
| `single_search` | 6 | One question, one search: product, place, dates, cloud limit |
| `follow_up` | 3 | Two turns; the second depends on the first ("same area, but November 2024") |
| `robustness` | 4 | Typos, other languages, relative dates |
| `out_of_scope` | 3 | Requests the system must decline without searching, including a prompt injection |
| `clarify` | 2 | Unknown places: the system must ask instead of guessing coordinates |
| `edge` | 2 | Requests that cannot be served as asked (radar with a cloud limit, dates before launch) |

- **Split:** 4 tasks for development (`dev`), used to tune prompts and tool descriptions. The remaining tasks are the `test` split, which is only ever measured.
- **Freeze:** the test split, the scoring tolerances and `data/gazetteer.json` are frozen at the git tag `golden-test-v1`. A fix after that needs a dated entry in `docs/decisions.md` and a new tag.
- **No leakage:** prompts, tool descriptions and examples never use the test split's places, dates or wording.

## Labelling rules

These rules decide the expected answer for every task. The agents' prompts state the same conventions, so the system is scored on following stated rules, not on guessing ours.

- **Reference date.** "Today" is **2025-12-15**. Relative dates are resolved against it: "last summer" is 2025-06-01 to 2025-08-31, "last month" is 2025-11-01 to 2025-11-30.
- **Months and weeks.** "July 2025" is the whole calendar month. "The first week of February 2025" is 1 to 7 February.
- **Sensor.** "Radar", "SAR" or "through clouds" means `sentinel-1-grd`. "Sentinel-2", "optical", "images", "photos", or no sensor named means `sentinel-2-l2a`.
- **Clouds** (Sentinel-2 only).

  | Wording | Accepted `max_cloud_cover` |
  |---|---|
  | "cloud-free", "no clouds", "clear" | any value from 0 to 10 |
  | "mostly clear", "little cloud", "low cloud" | any value from 0 to 20 |
  | no cloud wording | no limit: the argument must be absent |

  A cloud limit on a Sentinel-1 search is always wrong: radar sees through clouds.
- **Places.** Bounding boxes come from `resolve_place`, never from the model's memory. A place missing from the gazetteer is a `clarify` task.
- **Limit.** Scored only when the question asks for a number of scenes.

## Task format

```json
{
  "id": "s2-florence-july",
  "split": "test",
  "category": "single_search",
  "turns": [
    {
      "user": "Find cloud-free Sentinel-2 images of Florence from July 2025.",
      "expect": {
        "route": "delegate",
        "required_tools": ["resolve_place", "search_scenes"],
        "forbidden_tools": [],
        "search": {
          "product": "sentinel-2-l2a",
          "place": "florence",
          "start": "2025-07-01",
          "end": "2025-07-31",
          "cloud": [0, 10]
        }
      }
    }
  ],
  "notes": "Optional: why the expected answer is what it is."
}
```

- `route` is the system's decision for the turn: `delegate`, `decline` or `clarify`. When more than one is
  correct (for example, dates before a satellite launched may be declined at once, or searched and the error
  explained), it is a list of the acceptable routes.
- `required_tools` must be called in this order (other calls may come in between). `forbidden_tools` must not be called.
- `search` is the expected `search_scenes` call. `place` is a gazetteer key. `cloud` is the accepted range, or `null` when the argument must be absent. `limit` is present only when the question states one.
- `outcomes`, when present, lists the acceptable final statuses (`ok`, `empty`, `error`, `declined`, `clarified`), for tasks where the right behaviour is to report a problem.

`eo_agent_demo.evals.golden` validates every task: known category and split, gazetteer keys that exist, and an expected search that passes the same validators as a real tool call.

## Scoring

Per turn:

1. **Route:** matches `route`.
2. **Tools:** `required_tools` appear in order; no `forbidden_tools`.
3. **Search arguments:** the last successful `search_scenes` call of the turn is compared with `search`.
   - `product`: exact.
   - `bbox`: overlap of at least 0.95 (intersection over union) with the gazetteer box.
   - `start` and `end`: exact.
   - `max_cloud_cover`: inside the accepted range, or absent when `cloud` is `null`.
   - `limit`: exact when expected.
   - Any argument the user did not ask for counts as an error.
   - The first `search_scenes` call is also scored and reported separately, to show self-correction.
4. **Grounding:** every scene id in the answer was returned by a tool in the same conversation. A fabricated id fails the task, whatever else is right.
5. **Task success:** the scene ids returned are compared with a direct catalogue query made with the expected arguments at the same time. Precision on the returned ids, and recall up to the requested limit.

Infrastructure failures (a server not answering, a timeout outside the model's control) are retried, excluded from scores and counted in the report.

## Runs and statistics

- Every model runs every task 3 times, task order shuffled with a fixed seed.
- Results are reported with 95% bootstrap confidence intervals that resample whole tasks.
- With about 16 test tasks, differences of less than about 15 percentage points are within noise; per-category numbers are descriptive only.
- Every run record stores the git commit, the model id, the prompt and golden-set hashes, and the seed.
