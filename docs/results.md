# Results: the first evaluation (v1.0)

Two local models, each run three times over the 18 tasks of the frozen test split. Every result here comes from
the run records in [`results/runs/`](../results/runs/), and the report made from them is
[`results/summary.md`](../results/summary.md); the machine's details and the caching figures come from
[`decisions.md`](decisions.md). The rules are in [`evaluation.md`](evaluation.md), frozen at tag
`golden-test-v1` before any prompt was written; the decisions behind the runner and the report are in
[`decisions.md`](decisions.md).

## What was run

| | |
|---|---|
| Models | `qwen3.5:9b` and `ministral-3:8b` through Ollama 0.40.1, temperature 0, seed 42, a 16,384-token context |
| Tasks | the 18 tasks of the test split, in one order shuffled with seed 1, the reference date 2025-12-15 |
| Runs | 3 per model, on 2026-11-08, with `eo-agent eval --split test --for-report` from clean commit `e596132` |
| Each run | its own MCP server and retrieval agent, the model loaded afresh and warmed up, the catalogue queried live |
| Machine | a fanless MacBook Air (Apple M5, 16 GB), on battery, with other applications open (as noted by the maintainer at the start; the run records name only macOS on Apple silicon) |

All six runs completed. No attempt was lost to the infrastructure or needed a retry, no task errored, and the MCP
server logged no call outside the runs' turns. The six runs took 25 minutes in all.

## The results

| | qwen3.5:9b | ministral-3:8b |
|---|---|---|
| **Passed tasks** (95% bootstrap interval over tasks) | **17 of 18** (15 to 18) | **15 of 18** (12 to 18) |
| Same result in all 3 runs | every task | every task |
| Search turns whose last search call had every argument right | 39 of 42 | 33 of 39 |
| Search turns whose first search call did | 39 of 42 | 33 of 39 |
| Search turns that ended asking instead of searching | 0 | 0 |
| Search turns that ended declining instead of searching | 0 | 6 of 39 |
| Scene precision, the mean over turns | 0.92 over 39 turns | 1.00 over 30 turns |
| Answers citing a scene that no tool returned | 0 | 0 |
| Seconds per task, median (90th percentile) | 16.3 (28.6) | 10.4 (14.3) |
| Input and output tokens per task, median | 5,830 and 236 | 4,558 and 141 |

A task passes when every one of its turns passes all five checks of the scoring: the route, the tools, each
search argument, grounding and the final status. The search turns differ in number (42 and 39) because an attempt
stops at its first failed turn: ministral's failed follow-up never reached its second turn.

### The two models compared

On 14 tasks both models passed, on 3 only qwen3.5:9b did, on 1 only ministral-3:8b, and on none did both fail.
The exact sign test on the 4 tasks where they differ gives p = 0.63.

**This is not evidence that one model is better.** With 18 tasks, one task is 5.6 points; the rules, written
before any run, estimate a single model's interval at roughly 20 points either way (here 15 to 18 and 12 to 18
tasks), and say that a difference of fewer than 5 or 6 tasks can be noise. Here the difference is 2.

## What went wrong, task by task

Every failure was the same in all three runs. The ids of the failed tasks name their category; the tasks' text is
in `data/golden/test.jsonl`.

- **`robust-relative-garda-summer` (robustness), failed by qwen3.5:9b.** A relative date. The expected search,
  as the catalogue's answer to it shows, covers June to August 2025: under the labelling rules, the most recent
  complete summer before the reference date of 2025-12-15. The model searched June to August 2024, a year early.
  Every other argument was right. The search ran and found ten scenes, none of them in the period asked for: its
  scene precision on this task is 0, the only turns below 1 for this model. ministral-3:8b searched 2025 and
  passed.
- **`follow-trento-switch-to-radar` (follow-up), failed by ministral-3:8b.** On the first turn it delegated, the
  retrieval agent looked the place up, and then searched nothing. Its result, `not_searched`, reaches the user as
  a decline (the turn's status is `declined`), so the second turn, the one that switches to radar, never ran.
  Answering after `resolve_place` without searching is a pattern seen on the dev split while the prompts were
  tuned (decisions of 2026-11-02 and 2026-11-03).
- **`robust-italian-rome` (robustness), failed by ministral-3:8b.** A request in Italian, which the orchestrator
  declined without delegating it. The same model declines the German question of the dev split, which neither
  the earlier nor the tuned prompt fixed (decision of 2026-11-03).
- **`clarify-typo-florance` (clarify), failed by ministral-3:8b.** A misspelt place. The orchestrator asked
  "which place?" itself, without delegating. The route and the outcome, a question to the user, were accepted,
  but no tool was called, where the task requires one: its tools check failed.

The six declines on search turns of ministral-3:8b are the first two of these three tasks, in each of its runs.

## Other findings

- **Grounding held:** no answer cited a scene that a tool had not returned. Answers are built from templates, so
  this is a property of the design rather than a skill of the models.
- **No self-correction was needed:** on every search turn the first search call was as right as the last, for
  both models.
- **The runs repeated themselves exactly:** at temperature 0, every task had the same result in all three runs of
  each model, so the three runs show stability, not three times the evidence.
- **Prompt caching nearly halved the measured latency of later runs, until each run loaded its model afresh.**
  Ollama 0.40.1 answers a prompt it has seen from a cache while still reporting every prompt token, and at
  temperature 0 a run repeats the prompts of the run before. In a dev-split rehearsal, qwen3.5:9b's second and
  third runs took 35 to 40 model seconds against 75 to 76 for the first (decision of 2026-11-08). An evaluation
  that repeats runs against an Ollama server that keeps its cache between runs is exposed to this.

## Limitations

- **18 tasks, 2 models.** The intervals are wide, and only one comparison could be made, where the rules foresaw
  six among four models: the two Mistral API models moved to a follow-up milestone (decision of 2026-11-05).
- **The timings come from one fanless laptop, on battery, with other applications open.** qwen3.5:9b took 245,
  274 and 292 model seconds over its three runs, 19% more from the first to the third, with the same work in
  each; ministral-3:8b stayed level at 149 to 154. Heat on a fanless laptop is one plausible cause; battery
  power management and the other applications are others, and the records cannot tell them apart. The scores did
  not move. The report shows every run's time, so the drift can be read.
- **One person wrote the golden set and its labelling rules.** They were frozen before any prompt existed, and the
  prompts were tuned on the dev split only, but no second person labelled the tasks. One review of the code read
  the test split, to check a prompt against the labelling rules; its one finding that came only from test
  questions was deliberately not acted on (decision of 2026-11-02).
- **The test split is published with the repository,** so models trained later may have seen these tasks.
- **Known limits of the frozen scoring and of the checks around it,** all recorded with their reasons in
  `decisions.md` (2026-11-07): the scorer reads a box of numbers written as text, so a refused first call could
  still pass first-call accuracy on its box (no search call of these runs sent its box as text); the "start date
  in the future" check uses the machine's clock rather than the agents' reference date (no task is affected); recall
  is taken over the scenes of a turn's last successful search; and two latent defects in the frozen files, which
  no task reaches.

## Checking these numbers

```bash
make report    # remakes results/summary.md from results/runs/
```

Each run's folder holds `run.json` (the commit, the model build, the hashes of the prompts and of the frozen
files, the settings), `attempts.jsonl` (every attempt with its turns, their scores and their tool calls) and
`tool_calls.jsonl` (the MCP server's own log of every call). The report refuses to count a run whose record does
not add up, or whose setup differs from the others.
