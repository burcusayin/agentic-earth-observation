# Evaluation report: test split

Made from 6 runs that count, of 6 listed, on commit e5961321fd75.
Reference date 2025-12-15. Frozen files: golden set 743d5e2abb5d, evaluation code 72d140b004d9, gazetteer af181e26cb48, labelling rules 9f3ae9ba0643.

## Every run

| Run | State | Why it does not count | Model | Passed | Failed | Errored | Lost | Unattributed calls |
|---|---|---|---|---|---|---|---|---|
| 20261108T112631Z-qwen3.5-9b-r1 | counted |  | qwen3.5:9b | 17 | 1 | 0 | 0 | 0 |
| 20261108T113118Z-qwen3.5-9b-r2 | counted |  | qwen3.5:9b | 17 | 1 | 0 | 0 | 0 |
| 20261108T113633Z-qwen3.5-9b-r3 | counted |  | qwen3.5:9b | 17 | 1 | 0 | 0 | 0 |
| 20261108T114204Z-ministral-3-8b-r1 | counted |  | ministral-3:8b | 15 | 3 | 0 | 0 | 0 |
| 20261108T114513Z-ministral-3-8b-r2 | counted |  | ministral-3:8b | 15 | 3 | 0 | 0 | 0 |
| 20261108T114826Z-ministral-3-8b-r3 | counted |  | ministral-3:8b | 15 | 3 | 0 | 0 | 0 |

## Passed tasks

For each task, the share of its runs that passed, summed over the tasks: with lost runs left out (out of the tasks with a result) and counted as failed (out of all tasks). A run is lost when every attempt fell to the infrastructure. The 95% interval resamples whole tasks.

| Model | Lost tasks left out | Lost tasks counted as failed | Same result in every run | Runs |
|---|---|---|---|---|
| qwen3.5:9b | 17.0 of 18 (95% interval 15.0 to 18.0) | 17.0 of 18 (95% interval 15.0 to 18.0) | 100% | 3 |
| ministral-3:8b | 15.0 of 18 (95% interval 12.0 to 18.0) | 15.0 of 18 (95% interval 12.0 to 18.0) | 100% | 3 |

## Between models

A task's result for a model is the majority of its runs; a task without one is left out. The exact sign test uses the tasks only one of the two passed. 1 comparison; differences of less than about 25 to 35 points (5 or 6 tasks) can be noise (docs/evaluation.md).

| First | Second | Only the first passed | Only the second passed | Left out (no majority) | p | Both passed | Both failed |
|---|---|---|---|---|---|---|---|
| qwen3.5:9b | ministral-3:8b | 3 | 1 | 0 | 0.63 | 14 | 0 |

## qwen3.5:9b

Build d63f830f28d515fefea546f213fca7a75e9dd5176ef51a6deabc952d1c99a911, Ollama 0.40.1, temperature 0; 3 runs: 20261108T112631Z-qwen3.5-9b-r1, 20261108T113118Z-qwen3.5-9b-r2, 20261108T113633Z-qwen3.5-9b-r3.

| Measure | Value |
|---|---|
| Last search call with every argument right | 39 of 42 (93%) |
| First search call with every argument right | 39 of 42 (93%) |
| Asked instead of searching | 0 of 42 (0%) |
| Declined instead of searching | 0 of 42 (0%) |
| Scene precision, the mean over turns | 0.92 over 39 turns |
| Recall, the mean over turns | 1.00 over 3 turns |
| Catalogue queries for them that failed (turns left out) | 0 |
| Answers citing a scene no tool returned | 0 |
| Model failures | none |
| Errored tasks (a bug of ours, counted as failed) | none |
| Attempts lost to the infrastructure (retried) | 0 |
| Tasks lost in every attempt | 0 |
| Tool calls under no trace id of the run | 0 |
| Warm-up after each run's fresh load of the model, seconds | 3.7, 3.0, 2.6 |
| Seconds in turns per run, over its scored tasks | 261.3 (18 tasks), 290.5 (18 tasks), 309.2 (18 tasks) |
| Model seconds per run, over its scored tasks | 244.9, 273.7, 292.1 |
| Seconds per task, median and 90th percentile | 16.3 and 28.6 |
| Model seconds and load seconds per task, median | 15.5 and 0.05 |
| Input and output tokens per task, median | 5830 and 236 |

| Category | Passed, lost runs left out |
|---|---|
| single_search | 5.00 of 5 (of 5 tasks) |
| follow_up | 2.00 of 2 (of 2 tasks) |
| robustness | 2.00 of 3 (of 3 tasks) |
| out_of_scope | 2.00 of 2 (of 2 tasks) |
| clarify | 3.00 of 3 (of 3 tasks) |
| edge | 3.00 of 3 (of 3 tasks) |

| Task | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| s2-florence-july | passed | passed | passed |
| s1-etna-first-week | passed | passed | passed |
| s2-garda-five-latest | passed | passed | passed |
| s2-paris-spring-mostly-clear | passed | passed | passed |
| s1-venice-january | passed | passed | passed |
| follow-rome-other-month | passed | passed | passed |
| follow-trento-switch-to-radar | passed | passed | passed |
| robust-italian-rome | passed | passed | passed |
| robust-typos-munich | passed | passed | passed |
| robust-relative-garda-summer | failed | failed | failed |
| oos-image-analysis | passed | passed | passed |
| oos-injection-ids-from-memory | passed | passed | passed |
| clarify-springfield | passed | passed | passed |
| clarify-typo-florance | passed | passed | passed |
| clarify-bologna | passed | passed | passed |
| edge-radar-with-cloud-limit | passed | passed | passed |
| edge-no-scenes-paris | passed | passed | passed |
| edge-before-launch | passed | passed | passed |

## ministral-3:8b

Build 87fdbf184e8b785e7baa1f9c8c121fbccf059c0bc64267c29e075784031fc584, Ollama 0.40.1, temperature 0; 3 runs: 20261108T114204Z-ministral-3-8b-r1, 20261108T114513Z-ministral-3-8b-r2, 20261108T114826Z-ministral-3-8b-r3.

| Measure | Value |
|---|---|
| Last search call with every argument right | 33 of 39 (85%) |
| First search call with every argument right | 33 of 39 (85%) |
| Asked instead of searching | 0 of 39 (0%) |
| Declined instead of searching | 6 of 39 (15%) |
| Scene precision, the mean over turns | 1.00 over 30 turns |
| Recall, the mean over turns | 1.00 over 3 turns |
| Catalogue queries for them that failed (turns left out) | 0 |
| Answers citing a scene no tool returned | 0 |
| Model failures | none |
| Errored tasks (a bug of ours, counted as failed) | none |
| Attempts lost to the infrastructure (retried) | 0 |
| Tasks lost in every attempt | 0 |
| Tool calls under no trace id of the run | 0 |
| Warm-up after each run's fresh load of the model, seconds | 3.7, 3.0, 2.7 |
| Seconds in turns per run, over its scored tasks | 163.0 (18 tasks), 168.8 (18 tasks), 169.3 (18 tasks) |
| Model seconds per run, over its scored tasks | 149.0, 153.9, 152.5 |
| Seconds per task, median and 90th percentile | 10.4 and 14.3 |
| Model seconds and load seconds per task, median | 9.2 and 0.01 |
| Input and output tokens per task, median | 4558 and 141 |

| Category | Passed, lost runs left out |
|---|---|
| single_search | 5.00 of 5 (of 5 tasks) |
| follow_up | 1.00 of 2 (of 2 tasks) |
| robustness | 2.00 of 3 (of 3 tasks) |
| out_of_scope | 2.00 of 2 (of 2 tasks) |
| clarify | 2.00 of 3 (of 3 tasks) |
| edge | 3.00 of 3 (of 3 tasks) |

| Task | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| s2-florence-july | passed | passed | passed |
| s1-etna-first-week | passed | passed | passed |
| s2-garda-five-latest | passed | passed | passed |
| s2-paris-spring-mostly-clear | passed | passed | passed |
| s1-venice-january | passed | passed | passed |
| follow-rome-other-month | passed | passed | passed |
| follow-trento-switch-to-radar | failed | failed | failed |
| robust-italian-rome | failed | failed | failed |
| robust-typos-munich | passed | passed | passed |
| robust-relative-garda-summer | passed | passed | passed |
| oos-image-analysis | passed | passed | passed |
| oos-injection-ids-from-memory | passed | passed | passed |
| clarify-springfield | passed | passed | passed |
| clarify-typo-florance | failed | failed | failed |
| clarify-bologna | passed | passed | passed |
| edge-radar-with-cloud-limit | passed | passed | passed |
| edge-no-scenes-paris | passed | passed | passed |
| edge-before-launch | passed | passed | passed |

## Reading these numbers

- With 18 tasks, one task is 5.6 points. Per-category numbers are descriptive only.
- The runs were at temperature 0: at 0 a model's runs are nearly identical, so they show stability, not more evidence.
- Infrastructure failures are retried and excluded; errored tasks are a bug of ours and count as failed.
- An attempt stops at its first failed turn, so the measures per turn and the cost of a task cover the turns that ran: a follow-up failed at its first turn adds one turn and a shorter task.
- Answers are built from templates and cite only scene ids the tools returned, so grounding is a property of the design: no answer cited another.
- Rules and limitations: `docs/evaluation.md` and `docs/decisions.md`.
