# Contributing

Bug reports and small fixes are welcome. For anything larger, please open an issue first: the scope of the first
milestone is fixed (see the roadmap in the [README](README.md)).

## Before a pull request

```bash
make setup
make check   # lint, format check, mypy --strict, tests with coverage; no network, no models
make hooks   # every commit check
make secrets # the secret scan of the whole history, as CI runs it
```

Tests make no network or model calls by default. Tests marked `integration` need local servers and Ollama;
`live` tests call a public service or a paid model, and do not run in CI.

## What must not change

The evaluation's numbers mean something only if these stay fixed:

- **Frozen at tag `golden-test-v1`:** `data/golden/test.jsonl`, `data/gazetteer.json`, the labelling rules and
  scoring in `docs/evaluation.md`, `evals/golden.py` and `evals/scoring.py`. A test compares them with their hashes.
  A real fix needs a dated entry in `docs/decisions.md` and a new tag.
- **No tuning on the test split.** Prompts and tool descriptions are tuned on the dev split only, and must not use
  the test split's places, dates or wording. `tests/test_no_leakage.py` checks part of this.
- **The models under evaluation are fixed.** Adding one starts with a dated decision.

Design decisions and their reasons are in [`docs/decisions.md`](docs/decisions.md): please read the relevant ones
before proposing to change something, and add a dated entry when a decision changes.

## Licence

By contributing you agree that your contribution is licensed under the [Apache License 2.0](LICENSE).
