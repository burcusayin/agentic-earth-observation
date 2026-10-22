"""Evaluation runner (milestone 4).

For each task in data/golden.jsonl and each model under test:
    run the orchestrator, record every tool call, the final answer, latency and token usage
    score with evals/scoring.py
Write one JSON line per (task, model, run) to results/raw/, then a summary table with
bootstrap confidence intervals to results/summary.md.
"""

from __future__ import annotations

# TODO(M4)
