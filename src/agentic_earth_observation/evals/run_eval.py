"""Evaluation runner (planned, Milestone 1). Not implemented yet.

For each task in data/golden/{dev,test}.jsonl and each model under test, run the system, record every tool
call at the MCP server, the final status, the cited scene ids, latency and token usage, and score each turn
with evals/scoring.py as described in docs/evaluation.md. One JSON record per (task, model, run).
"""

from __future__ import annotations
