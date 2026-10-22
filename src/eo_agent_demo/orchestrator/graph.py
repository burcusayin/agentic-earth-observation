"""Orchestrator graph (milestone 3).

State: messages, plan, delegated task ids, final answer.
Nodes:
    plan        decide: answer directly, delegate to the retrieval agent, or refuse (out of scope)
    delegate    call the retrieval agent over A2A, attach its result to the state
    synthesise  write the answer, citing scene ids returned by the sub-agent only
Memory: a checkpointer so a follow-up ("same area, but last winter") reuses the earlier bbox.
"""

from __future__ import annotations

# TODO(M3): build_orchestrator(model: str) -> compiled graph
