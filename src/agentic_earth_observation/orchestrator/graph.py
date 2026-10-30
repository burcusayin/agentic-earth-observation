"""Orchestrator graph (planned, Milestone 1). Not implemented yet.

The orchestrator decides how each user turn is handled and is the A2A client of the retrieval agent:
    decide       delegate to the retrieval agent, decline (out of scope), or clarify (missing information)
    delegate     send the request to the retrieval agent over A2A and keep its structured result
    answer       reply to the user, citing only scene ids that the retrieval agent's tools returned
Memory: a checkpointer, so a follow-up question can reuse the earlier place, box and product.
"""

from __future__ import annotations
