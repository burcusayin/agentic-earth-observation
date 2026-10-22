"""Data-retrieval sub-agent: a LangGraph agent that answers scene-search requests
using only the catalogue MCP tools.

Milestone 2: build the graph (model node + tool node, with a step limit) and load
the tools from the MCP server.
Milestone 3: expose it as an A2A server (agent card, one skill: "find satellite
scenes for a place and period") in a2a_server.py.
"""

from __future__ import annotations

# TODO(M2): build_retrieval_agent(model: str) -> compiled graph
