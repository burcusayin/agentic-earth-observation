"""Retrieval agent (planned, Milestone 1). Not implemented yet.

A LangGraph agent that answers scene-search requests using only the catalogue MCP tools: a model node and a
tool node with an explicit step limit. Tool errors are passed back to the model. The model's own failures
(running out of steps, malformed tool calls, unparseable output) fail the task; failures outside its control
(the catalogue, the network, a server) are infrastructure errors, excluded from scores. a2a_server.py
publishes it over A2A.
"""

from __future__ import annotations
