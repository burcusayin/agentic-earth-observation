"""Retrieval agent (planned, Milestone 1). Not implemented yet.

A LangGraph agent that answers scene-search requests using only the catalogue MCP tools: a model node and a
tool node with an explicit step limit. Tool errors are passed back to the model; any other failure ends the
run as an infrastructure error. a2a_server.py publishes it over A2A.
"""

from __future__ import annotations
