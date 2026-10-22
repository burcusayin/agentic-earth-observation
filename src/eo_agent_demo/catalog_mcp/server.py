"""MCP server exposing a STAC catalogue of satellite scenes as tools.

Milestone 1. No LLM is involved here: every tool is a plain, validated function
over the STAC API, so it can be unit-tested and reused by any MCP client.

Tools to implement:
    list_collections() -> list[dict]
    resolve_place(name: str) -> dict            # place name -> bbox, from data/gazetteer.json
    search_scenes(collection, bbox, start_date, end_date,
                  max_cloud_cover=None, limit=10) -> list[dict]
    get_scene(collection: str, item_id: str) -> dict

Validation at the boundary (reject, do not guess):
    bbox: four floats, west < east, south < north, within [-180, 180] / [-90, 90]
    dates: ISO 8601, start <= end
    max_cloud_cover: 0-100
    limit: 1-50
    collection: must be one returned by list_collections()
"""

from __future__ import annotations

# TODO(M1): create the server with the MCP Python SDK, register the four tools,
# and serve over streamable HTTP on the port in CATALOG_MCP_URL.
