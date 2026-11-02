"""Shared pytest configuration."""

from __future__ import annotations

import os
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _no_eo_variables_from_the_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts without the developer's EO_* variables; a test that needs one sets it."""
    for name in list(os.environ):
        if name.upper().startswith("EO_"):
            monkeypatch.delenv(name)


def _keep_only_content_type(response: dict[str, Any]) -> dict[str, Any]:
    """Drop every response header except Content-Type before a response is recorded.

    CDN and server headers (edge location, trace ids, dates) say nothing the tests need, and the edge location
    reveals roughly where the recording was made.
    """
    headers = response.get("headers", {})
    response["headers"] = {name: value for name, value in headers.items() if name.lower() == "content-type"}
    return response


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, Any]:
    """Settings for recorded HTTP responses (pytest-recording / vcrpy).

    Requests are matched on the body too: every STAC search is a POST to the same /search URL, and vcrpy
    ignores the body by default, so it would replay the wrong answer. Responses are stored decompressed so the
    cassettes are readable and reviewable.
    """
    return {
        "match_on": ["method", "scheme", "host", "port", "path", "query", "body"],
        "decode_compressed_response": True,
        "filter_headers": ["authorization", "cookie", "set-cookie"],
        "before_record_response": _keep_only_content_type,
    }
