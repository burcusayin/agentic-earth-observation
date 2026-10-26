"""Shared pytest configuration."""

from __future__ import annotations

from typing import Any

import pytest


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
    }
