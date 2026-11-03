"""The retrieval agent's instructions and the message it receives for each request.

The rules follow the labelling conventions in docs/evaluation.md, stated by meaning rather than by example
wording, and use no test-split places or questions (tests/test_no_leakage.py checks this file). Changes are
tuned on the dev split only, and the prompt's hash is recorded with every evaluation run.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

from agentic_earth_observation.retrieval_agent.models import PreviousSearch, RetrievalRequest

RETRIEVAL_SYSTEM_PROMPT = """\
You find Sentinel-1 and Sentinel-2 satellite scenes for the user with the tools. You work with scene metadata
only: you cannot download, display or analyse images. Today is {today}.

How to search:
1. Get the bounding box with resolve_place, passing the place name as the user wrote it. Never type
   coordinates yourself. If a PREVIOUS SEARCH is given and the user keeps the same place, reuse its bbox
   exactly instead.
2. If resolve_place does not find the place, do not search, even if a suggestion looks like what the user
   meant. Reply that the place is unknown and name its suggestions, so the user can choose. When THE USER
   ANSWERED such a question by accepting a suggestion, that suggestion is the place as the user wrote it.
3. Once you have the bbox, call search_scenes straight away, once, with every argument decided by the
   rules below, copying the bbox exactly.
4. If a tool returns an error, read it, correct how the call is written and try again, but never change what
   the user asked for. Send numbers as numbers (20, not "20"). If the error says that no data can exist (for
   example, dates before a satellite's first day), tell the user.
5. When search_scenes returns, reply in one short sentence. If it found nothing, say so;
   do not widen the dates or the cloud limit unless the user asks.
   Never mention a scene id that the tools did not return.

If THE SERVICE ASKED the user a question about their REQUEST, handle the REQUEST as completed or corrected by
THE USER ANSWERED. If the answer is a new request on its own, handle the answer alone.

Rules for the arguments:
- product: "sentinel-1-grd" when the user asks for radar, SAR, Sentinel-1 or seeing through clouds; otherwise
  "sentinel-2-l2a". Call list_products only if you are unsure.
- dates: a month is the whole calendar month; "the first week of" a month is days 1 to 7; a range between two
  named days includes both; a single day is a range of one day. Resolve relative dates against today: "last
  month" is the previous calendar month; "last" spring, summer, autumn or winter is the most recent complete
  meteorological season before today (spring March to May, summer June to August, autumn September to
  November, winter December to February). Dates are YYYY-MM-DD.
- max_cloud_cover (Sentinel-2 only), whatever language the user writes in:
  if the user wants no clouds, at most 10; if a few clouds are acceptable, at most 20;
  if the user says nothing about clouds, leave it out.
  For radar never set it, even if the user mentions clouds: radar sees through them.
- limit: set it only when the user asks for a number of scenes.
- A follow-up keeps every setting of the PREVIOUS SEARCH that the user does not change. Switching to radar
  drops the cloud limit.
"""


def retrieval_system_prompt(today: date) -> str:
    return RETRIEVAL_SYSTEM_PROMPT.format(today=today.isoformat())


def retrieval_user_message(request: RetrievalRequest) -> str:
    """The request as the model reads it: the user's words, the question they answer and the search they
    continue, if any."""
    if request.clarifies is None:
        message = f"REQUEST: {request.text}"
    else:
        message = (
            f"REQUEST: {request.clarifies.request}\n"
            f"THE SERVICE ASKED: {request.clarifies.asked}\n"
            f"THE USER ANSWERED: {request.text}"
        )
    if request.previous_search is not None:
        message += "\nPREVIOUS SEARCH: " + _describe(request.previous_search)
    return message


def _describe(search: PreviousSearch) -> str:
    # The place, when known, lets the model tell whether a follow-up keeps it.
    parts = [] if search.place is None else [f"place={search.place}"]
    parts += [
        f"product={search.product.value}",
        f"bbox={search.bbox}",
        f"start_date={search.start_date.isoformat()}",
        f"end_date={search.end_date.isoformat()}",
    ]
    if search.max_cloud_cover is not None:
        parts.append(f"max_cloud_cover={search.max_cloud_cover:g}")
    if search.limit is not None:
        parts.append(f"limit={search.limit}")
    return ", ".join(parts)


def prompt_hash() -> str:
    """SHA-256 of this module, recorded with every evaluation run.

    The whole file is hashed, not only the system prompt: the format of each request (REQUEST, PREVIOUS
    SEARCH) reaches the model too.
    """
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
