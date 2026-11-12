"""Groups of test tasks, read from their ids (decision of 2026-11-12).

The test split of tag `golden-test-v2` is the 18 tasks of v1 followed by new ones. A report has to tell them
apart, and to group the new relative-date tasks by the period they ask for, but a task has no field for
either and `golden.py` is frozen. So the ids of the new tasks carry both:

    v2-...                        a task added in v2
    v2-relative-<period>-...      a new relative-date task; <period> is one of PERIODS
"""

from __future__ import annotations

V2_PREFIX = "v2-"
RELATIVE_PREFIX = "v2-relative-"
# The relative periods that the labelling rules of docs/evaluation.md define, as they appear in an id.
PERIODS = ("last-month", "last-spring", "last-summer", "last-autumn", "last-winter")


def added_in_v2(task_id: str) -> bool:
    """Whether the task was added in v2, and so had not been run when v1.0's results were seen."""
    return task_id.startswith(V2_PREFIX)


def relative_period(task_id: str) -> str | None:
    """The relative period a new relative-date task asks for; None for any other task.

    ValueError for an id that starts as a relative-date task's and names none of the periods.
    """
    if not task_id.startswith(RELATIVE_PREFIX):
        return None
    rest = task_id.removeprefix(RELATIVE_PREFIX)
    for period in PERIODS:
        if rest == period or rest.startswith(f"{period}-"):
            return period
    raise ValueError(
        f"{task_id} starts with {RELATIVE_PREFIX} but names none of the periods {', '.join(PERIODS)}"
    )
