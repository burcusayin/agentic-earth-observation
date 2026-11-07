"""Validated types shared by the MCP server, the agents and the evaluation.

Every value that crosses a boundary (tool arguments, configuration, golden tasks) is turned into one of these
types first, so invalid input is rejected in one place, with a message that says how to fix it.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, time
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Validated types reject unknown fields (a misspelt argument must not be dropped silently)
# and are strict about numbers (True is not 1, "10" is not 10).
_STRICT = ConfigDict(frozen=True, extra="forbid", strict=True)


class Product(StrEnum):
    """Products offered by the catalogue tools. The values are the names a model sees; backends map them."""

    SENTINEL_2_L2A = "sentinel-2-l2a"  # optical imagery: cloud cover applies
    SENTINEL_1_GRD = "sentinel-1-grd"  # radar: sees through clouds


MISSION_START: dict[Product, date] = {
    Product.SENTINEL_2_L2A: date(2015, 6, 23),
    Product.SENTINEL_1_GRD: date(2014, 4, 3),
}
MAX_BBOX_AREA_DEG2 = 25.0  # about 5 x 5 degrees: a region, not a continent
MAX_RANGE_DAYS = 366
_AREA_TOLERANCE = 1e-9  # so that a box exactly at the limit is not rejected by floating-point rounding


class BBox(BaseModel):
    """A longitude/latitude bounding box in degrees, in STAC order: west, south, east, north."""

    model_config = _STRICT

    west: float
    south: float
    east: float
    north: float

    @model_validator(mode="after")
    def _check(self) -> Self:
        # Finite values first: comparisons with NaN are always False, so the later checks would pass silently.
        values = self.as_list()
        if not all(math.isfinite(v) for v in values):
            raise ValueError("All bounding box values must be finite numbers (not NaN or infinity).")
        if not (
            -180 <= self.west <= 180
            and -180 <= self.east <= 180
            and -90 <= self.south <= 90
            and -90 <= self.north <= 90
        ):
            raise ValueError(
                "bbox coordinates out of range: "
                "longitude must be in [-180, 180] "
                "and latitude in [-90, 90]; got " + repr(values)
            )
        if self.west >= self.east:
            raise ValueError(
                f"west must be less than east (got west={self.west}, east={self.east}). "
                "If the values are swapped, exchange them; if the area crosses the antimeridian (180°), "
                "split it into two boxes, one on each side."
            )
        if self.south >= self.north:
            raise ValueError(
                f"south must be less than north (got south={self.south}, north={self.north}); "
                "swap them if they are reversed."
            )
        area = (self.east - self.west) * (self.north - self.south)
        if area > MAX_BBOX_AREA_DEG2 + _AREA_TOLERANCE:
            raise ValueError(
                f"Bounding box area {area:.2f} exceeds the maximum allowed area "
                f"of {MAX_BBOX_AREA_DEG2:.2f} square degrees; search a smaller area or split it."
            )
        return self

    def as_list(self) -> list[float]:
        return [self.west, self.south, self.east, self.north]


class DateRange(BaseModel):
    """An inclusive range of calendar days, in UTC."""

    model_config = ConfigDict(frozen=True, extra="forbid")  # not strict: dates arrive as ISO strings

    start: date
    end: date

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.start > self.end:
            raise ValueError(f"Start date {self.start} must not be after end date {self.end}.")
        if (self.end - self.start).days + 1 > MAX_RANGE_DAYS:
            raise ValueError(f"Date range exceeds the maximum allowed range of {MAX_RANGE_DAYS} days.")
        return self

    def to_stac_interval(self) -> str:
        """The STAC `datetime` parameter: from the start of `start` to the very end of `end`, in UTC."""
        first = datetime.combine(self.start, time.min, tzinfo=UTC)
        last = datetime.combine(self.end, time.max, tzinfo=UTC)  # 23:59:59.999999, so no late scene is missed
        return f"{first:%Y-%m-%dT%H:%M:%SZ}/{last:%Y-%m-%dT%H:%M:%S.%fZ}"


class SearchRequest(BaseModel):
    """One scene search, as the catalogue receives it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    product: Product
    bbox: BBox
    dates: DateRange
    max_cloud_cover: float | None = Field(default=None, ge=0, le=100, strict=True)
    limit: int = Field(default=10, ge=1, le=50, strict=True)

    @model_validator(mode="after")
    def _check(self) -> Self:
        # Radar sees through clouds, so a cloud filter on Sentinel-1 is meaningless rather than harmless.
        if self.product == Product.SENTINEL_1_GRD and self.max_cloud_cover is not None:
            raise ValueError(
                "max_cloud_cover filter is not applicable "
                "for Sentinel-1 (radar) products. Please remove this filter."
            )
        if self.dates.start < MISSION_START[self.product]:
            raise ValueError(
                f"Start date {self.dates.start} is before the mission start date "
                f"{MISSION_START[self.product]} for product {self.product}: "
                f"start on {MISSION_START[self.product]} or later."
            )
        today = datetime.now(UTC).date()
        if self.dates.start > today:
            raise ValueError(
                f"Start date {self.dates.start} is in the future (today is {today} in UTC); "
                "no scenes exist yet."
            )
        return self
