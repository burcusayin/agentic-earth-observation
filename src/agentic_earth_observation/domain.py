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


class BBox(BaseModel):
    """A longitude/latitude bounding box in degrees, in STAC order: west, south, east, north."""

    model_config = ConfigDict(frozen=True)

    west: float
    south: float
    east: float
    north: float

    @model_validator(mode="after")
    def _check(self) -> Self:
        #   1. A value that is not finite (NaN or infinity). Check this first: comparisons with NaN are always
        #      False, so the later checks would silently pass. Hint: math.isfinite.
        #   2. A longitude outside [-180, 180] or a latitude outside [-90, 90].
        #      The message must mention "range".
        #   3. west >= east. This also catches boxes crossing the antimeridian (west 179, east -179): mention
        #      "antimeridian" in the message and suggest splitting the box in two.
        #   4. south >= north.
        #   5. An area (width x height, in square degrees) above MAX_BBOX_AREA_DEG2. Mention "area".
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
            raise ValueError("Bounding box south coordinate must be less than north coordinate.")
        area = (self.east - self.west) * (self.north - self.south)
        if area > MAX_BBOX_AREA_DEG2:
            raise ValueError(
                f"Bounding box area {area:.2f} exceeds the maximum allowed area "
                f"of {MAX_BBOX_AREA_DEG2:.2f} square degrees."
            )
        return self

    def as_list(self) -> list[float]:
        return [self.west, self.south, self.east, self.north]


class DateRange(BaseModel):
    """An inclusive range of calendar days, in UTC."""

    model_config = ConfigDict(frozen=True)

    start: date
    end: date

    @model_validator(mode="after")
    def _check(self) -> Self:
        #   1. start after end: the message must mention "start".
        #   2. More than MAX_RANGE_DAYS days, counting both the first and the last day: the message must
        #      mention the limit (366). Hint: (self.end - self.start).days + 1 is the number of days.
        if self.start > self.end:
            raise ValueError("Start date must not be after end date.")
        if (self.end - self.start).days + 1 > MAX_RANGE_DAYS:
            raise ValueError(f"Date range exceeds the maximum allowed range of {MAX_RANGE_DAYS} days.")
        return self

    def to_stac_interval(self) -> str:
        """The STAC `datetime` parameter: first second of `start` to last second of `end`, in UTC."""
        first = datetime.combine(self.start, time.min, tzinfo=UTC)
        last = datetime.combine(self.end, time(23, 59, 59), tzinfo=UTC)
        return f"{first:%Y-%m-%dT%H:%M:%SZ}/{last:%Y-%m-%dT%H:%M:%SZ}"


class SearchRequest(BaseModel):
    """One scene search, as the catalogue receives it."""

    model_config = ConfigDict(frozen=True)

    product: Product
    bbox: BBox
    dates: DateRange
    max_cloud_cover: float | None = Field(default=None, ge=0, le=100)
    limit: int = Field(default=10, ge=1, le=50)

    @model_validator(mode="after")
    def _check(self) -> Self:
        #   1. max_cloud_cover is set for Sentinel-1: radar is not affected by clouds, so the filter is
        #      meaningless.
        #      The message must mention "Sentinel-1" and say to drop the filter.
        #   2. dates.start before the product's MISSION_START: the message must name the first available date
        #      (e.g. 2015-06-23).
        #   3. dates.start after today (UTC): no scenes exist yet. The message must mention "future".
        #      Hint: datetime.now(UTC).date() is today in UTC.
        if self.product == Product.SENTINEL_1_GRD and self.max_cloud_cover is not None:
            raise ValueError(
                "max_cloud_cover filter is not applicable "
                "for Sentinel-1 (radar) products. Please remove this filter."
            )
        if self.dates.start < MISSION_START[self.product]:
            raise ValueError(
                f"Start date {self.dates.start} is before the mission start date "
                f"{MISSION_START[self.product]} for product {self.product}."
            )
        if self.dates.start > datetime.now(UTC).date():
            raise ValueError(f"Start date {self.dates.start} is in the future. No scenes exist yet.")
        return self
