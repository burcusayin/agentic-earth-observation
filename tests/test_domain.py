"""Specification for the validated domain types: what is accepted and what is rejected."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date, datetime, tzinfo

import pytest
from pydantic import ValidationError

from agentic_earth_observation import domain
from agentic_earth_observation.domain import BBox, DateRange, Product, SearchRequest

ROME = {"west": 12.35, "south": 41.78, "east": 12.65, "north": 42.00}
JULY_2025 = {"start": "2025-07-01", "end": "2025-07-31"}


def make_request(**overrides: object) -> SearchRequest:
    """A valid Sentinel-2 search over Rome in July 2025, with any field replaced by `overrides`."""
    fields: dict[str, object] = {"product": "sentinel-2-l2a", "bbox": ROME, "dates": JULY_2025}
    fields.update(overrides)
    return SearchRequest.model_validate(fields)


def assert_rejected(build: Callable[[], object], *words: str) -> str:
    """`build()` must raise a ValidationError whose message contains every word (ignoring case).

    Only the error message is checked, not pydantic's echo of the input, so a word that merely appears in the
    input (such as a field name) does not count. A `{` in the message means an f-string was not formatted.
    """
    with pytest.raises(ValidationError) as excinfo:
        build()
    message = str(excinfo.value.errors()[0]["msg"])
    for word in words:
        assert word.lower() in message.lower(), f"{word!r} is missing from the message: {message!r}"
    assert "{" not in message, f"unformatted placeholder in the message: {message!r}"
    return message


class TestBBox:
    def test_accepts_a_normal_box(self) -> None:
        assert BBox(**ROME).as_list() == [12.35, 41.78, 12.65, 42.00]

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_values_that_are_not_finite(self, bad: float) -> None:
        assert_rejected(lambda: BBox(west=bad, south=41.0, east=12.0, north=42.0), "finite")

    @pytest.mark.parametrize(
        ("west", "south", "east", "north"),
        [(-181, 0, 1, 1), (0, 0, 181, 1), (0, -91, 1, 1), (0, 0, 1, 91)],
    )
    def test_rejects_coordinates_out_of_range(
        self, west: float, south: float, east: float, north: float
    ) -> None:
        assert_rejected(lambda: BBox(west=west, south=south, east=east, north=north), "range")

    def test_rejects_west_east_of_east(self) -> None:
        assert_rejected(lambda: BBox(west=12.65, south=41.78, east=12.35, north=42.0), "west", "east")

    def test_rejects_zero_width(self) -> None:
        assert_rejected(lambda: BBox(west=12.5, south=41.78, east=12.5, north=42.0), "west", "east")

    def test_explains_antimeridian_crossing(self) -> None:
        assert_rejected(lambda: BBox(west=179.0, south=-10.0, east=-179.0, north=10.0), "antimeridian")

    def test_rejects_south_north_of_north(self) -> None:
        assert_rejected(lambda: BBox(west=12.35, south=42.0, east=12.65, north=41.78), "south", "north")

    def test_rejects_a_box_that_is_too_large_and_states_the_limit(self) -> None:
        assert_rejected(lambda: BBox(west=0.0, south=0.0, east=10.0, north=10.0), "area", "25")

    def test_accepts_a_box_exactly_at_the_area_limit(self) -> None:
        assert BBox(west=0.0, south=0.0, east=5.0, north=5.0).north == 5.0


class TestDateRange:
    def test_parses_iso_strings_and_builds_the_stac_interval(self) -> None:
        dates = DateRange.model_validate(JULY_2025)
        assert dates.to_stac_interval() == "2025-07-01T00:00:00Z/2025-07-31T23:59:59.999999Z"

    def test_accepts_a_single_day_including_a_leap_day(self) -> None:
        assert DateRange(start=date(2024, 2, 29), end=date(2024, 2, 29)).start == date(2024, 2, 29)

    def test_rejects_a_date_that_does_not_exist(self) -> None:
        # pydantic does this one for free: 2025 is not a leap year.
        with pytest.raises(ValidationError):
            DateRange.model_validate({"start": "2025-02-29", "end": "2025-03-01"})

    def test_rejects_start_after_end(self) -> None:
        assert_rejected(lambda: DateRange(start=date(2025, 7, 31), end=date(2025, 7, 1)), "start", "end")

    def test_accepts_a_whole_leap_year(self) -> None:
        # 2024-01-01 to 2024-12-31 is 366 days, counting both ends.
        assert DateRange(start=date(2024, 1, 1), end=date(2024, 12, 31)).end == date(2024, 12, 31)

    def test_rejects_more_than_366_days(self) -> None:
        assert_rejected(lambda: DateRange(start=date(2024, 1, 1), end=date(2025, 1, 1)), "366")


class TestSearchRequest:
    def test_accepts_a_typical_request_and_fills_defaults(self) -> None:
        request = make_request()
        assert request.product is Product.SENTINEL_2_L2A
        assert request.limit == 10
        assert request.max_cloud_cover is None

    def test_accepts_a_cloud_filter_for_sentinel_2(self) -> None:
        assert make_request(max_cloud_cover=10).max_cloud_cover == 10

    @pytest.mark.parametrize("cloud", [-1, 101])
    def test_rejects_cloud_cover_out_of_range(self, cloud: float) -> None:
        with pytest.raises(ValidationError):
            make_request(max_cloud_cover=cloud)

    @pytest.mark.parametrize("limit", [0, 51])
    def test_rejects_limit_out_of_range(self, limit: int) -> None:
        with pytest.raises(ValidationError):
            make_request(limit=limit)

    def test_rejects_an_unknown_product(self) -> None:
        with pytest.raises(ValidationError):
            make_request(product="landsat-8")

    def test_rejects_a_cloud_filter_for_sentinel_1(self) -> None:
        assert_rejected(lambda: make_request(product="sentinel-1-grd", max_cloud_cover=10), "Sentinel-1")

    def test_accepts_sentinel_1_without_a_cloud_filter(self) -> None:
        assert make_request(product="sentinel-1-grd").product is Product.SENTINEL_1_GRD

    @pytest.mark.parametrize(
        ("product", "dates", "first_day"),
        [
            ("sentinel-2-l2a", {"start": "2015-01-01", "end": "2015-01-31"}, "2015-06-23"),
            ("sentinel-1-grd", {"start": "2014-01-01", "end": "2014-01-31"}, "2014-04-03"),
        ],
    )
    def test_rejects_dates_before_the_mission_started(
        self, product: str, dates: dict[str, str], first_day: str
    ) -> None:
        assert_rejected(lambda: make_request(product=product, dates=dates), first_day)

    def test_accepts_the_first_day_of_the_mission(self) -> None:
        request = make_request(dates={"start": "2015-06-23", "end": "2015-06-30"})
        assert request.dates.start == date(2015, 6, 23)

    def test_rejects_a_start_in_the_future(self) -> None:
        assert_rejected(lambda: make_request(dates={"start": "2999-01-01", "end": "2999-01-31"}), "future")


class TestStrictness:
    """Misspelt fields and booleans must be refused, never silently dropped or turned into numbers."""

    def test_rejects_an_unknown_field_instead_of_dropping_it(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            make_request(maxCloudCover=5)

    def test_rejects_an_unknown_bbox_field(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            BBox.model_validate({**ROME, "height": 1.0})

    @pytest.mark.parametrize("field", ["max_cloud_cover", "limit"])
    def test_rejects_a_boolean_where_a_number_is_expected(self, field: str) -> None:
        with pytest.raises(ValidationError):
            make_request(**{field: True})

    def test_rejects_a_number_given_as_text(self) -> None:
        with pytest.raises(ValidationError):
            make_request(max_cloud_cover="10")

    def test_accepts_integers_for_coordinates_and_cloud_cover(self) -> None:
        assert BBox(west=12, south=41, east=13, north=42).west == 12
        assert make_request(max_cloud_cover=10).max_cloud_cover == 10


class TestBoundaries:
    def test_accepts_a_box_at_the_area_limit_despite_float_rounding(self) -> None:
        # 17.35 - 12.35 is 4.999999999999998 in floating point; the box is exactly 5 x 5 degrees.
        assert BBox(west=12.35, south=12.35, east=17.35, north=17.35).east == 17.35

    def test_the_last_moment_of_the_end_day_is_inside_the_interval(self) -> None:
        dates = DateRange(start=date(2025, 7, 1), end=date(2025, 7, 31))
        assert dates.to_stac_interval().endswith("23:59:59.999999Z")

    def test_start_today_is_accepted_and_tomorrow_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FrozenClock(datetime):
            @classmethod
            def now(cls, tz: tzinfo | None = None) -> FrozenClock:
                return cls(2025, 12, 15, 23, 30, tzinfo=tz)

        monkeypatch.setattr(domain, "datetime", FrozenClock)
        today = {"start": "2025-12-15", "end": "2025-12-15"}
        assert make_request(dates=today).dates.start == date(2025, 12, 15)
        tomorrow = {"start": "2025-12-16", "end": "2025-12-16"}
        with pytest.raises(ValidationError, match="today is 2025-12-15"):
            make_request(dates=tomorrow)
