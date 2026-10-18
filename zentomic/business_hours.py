"""Bounded, offline business-hours routing from trusted workspace-local time."""

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from itertools import islice
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from zentomic.routing import _valid_target


MAX_DAILY_WINDOWS = 24
MAX_DATE_OVERRIDES = 366
MINUTES_PER_DAY = 1440


def resolve_business_hours_target_at(
    hours: Mapping[int, Sequence[tuple[int, int]]], *,
    instant: datetime, timezone_name: str, open_target: str, closed_target: str,
    date_overrides: Mapping[date, Sequence[tuple[int, int]]] | None = None,
) -> str:
    """Resolve weekly hours at an explicit trusted, timezone-aware instant.

    The caller supplies a fresh clock sample and an authorized IANA workspace
    timezone. No clock is read and no system default timezone is inferred.
    System zoneinfo data (or separately installed tzdata) must be available;
    invalid/missing zones and unrepresentable conversions fail closed.
    UTC round-tripping ensures repeated/skipped local times are resolved from
    the absolute instant, even when its tzinfo is already the workspace zone.
    Optional exact local-date overrides replace (not extend) weekly hours for
    that date. An empty list means closed. At most 366 dates are accepted, all
    validated even when inactive. No holiday calendar is inferred or fetched.
    Clock freshness, call authorization and budgets remain external.
    """
    if not isinstance(instant, datetime):
        raise ValueError("instant must be a timezone-aware datetime")
    if not _valid_target(timezone_name):
        raise ValueError("timezone_name must identify an available IANA timezone")
    try:
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ValueError
        utc = instant.astimezone(timezone.utc)
    except (ValueError, OverflowError, TypeError):
        raise ValueError("instant must be a representable timezone-aware datetime") from None
    try:
        zone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError, UnicodeError):
        raise ValueError("timezone_name must identify an available IANA timezone") from None
    try:
        local = utc.astimezone(zone)
    except (ValueError, OverflowError):
        raise ValueError("instant cannot be represented in the workspace timezone") from None
    weekday = local.weekday()
    minute = local.hour * 60 + local.minute
    target = resolve_business_hours_target(
        hours, weekday=weekday, minute_of_day=minute,
        open_target=open_target, closed_target=closed_target,
    )
    if date_overrides is None:
        return target
    if not isinstance(date_overrides, Mapping):
        raise ValueError("date_overrides must be a local-date mapping")
    dates = tuple(islice(date_overrides, MAX_DATE_OVERRIDES + 1))
    if len(dates) > MAX_DATE_OVERRIDES:
        raise ValueError("date_overrides must contain at most 366 dates")
    for day in dates:
        # Reject datetime (a date subclass) and ambiguous string/annual keys.
        if type(day) is not date:
            raise ValueError("date_overrides keys must be exact date objects")
        override_target = resolve_business_hours_target(
            {weekday: date_overrides[day]}, weekday=weekday, minute_of_day=minute,
            open_target=open_target, closed_target=closed_target,
        )
        if day == local.date():
            target = override_target
    return target


def resolve_business_hours_target(
    hours: Mapping[int, Sequence[tuple[int, int]]], *,
    weekday: int, minute_of_day: int, open_target: str, closed_target: str,
) -> str:
    """Choose an opaque workspace target using a weekly opening-hours policy.

    Monday is 0 and Sunday is 6. Each day has up to 24 (start, end) windows
    measured in local minutes since midnight. Start is inclusive, end exclusive;
    1440 is allowed only as an end. Missing days and empty lists mean closed.
    Unordered/overlapping windows are a union. Split overnight hours across two
    days, for example Monday (1320, 1440) and Tuesday (0, 120).

    All configuration is validated, including inactive days. Inputs must come
    from authorized workspace configuration and a fresh trusted clock converted
    to that workspace's timezone, not callback fields or the server's timezone.
    This function reads no clock or timezone database. Repeated local minutes
    during a DST fallback follow the same schedule; holidays/date exceptions
    must be handled separately. Minute precision intentionally ignores seconds.

    Returned identifiers are unchanged, not phone numbers or permission to dial.
    Authenticate, bind, deduplicate, check budgets/terminal state, and authorize
    the selected target before acting. This does not persist or contact services.
    """
    if not _valid_target(open_target) or not _valid_target(closed_target):
        raise ValueError("business-hours targets must be nonblank UTF-8 strings of at most 256 bytes")
    if type(weekday) is not int or not 0 <= weekday <= 6:
        raise ValueError("weekday must be an integer from 0 to 6")
    if type(minute_of_day) is not int or not 0 <= minute_of_day < MINUTES_PER_DAY:
        raise ValueError("minute_of_day must be an integer from 0 to 1439")
    if not isinstance(hours, Mapping):
        raise ValueError("hours must be a weekday mapping")
    # Bound arbitrary mappings and list subclasses before allocating snapshots.
    days = tuple(islice(hours, 8))
    if len(days) > 7:
        raise ValueError("hours must contain at most 7 days")
    active = False
    for day in days:
        if type(day) is not int or not 0 <= day <= 6:
            raise ValueError("hours keys must be integers from 0 to 6")
        windows = hours[day]
        if not isinstance(windows, (list, tuple)):
            raise ValueError("daily hours must be a list or tuple of at most 24 windows")
        snapshot = tuple(islice(windows, MAX_DAILY_WINDOWS + 1))
        if len(snapshot) > MAX_DAILY_WINDOWS:
            raise ValueError("daily hours must be a list or tuple of at most 24 windows")
        for window in snapshot:
            if not isinstance(window, (list, tuple)):
                raise ValueError("each hours window must contain start and end minutes")
            endpoints = tuple(islice(window, 3))
            if len(endpoints) != 2:
                raise ValueError("each hours window must contain start and end minutes")
            start, end = endpoints
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= MINUTES_PER_DAY):
                raise ValueError("hours windows must satisfy 0 <= start < end <= 1440")
            if day == weekday and start <= minute_of_day < end:
                active = True
    return open_target if active else closed_target
