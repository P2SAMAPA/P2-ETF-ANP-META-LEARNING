"""us_calendar.py — US trading calendar helper."""

from __future__ import annotations

from datetime import date, timedelta

# NYSE holidays (fixed-date approximation; covers common market closures)
_FIXED_HOLIDAYS = {
    (1,  1),   # New Year's Day
    (7,  4),   # Independence Day
    (12, 25),  # Christmas Day
}


def _is_trading_day(d: date) -> bool:
    """Return True if d is a US equity trading day (rough approximation)."""
    if d.weekday() >= 5:          # Saturday or Sunday
        return False
    if (d.month, d.day) in _FIXED_HOLIDAYS:
        return False
    return True


def next_trading_day(from_date: date | None = None) -> str:
    """Return the next US trading day as YYYY-MM-DD string."""
    d = (from_date or date.today()) + timedelta(days=1)
    while not _is_trading_day(d):
        d += timedelta(days=1)
    return d.strftime("%Y-%m-%d")
