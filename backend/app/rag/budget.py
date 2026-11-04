"""Daily provider-token budget, reserved transactionally in PostgreSQL.

Before every provider call the worst case (input estimate + maximum output) is *reserved* under
a row lock on today's ``llm_usage_days`` row, so concurrent requests (and several API processes)
cannot spend past the budget. After the call the reservation is *settled* with the tokens the
provider reported (or the whole reservation if it reported none). Failures that cannot have
consumed tokens release the reservation; a timeout keeps it, because the provider may still
have billed it.

If a process dies between reserve and settle, that day's ``reserved_tokens`` stays raised until
UTC midnight. That errs on the side of spending less, never more.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import LlmUsageDay


@dataclass(frozen=True)
class Reservation:
    usage_date: date
    tokens: int


def today_utc() -> date:
    return datetime.now(UTC).date()


def _locked_row(session: Session, day: date) -> LlmUsageDay:
    session.execute(
        pg_insert(LlmUsageDay)
        .values(usage_date=day, reserved_tokens=0, used_tokens=0)
        .on_conflict_do_nothing(index_elements=["usage_date"])
    )
    return session.execute(
        select(LlmUsageDay).where(LlmUsageDay.usage_date == day).with_for_update()
    ).scalar_one()


def reserve(
    factory: sessionmaker[Session], tokens: int, daily_budget: int, day: date | None = None
) -> Reservation | None:
    """Reserve ``tokens`` of today's budget, or return None if that would exceed it."""
    day = day or today_utc()
    with factory() as session:
        row = _locked_row(session, day)
        if tokens <= 0 or row.reserved_tokens + row.used_tokens + tokens > daily_budget:
            session.commit()
            return None
        row.reserved_tokens += tokens
        session.commit()
    return Reservation(day, tokens)


def settle(factory: sessionmaker[Session], reservation: Reservation, actual: int | None) -> None:
    """Turn a reservation into usage: the reported tokens, or all of it when none were reported."""
    with factory() as session:
        row = _locked_row(session, reservation.usage_date)
        row.reserved_tokens = max(0, row.reserved_tokens - reservation.tokens)
        row.used_tokens += reservation.tokens if actual is None else max(0, actual)
        session.commit()


def release(factory: sessionmaker[Session], reservation: Reservation) -> None:
    """Give a reservation back (the call cannot have consumed tokens)."""
    with factory() as session:
        row = _locked_row(session, reservation.usage_date)
        row.reserved_tokens = max(0, row.reserved_tokens - reservation.tokens)
        session.commit()


def snapshot(factory: sessionmaker[Session], day: date | None = None) -> dict:
    day = day or today_utc()
    with factory() as session:
        row = session.get(LlmUsageDay, day)
        return {
            "date": day.isoformat(),
            "used_tokens": int(row.used_tokens) if row else 0,
            "reserved_tokens": int(row.reserved_tokens) if row else 0,
        }
