"""Daily provider-token budget, reserved transactionally in PostgreSQL.

Before every provider call the worst case (input estimate + maximum output) is *reserved* under
a row lock on today's ``llm_usage_days`` row, so concurrent requests (and several API processes)
cannot spend past the budget. After the call the reservation is *settled* with the tokens the
provider reported (or the whole reservation if it reported none). A failure that cannot have
consumed tokens releases it; a timeout settles it in full, because the provider may still have
billed the call.

Each reservation is its own row with an expiry. If a process dies between reserve and settle,
its reservation simply stops counting once it expires (default 5 minutes, longer than any call
can last). The cost of that choice is bounded: at most one worst-case call per crash may have
been billed without being counted. ``settle`` still records usage for a call whose reservation
had already expired.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import LlmReservation, LlmUsageDay

DEFAULT_TTL_SECONDS = 300


@dataclass(frozen=True)
class Reservation:
    id: uuid.UUID
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


def _refresh_reserved(session: Session, row: LlmUsageDay, now: datetime) -> int:
    """Drop expired reservations of this day and return what is still reserved."""
    session.execute(
        delete(LlmReservation).where(
            LlmReservation.usage_date == row.usage_date, LlmReservation.expires_at <= now
        )
    )
    live = session.scalar(
        select(func.coalesce(func.sum(LlmReservation.tokens), 0)).where(
            LlmReservation.usage_date == row.usage_date
        )
    )
    row.reserved_tokens = int(live)
    return int(live)


def reserve(
    factory: sessionmaker[Session],
    tokens: int,
    daily_budget: int,
    day: date | None = None,
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
    now: datetime | None = None,
) -> Reservation | None:
    """Reserve ``tokens`` of today's budget, or return None if that would exceed it."""
    day = day or today_utc()
    now = now or datetime.now(UTC)
    with factory() as session:
        row = _locked_row(session, day)
        reserved = _refresh_reserved(session, row, now)
        if tokens <= 0 or reserved + row.used_tokens + tokens > daily_budget:
            session.commit()
            return None
        reservation = Reservation(uuid.uuid4(), day, tokens)
        session.add(
            LlmReservation(
                id=reservation.id,
                usage_date=day,
                tokens=tokens,
                expires_at=now + timedelta(seconds=ttl_seconds),
            )
        )
        row.reserved_tokens = reserved + tokens
        session.commit()
    return reservation


def settle(
    factory: sessionmaker[Session],
    reservation: Reservation,
    actual: int | None,
    now: datetime | None = None,
) -> None:
    """Turn a reservation into usage: the reported tokens, or all of it when none were reported."""
    with factory() as session:
        row = _locked_row(session, reservation.usage_date)
        session.execute(delete(LlmReservation).where(LlmReservation.id == reservation.id))
        row.used_tokens += reservation.tokens if actual is None else max(0, actual)
        _refresh_reserved(session, row, now or datetime.now(UTC))
        session.commit()


def release(
    factory: sessionmaker[Session], reservation: Reservation, now: datetime | None = None
) -> None:
    """Give a reservation back (the call cannot have consumed tokens)."""
    with factory() as session:
        row = _locked_row(session, reservation.usage_date)
        session.execute(delete(LlmReservation).where(LlmReservation.id == reservation.id))
        _refresh_reserved(session, row, now or datetime.now(UTC))
        session.commit()


def snapshot(factory: sessionmaker[Session], day: date | None = None) -> dict:
    day = day or today_utc()
    now = datetime.now(UTC)
    with factory() as session:
        row = session.get(LlmUsageDay, day)
        live = session.scalar(
            select(func.coalesce(func.sum(LlmReservation.tokens), 0)).where(
                LlmReservation.usage_date == day, LlmReservation.expires_at > now
            )
        )
        return {
            "date": day.isoformat(),
            "used_tokens": int(row.used_tokens) if row else 0,
            "reserved_tokens": int(live),
        }
