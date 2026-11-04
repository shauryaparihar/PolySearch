"""R6-style hardening: reservations that expire after a crash, slots shared across processes."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta

import pytest
from alembic import command
from answer_support import ask
from app.config import Settings
from app.db.models import LlmReservation, LlmUsageDay
from app.rag import budget
from app.rag.fake import FakeLLMClient
from app.rag.guards import GenerationBusy, GenerationGuard
from conftest import reset_schema
from sqlalchemy import func, inspect, select, text

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
DAY = T0.date()


def reserved_rows(factory) -> int:
    with factory() as session:
        return session.scalar(select(func.count()).select_from(LlmReservation))


def usage(factory) -> LlmUsageDay:
    with factory() as session:
        return session.scalar(select(LlmUsageDay))


# ================================================================ expiring reservations
def test_a_crashed_call_stops_holding_budget_once_its_reservation_expires(session_factory):
    lost = budget.reserve(session_factory, 900, 1000, day=DAY, ttl_seconds=60, now=T0)
    assert lost is not None  # ... and the process "crashes" here: it is never settled
    soon = T0 + timedelta(seconds=30)
    assert budget.reserve(session_factory, 500, 1000, day=DAY, ttl_seconds=60, now=soon) is None
    later = T0 + timedelta(seconds=61)
    assert budget.reserve(session_factory, 500, 1000, day=DAY, ttl_seconds=60, now=later)
    assert reserved_rows(session_factory) == 1  # the dead one was removed
    assert usage(session_factory).reserved_tokens == 500


def test_a_live_reservation_keeps_counting_until_its_expiry(session_factory):
    assert budget.reserve(session_factory, 900, 1000, day=DAY, ttl_seconds=60, now=T0)
    edge = T0 + timedelta(seconds=59)
    assert budget.reserve(session_factory, 200, 1000, day=DAY, ttl_seconds=60, now=edge) is None


def test_settling_after_expiry_still_records_what_was_used(session_factory):
    slow = budget.reserve(session_factory, 400, 1000, day=DAY, ttl_seconds=10, now=T0)
    budget.reserve(
        session_factory, 100, 1000, day=DAY, ttl_seconds=10, now=T0 + timedelta(seconds=20)
    )
    budget.settle(session_factory, slow, 350, now=T0 + timedelta(seconds=21))  # finished late
    row = usage(session_factory)
    assert row.used_tokens == 350  # a call that really happened is never forgotten


def test_releasing_an_expired_or_unknown_reservation_is_harmless(session_factory):
    gone = budget.reserve(session_factory, 400, 1000, day=DAY, ttl_seconds=10, now=T0)
    budget.reserve(
        session_factory, 1, 1000, day=DAY, ttl_seconds=10, now=T0 + timedelta(seconds=30)
    )
    budget.release(session_factory, gone, now=T0 + timedelta(seconds=31))
    budget.release(session_factory, gone, now=T0 + timedelta(seconds=31))  # twice
    assert usage(session_factory).used_tokens == 0


def test_the_stored_total_always_matches_the_live_reservations(session_factory):
    a = budget.reserve(session_factory, 300, 5000, day=DAY, now=T0)
    b = budget.reserve(session_factory, 200, 5000, day=DAY, now=T0)
    budget.settle(session_factory, a, 120, now=T0)
    assert (
        usage(session_factory).reserved_tokens == 200 and usage(session_factory).used_tokens == 120
    )
    budget.release(session_factory, b, now=T0)
    assert usage(session_factory).reserved_tokens == 0 and reserved_rows(session_factory) == 0


def test_the_snapshot_ignores_expired_reservations(session_factory):
    budget.reserve(session_factory, 400, 1000, ttl_seconds=0.01)
    time.sleep(0.05)
    assert budget.snapshot(session_factory)["reserved_tokens"] == 0


def test_expiry_cannot_be_shorter_than_the_longest_possible_call():
    base = dict(_env_file=None)
    with pytest.raises(ValueError, match="RAG_RESERVATION_TTL_SECONDS"):
        Settings(**base, llm_timeout_seconds=100, rag_reservation_ttl_seconds=150)
    assert Settings(**base).rag_reservation_ttl_seconds == 300  # the default is safe


def test_concurrent_reservations_still_cannot_exceed_the_budget_with_expiry(session_factory):
    granted, lock = [], threading.Lock()
    barrier = threading.Barrier(20)

    def worker():
        barrier.wait()
        result = budget.reserve(session_factory, 100, 1000)
        with lock:
            granted.append(result is not None)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(granted) == 10 and reserved_rows(session_factory) == 10


def test_0006_adds_and_removes_only_the_reservations_table(engine, alembic_config):
    reset_schema(engine)
    command.upgrade(alembic_config, "head")
    assert "llm_reservations" in set(inspect(engine).get_table_names())
    command.downgrade(alembic_config, "0005")
    names = set(inspect(engine).get_table_names())
    assert "llm_reservations" not in names and "llm_usage_days" in names
    command.upgrade(alembic_config, "head")
    assert "llm_reservations" in set(inspect(engine).get_table_names())


# ========================================================== slots shared across processes
def two_processes(engine, slots=1, wait=0.15):
    """Two guards on one database: what two API processes look like to PostgreSQL."""
    return GenerationGuard(slots, wait, engine=engine), GenerationGuard(slots, wait, engine=engine)


def test_a_slot_held_by_one_process_blocks_another(migrated_engine):
    first, second = two_processes(migrated_engine)
    with first.slot():
        started = time.monotonic()
        with pytest.raises(GenerationBusy), second.slot():
            pass
        assert time.monotonic() - started < 2
    with second.slot():  # free again once the first finished
        pass


def test_the_limit_counts_slots_not_processes(migrated_engine):
    first, second = two_processes(migrated_engine, slots=2)
    third = GenerationGuard(2, 0.15, engine=migrated_engine)
    with first.slot(), second.slot():
        with pytest.raises(GenerationBusy), third.slot():
            pass
    with third.slot():
        pass


def test_a_slot_is_released_when_the_call_fails(migrated_engine):
    first, second = two_processes(migrated_engine)
    with pytest.raises(RuntimeError), first.slot():
        raise RuntimeError("provider exploded")
    with second.slot():
        pass


def test_a_dead_process_cannot_keep_a_slot(migrated_engine):
    from app.rag.guards import LOCK_NAMESPACE

    guard = GenerationGuard(1, 0.15, engine=migrated_engine)
    dying = migrated_engine.connect().execution_options(isolation_level="AUTOCOMMIT")
    try:
        dying.execute(text("select pg_advisory_lock(:ns, 0)"), {"ns": LOCK_NAMESPACE})
        with pytest.raises(GenerationBusy), guard.slot():
            pass
    finally:
        dying.invalidate()  # the socket closes, exactly as when a process is killed
        dying.close()
    with guard.slot():  # PostgreSQL released the dead connection's lock by itself
        pass


def test_without_a_database_the_local_limit_still_applies():
    guard = GenerationGuard(1, 0.05)
    with guard.slot():
        with pytest.raises(GenerationBusy), guard.slot():
            pass
    with guard.slot():
        pass


def test_waiting_requests_get_the_slot_when_it_frees_up(migrated_engine):
    first, second = two_processes(migrated_engine, wait=2.0)
    got = []

    def waiter():
        with second.slot():
            got.append(True)

    with first.slot():
        thread = threading.Thread(target=waiter)
        thread.start()
        time.sleep(0.2)
        assert got == []
    thread.join(3)
    assert got == [True]


def test_the_answer_endpoint_falls_back_when_another_process_holds_the_slot(
    make_answer_client,
):
    llm = FakeLLMClient("success")
    client = make_answer_client(
        llm, generation_max_concurrency=1, inference_wait_timeout_seconds=0.1
    )
    engine = client.app.state.polysearch.session_factory.kw["bind"]
    other_process = GenerationGuard(1, 0.1, engine=engine)
    with other_process.slot():
        body = ask(client).json()
    assert body["fallback_reason"] == "BUSY" and llm.calls == [] and body["search_results"]
    assert ask(client).json()["answer"]  # and it recovers
