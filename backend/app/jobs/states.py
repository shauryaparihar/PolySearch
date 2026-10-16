"""Job status transitions. Terminal states never change again."""

from __future__ import annotations

ACTIVE = ("queued", "running")
TERMINAL = ("completed", "partial", "failed")
ALLOWED: dict[str, set[str]] = {
    "queued": {"running", "failed"},  # queued -> failed: shutdown/restart before it started
    "running": {"completed", "partial", "failed"},
    "completed": set(),
    "partial": set(),
    "failed": set(),
}


class InvalidTransition(ValueError):
    pass


def check_transition(old: str, new: str) -> None:
    if new not in ALLOWED.get(old, set()):
        raise InvalidTransition(f"job cannot change from {old!r} to {new!r}")
