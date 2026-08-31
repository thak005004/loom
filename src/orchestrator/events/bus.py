"""The unified event stream (Section 2 / Section 7 of the design doc).

For this step it's deliberately the simplest thing that could work: a
synchronous fan-out list of subscribers. There's no scheduler or
re-planner yet to subscribe, so this only needs to prove that events
published by *any* adapter reach *every* subscriber, uniformly. The
generator/asyncio.Queue-based incremental processing described in the
plan is a later step and slots in here without changing this
interface's shape (subscribe/publish).
"""

from __future__ import annotations

from typing import Callable, List

from orchestrator.events.types import Event

Handler = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._handlers: List[Handler] = []

    def subscribe(self, handler: Handler) -> None:
        self._handlers.append(handler)

    def publish(self, event: Event) -> None:
        for handler in self._handlers:
            handler(event)
