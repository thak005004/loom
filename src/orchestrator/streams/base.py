"""StreamAdapter: the one interface every data source implements.

Per Section 2 of the design doc, this is deliberately the *first* thing
built, before any real stream exists. Any source — a sensor feed, a job
queue, a human typing an override — implements `parse()` to translate
its own raw shape into a shared `Event`, and nothing else. The registry,
the bus, and (later) the re-planner and policy only ever see `Event`s;
they never know how many adapter subclasses exist or what they're for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional

from orchestrator.events.bus import EventBus
from orchestrator.events.types import Event


class StreamAdapter(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self._bus: Optional[EventBus] = None

    @abstractmethod
    def parse(self, raw: Any) -> Event:
        """Translate one raw record from this source into a shared Event."""

    def bind(self, bus: EventBus) -> None:
        """Called by the registry on register_stream(). Not meant to be
        called directly — an adapter shouldn't need to know what bus
        implementation it's talking to."""
        self._bus = bus

    def emit(self, raw: Any) -> Event:
        """Parse one raw record and publish the resulting Event onto
        whichever bus this adapter is currently registered with."""
        if self._bus is None:
            raise RuntimeError(
                f"stream adapter '{self.name}' emitted before being registered "
                "via register_stream()"
            )
        event = self.parse(raw)
        self._bus.publish(event)
        return event

    def run(self, raws: Iterable[Any]) -> List[Event]:
        """Convenience for feeding a batch of already-produced raw records
        through parse+publish. Generic across any adapter — it doesn't
        know or care where `raws` came from (a simulator's generator, a
        replayed log, a real queue)."""
        return [self.emit(raw) for raw in raws]
