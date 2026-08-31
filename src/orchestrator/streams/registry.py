"""The stream registry: the pluggability mechanism from Section 2.

`StreamRegistry.register_stream(adapter)` is the entire contract a new
data source has to satisfy to join the system — including one that
shows up after the system has already started. Registering an adapter
just binds it to the shared bus and records it for introspection (e.g.
a dashboard's "registered streams" count); it never touches any other
adapter, and nothing downstream needs to change to support it.
"""

from __future__ import annotations

from typing import Dict, List

from orchestrator.events.bus import EventBus
from orchestrator.streams.base import StreamAdapter


class StreamRegistry:
    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self._adapters: Dict[str, StreamAdapter] = {}

    def register_stream(self, adapter: StreamAdapter) -> None:
        if adapter.name in self._adapters:
            raise ValueError(f"a stream named '{adapter.name}' is already registered")
        adapter.bind(self.bus)
        self._adapters[adapter.name] = adapter

    @property
    def streams(self) -> List[StreamAdapter]:
        return list(self._adapters.values())

    def __len__(self) -> int:
        return len(self._adapters)


# A default, module-level registry + bus, so `register_stream(adapter)`
# works standalone exactly as the design doc describes it, without every
# caller having to wire up a registry by hand. Real code (and the demo)
# can use this; tests construct their own StreamRegistry/EventBus pair
# instead, so test cases can't leak adapters into each other via shared
# global state.
default_bus = EventBus()
default_registry = StreamRegistry(default_bus)


def register_stream(adapter: StreamAdapter) -> None:
    default_registry.register_stream(adapter)
