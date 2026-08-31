"""Proves the three properties Section 2 depends on:

  (a) two different adapters can register independently
  (b) events from both flow through the same interface (the shared bus)
  (c) a new adapter can be registered after the system is already
      "running", without touching any existing code

Each test builds its own EventBus/StreamRegistry rather than using the
module-level default, so the tests can't leak state into each other.
"""

from __future__ import annotations

from typing import Any, List

from orchestrator.events.bus import EventBus
from orchestrator.events.types import ChangeKind, Event, EventType
from orchestrator.streams.base import StreamAdapter
from orchestrator.streams.registry import StreamRegistry


class FakeTelemetryAdapter(StreamAdapter):
    """Stands in for a device-telemetry source: raw dict -> resource_changed."""

    def parse(self, raw: dict) -> Event:
        return Event(
            type=EventType.RESOURCE_CHANGED,
            change_kind=ChangeKind.CHANGED,
            source=self.name,
            payload=raw,
        )


class FakeDemandAdapter(StreamAdapter):
    """Stands in for a job-request source: raw dict -> demand_changed."""

    def parse(self, raw: dict) -> Event:
        return Event(
            type=EventType.DEMAND_CHANGED,
            change_kind=ChangeKind.ADDED,
            source=self.name,
            payload=raw,
        )


def test_two_independent_adapters_can_register():
    bus = EventBus()
    registry = StreamRegistry(bus)

    telemetry = FakeTelemetryAdapter("telemetry")
    demand = FakeDemandAdapter("demand")
    registry.register_stream(telemetry)
    registry.register_stream(demand)

    assert len(registry) == 2
    assert {a.name for a in registry.streams} == {"telemetry", "demand"}


def test_events_from_both_adapters_flow_through_the_same_bus():
    bus = EventBus()
    registry = StreamRegistry(bus)
    received: List[Event] = []
    bus.subscribe(received.append)

    telemetry = FakeTelemetryAdapter("telemetry")
    demand = FakeDemandAdapter("demand")
    registry.register_stream(telemetry)
    registry.register_stream(demand)

    telemetry.emit({"device_id": "dev-1", "battery": 12})
    demand.emit({"job_id": "job-1", "priority": "urgent"})

    assert len(received) == 2
    assert received[0].type == EventType.RESOURCE_CHANGED
    assert received[0].source == "telemetry"
    assert received[1].type == EventType.DEMAND_CHANGED
    assert received[1].source == "demand"


def test_new_adapter_registers_after_system_is_already_running():
    bus = EventBus()
    registry = StreamRegistry(bus)
    received: List[Event] = []
    bus.subscribe(received.append)

    telemetry = FakeTelemetryAdapter("telemetry")
    registry.register_stream(telemetry)
    telemetry.emit({"device_id": "dev-1", "battery": 90})  # system is "running"

    # A brand-new stream type shows up later. Nothing above this line
    # is touched to accommodate it.
    class FakeRuleAdapter(StreamAdapter):
        def parse(self, raw: Any) -> Event:
            return Event(
                type=EventType.RULE_CHANGED,
                change_kind=ChangeKind.CHANGED,
                source=self.name,
                payload=raw,
            )

    context = FakeRuleAdapter("context")
    registry.register_stream(context)
    context.emit({"rule": "peak_hours", "active": True})

    assert len(registry) == 2
    assert [e.source for e in received] == ["telemetry", "context"]
    assert received[-1].type == EventType.RULE_CHANGED
