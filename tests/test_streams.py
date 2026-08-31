"""Tests for the three Day-1 streams (Section 2): each emits well-formed
events of the right type/change_kind, and all three run concurrently
through one registry/bus without interfering with each other's state.
"""

from __future__ import annotations

import random
from typing import List

from orchestrator.events.bus import EventBus
from orchestrator.events.types import ChangeKind, Event, EventType
from orchestrator.streams.context_stream import ContextStreamAdapter
from orchestrator.streams.demand_stream import DemandStreamAdapter
from orchestrator.streams.registry import StreamRegistry
from orchestrator.streams.telemetry_stream import TelemetryStreamAdapter


def test_telemetry_stream_emits_well_formed_resource_events():
    adapter = TelemetryStreamAdapter(num_devices=60, rng=random.Random(1))
    events = [adapter.parse(raw) for raw in adapter.generate(500)]

    assert all(e.type == EventType.RESOURCE_CHANGED for e in events)
    assert all(
        e.change_kind in (ChangeKind.ADDED, ChangeKind.REMOVED, ChangeKind.CHANGED)
        for e in events
    )
    assert all("device_id" in e.payload and "kind" in e.payload for e in events)
    assert all(0.0 <= e.payload["battery"] <= 100.0 for e in events)
    # a removed device's battery should have hit zero
    removed = [e for e in events if e.change_kind == ChangeKind.REMOVED]
    assert all(e.payload["battery"] == 0.0 for e in removed)


def test_telemetry_default_device_count_within_spec_range():
    adapter = TelemetryStreamAdapter(rng=random.Random(2))
    assert 50 <= len(adapter.devices) <= 150


def test_demand_stream_emits_well_formed_events_structured_and_nl():
    adapter = DemandStreamAdapter(rng=random.Random(3))
    events = [adapter.parse(raw) for raw in adapter.generate(300)]

    assert all(e.type == EventType.DEMAND_CHANGED for e in events)
    assert all(
        e.change_kind in (ChangeKind.ADDED, ChangeKind.CHANGED, ChangeKind.REMOVED) for e in events
    )
    assert all("job_id" in e.payload for e in events)

    added = [e for e in events if e.change_kind == ChangeKind.ADDED]
    assert any(e.payload.get("structured") is False for e in added)
    assert any(e.payload.get("structured") is True for e in added)
    # messy NL requests pass raw text through untouched, unparsed
    nl_events = [e for e in added if e.payload.get("structured") is False]
    assert all(isinstance(e.payload["text"], str) and e.payload["text"] for e in nl_events)


def test_demand_stream_closes_jobs_with_removed():
    adapter = DemandStreamAdapter(rng=random.Random(8))
    events = [adapter.parse(raw) for raw in adapter.generate(400)]

    removed = [e for e in events if e.change_kind == ChangeKind.REMOVED]
    assert removed, "expected at least one job to close over 400 ticks"
    assert all(e.type == EventType.DEMAND_CHANGED for e in removed)
    assert all("job_id" in e.payload and "reason" in e.payload for e in removed)

    added_ids = {e.payload["job_id"] for e in events if e.change_kind == ChangeKind.ADDED}
    removed_ids = {e.payload["job_id"] for e in removed}
    # can only close jobs that were actually opened
    assert removed_ids <= added_ids
    # a closed job shouldn't still be open in the adapter's own bookkeeping
    assert removed_ids.isdisjoint(adapter._open_job_ids)


def test_context_stream_emits_well_formed_rule_events():
    adapter = ContextStreamAdapter(rng=random.Random(4))
    events = [adapter.parse(raw) for raw in adapter.generate(50)]

    assert all(e.type == EventType.RULE_CHANGED for e in events)
    assert all(e.change_kind == ChangeKind.CHANGED for e in events)
    assert all("rule" in e.payload and "active" in e.payload for e in events)


def test_all_three_streams_run_concurrently_through_same_registry_without_interfering():
    bus = EventBus()
    registry = StreamRegistry(bus)
    received: List[Event] = []
    bus.subscribe(received.append)

    telemetry = TelemetryStreamAdapter(num_devices=60, rng=random.Random(5))
    demand = DemandStreamAdapter(rng=random.Random(6))
    context = ContextStreamAdapter(rng=random.Random(7))
    for adapter in (telemetry, demand, context):
        registry.register_stream(adapter)

    assert len(registry) == 3

    for _ in range(100):
        telemetry.emit(telemetry.next_raw())
        demand.emit(demand.next_raw())
        context.emit(context.next_raw())

    assert len(received) == 300
    by_source = {"telemetry": 0, "demand": 0, "context": 0}
    for e in received:
        by_source[e.source] += 1
    assert by_source == {"telemetry": 100, "demand": 100, "context": 100}

    assert {e.type for e in received if e.source == "telemetry"} == {EventType.RESOURCE_CHANGED}
    assert {e.type for e in received if e.source == "demand"} == {EventType.DEMAND_CHANGED}
    assert {e.type for e in received if e.source == "context"} == {EventType.RULE_CHANGED}
