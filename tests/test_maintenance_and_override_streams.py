"""Tests for the two Day-2 streams (Section 2) that prove the registry
pattern actually holds: maintenance_stream is the first real producer
of CAPABILITY_CHANGED (a handler the re-planner has had since Day 1
Step 5 without ever seeing one), and override_stream reuses existing
event types/change_kinds for human-triggered input with no special
path anywhere downstream. The last test is the demo centerpiece: a
stream registered *after* the system is already running, with zero
changes to any other file.
"""

from __future__ import annotations

import random

from orchestrator.events.bus import EventBus
from orchestrator.events.types import ChangeKind, Event, EventType
from orchestrator.fleet.state import WorldState
from orchestrator.scheduling.replanner import RePlanner
from orchestrator.streams.context_stream import ContextStreamAdapter
from orchestrator.streams.demand_stream import DemandStreamAdapter
from orchestrator.streams.maintenance_stream import MaintenanceStreamAdapter
from orchestrator.streams.override_stream import OverrideStreamAdapter
from orchestrator.streams.registry import StreamRegistry
from orchestrator.streams.telemetry_stream import TelemetryStreamAdapter


def test_maintenance_stream_emits_partial_capability_changed_events():
    adapter = MaintenanceStreamAdapter(device_ids=["dev-1"], rng=random.Random(1))

    event = adapter.parse(adapter.next_raw())

    assert event.type == EventType.RESOURCE_CHANGED
    assert event.change_kind == ChangeKind.CAPABILITY_CHANGED
    assert event.payload["device_id"] == "dev-1"
    assert event.payload["reliability"] == "degraded"
    # deliberately partial — this stream doesn't own kind/battery/load
    assert "kind" not in event.payload
    assert "battery" not in event.payload


def test_maintenance_stream_can_report_recovery_after_degradation():
    adapter = MaintenanceStreamAdapter(device_ids=["dev-1"], rng=random.Random(1))
    degrade_event = adapter.parse(adapter.next_raw())
    assert degrade_event.payload["reliability"] == "degraded"

    recover_event = adapter.parse(adapter.next_raw())
    assert recover_event.change_kind == ChangeKind.CAPABILITY_CHANGED
    assert recover_event.payload["reliability"] == "nominal"


def test_override_stream_reuses_existing_event_types_and_change_kinds():
    adapter = OverrideStreamAdapter(device_ids=["dev-1"], job_ids=["job-1"], rng=random.Random(2))
    events = [adapter.parse(raw) for raw in adapter.generate(50)]

    resource_events = [e for e in events if e.type == EventType.RESOURCE_CHANGED]
    demand_events = [e for e in events if e.type == EventType.DEMAND_CHANGED]
    assert resource_events and demand_events  # both kinds actually occurred over 50 draws

    for e in resource_events:
        assert e.change_kind == ChangeKind.REMOVED  # a pre-existing ChangeKind, not a new one
        assert e.payload["device_id"] == "dev-1"
        assert e.payload["reason"] == "pulled_for_inspection"

    for e in demand_events:
        assert e.change_kind == ChangeKind.CHANGED  # a pre-existing ChangeKind, not a new one
        assert e.payload["job_id"] == "job-1"
        assert e.payload["priority"] == "urgent"

    # nothing about these events is special-cased at the type level —
    # source is the only thing distinguishing "human" input
    assert all(e.source == "override" for e in events)


def test_replanner_capability_changed_handler_works_with_zero_changes_to_replanner():
    world = WorldState()
    world.apply_event(
        Event(type=EventType.RESOURCE_CHANGED, change_kind=ChangeKind.ADDED, source="test", payload={"device_id": "dev-1", "kind": "npu", "battery": 80.0, "load": 0.1, "connected": True})
    )
    world.apply_event(
        Event(type=EventType.DEMAND_CHANGED, change_kind=ChangeKind.ADDED, source="test", payload={"job_id": "job-A", "structured": True, "job_type": "inference_batch", "priority": "normal", "requires": "npu"})
    )
    replanner = RePlanner(world)
    replanner.assignments = {"job-A": "dev-1"}

    maintenance = MaintenanceStreamAdapter(device_ids=["dev-1"], rng=random.Random(3))
    degrade_event = maintenance.parse(maintenance.next_raw())
    assert degrade_event.payload["reliability"] == "degraded"

    result = replanner.on_event(degrade_event)

    assert result is not None  # a real re-solve happened, not a silent no-op
    assert world.devices["dev-1"]["reliability"] == "degraded"
    assert world.devices["dev-1"]["kind"] == "npu"  # not wiped by the partial payload
    assert "job-A" in replanner.assignments  # still validly placed after re-evaluation


def test_registering_maintenance_stream_live_flows_through_world_and_replanner_with_no_restart():
    bus = EventBus()
    registry = StreamRegistry(bus)
    world = WorldState()
    world.attach(bus)
    replanner = RePlanner(world)
    replanner.attach(bus)

    telemetry = TelemetryStreamAdapter(num_devices=15, rng=random.Random(1))
    demand = DemandStreamAdapter(rng=random.Random(2))
    context = ContextStreamAdapter(rng=random.Random(3))
    registry.register_stream(telemetry)
    registry.register_stream(demand)
    registry.register_stream(context)
    assert len(registry) == 3

    # the system is "running": real traffic through the original 3 streams
    for _ in range(80):
        telemetry.emit(telemetry.next_raw())
    for _ in range(30):
        demand.emit(demand.next_raw())
    for _ in range(5):
        context.emit(context.next_raw())

    replanner.full_solve()
    assignments_before = dict(replanner.assignments)
    assert assignments_before  # the original 3 streams alone produced a real schedule

    busy_device_id = next(iter(assignments_before.values()))
    held_job_ids_before = {jid for jid, did in assignments_before.items() if did == busy_device_id}

    # --- the centerpiece moment: register a brand-new stream live ---
    maintenance = MaintenanceStreamAdapter(device_ids=[busy_device_id], rng=random.Random(4))
    registry.register_stream(maintenance)
    assert len(registry) == 4  # the "registered streams" count the demo dashboard would show

    degrade_event = maintenance.emit(maintenance.next_raw())  # single-device pool -> first event is a real transition

    assert degrade_event.type == EventType.RESOURCE_CHANGED
    assert degrade_event.change_kind == ChangeKind.CAPABILITY_CHANGED
    assert degrade_event.payload["reliability"] == "degraded"

    # flowed through WorldState (attached before this stream even
    # existed) with zero changes to fleet/state.py for this stream
    assert world.devices[busy_device_id]["reliability"] == "degraded"
    assert world.devices[busy_device_id]["kind"] in ("cpu", "gpu", "npu")  # original fields intact

    # flowed through the re-planner (attached before this stream
    # existed) with zero changes to replanner.py for this stream
    for jid in held_job_ids_before:
        assert jid in replanner.assignments

    # nothing about the *other* streams' prior assignments moved —
    # this stream's arrival touched only its own slice
    untouched = {jid: did for jid, did in assignments_before.items() if jid not in held_job_ids_before}
    for jid, did in untouched.items():
        assert replanner.assignments.get(jid) == did
