from __future__ import annotations

import random

from orchestrator.events.bus import EventBus
from orchestrator.events.types import ChangeKind, Event, EventType
from orchestrator.fleet.state import WorldState
from orchestrator.streams.context_stream import ContextStreamAdapter
from orchestrator.streams.demand_stream import DemandStreamAdapter
from orchestrator.streams.registry import StreamRegistry
from orchestrator.streams.telemetry_stream import TelemetryStreamAdapter


def _resource_event(device_id, change_kind, **fields) -> Event:
    return Event(
        type=EventType.RESOURCE_CHANGED,
        change_kind=change_kind,
        source="test",
        payload={"device_id": device_id, **fields},
    )


def _demand_event(job_id, change_kind, **fields) -> Event:
    return Event(
        type=EventType.DEMAND_CHANGED,
        change_kind=change_kind,
        source="test",
        payload={"job_id": job_id, **fields},
    )


def test_resource_added_then_changed_then_removed():
    world = WorldState()

    world.apply_event(_resource_event("dev-1", ChangeKind.ADDED, kind="gpu", battery=90.0, load=0.1))
    assert world.devices["dev-1"]["battery"] == 90.0

    world.apply_event(_resource_event("dev-1", ChangeKind.CHANGED, kind="gpu", battery=70.0, load=0.4))
    assert world.devices["dev-1"]["battery"] == 70.0
    assert world.devices["dev-1"]["load"] == 0.4

    world.apply_event(_resource_event("dev-1", ChangeKind.REMOVED, kind="gpu", battery=0.0, load=0.4))
    assert "dev-1" not in world.devices


def test_demand_added_then_changed_merges_then_removed_clears():
    world = WorldState()

    world.apply_event(
        _demand_event("job-1", ChangeKind.ADDED, structured=True, job_type="camera_check", priority="normal", requires="gpu")
    )
    assert world.jobs["job-1"]["priority"] == "normal"
    assert world.jobs["job-1"]["requires"] == "gpu"

    # a priority-change event only carries job_id + priority — everything
    # else on the job record should survive the merge
    world.apply_event(_demand_event("job-1", ChangeKind.CHANGED, priority="urgent"))
    assert world.jobs["job-1"]["priority"] == "urgent"
    assert world.jobs["job-1"]["requires"] == "gpu"
    assert world.jobs["job-1"]["job_type"] == "camera_check"

    world.apply_event(_demand_event("job-1", ChangeKind.REMOVED, reason="completed"))
    assert "job-1" not in world.jobs


def test_partial_capability_changed_event_does_not_wipe_other_device_fields():
    world = WorldState()
    world.apply_event(_resource_event("dev-1", ChangeKind.ADDED, kind="gpu", battery=90.0, load=0.1, connected=True))

    # a source that only knows about one signal (e.g. a maintenance
    # stream reporting reliability) shouldn't be able to blow away the
    # rest of the device's known state just by not mentioning it
    world.apply_event(
        Event(
            type=EventType.RESOURCE_CHANGED,
            change_kind=ChangeKind.CAPABILITY_CHANGED,
            source="test",
            payload={"device_id": "dev-1", "reliability": "degraded"},
        )
    )

    assert world.devices["dev-1"]["reliability"] == "degraded"
    assert world.devices["dev-1"]["kind"] == "gpu"
    assert world.devices["dev-1"]["battery"] == 90.0
    assert world.devices["dev-1"]["load"] == 0.1


def test_removed_device_no_longer_appears_in_available_devices():
    world = WorldState()
    world.apply_event(_resource_event("dev-1", ChangeKind.ADDED, kind="gpu", battery=90.0, load=0.1))
    assert any(d["device_id"] == "dev-1" for d in world.available_devices())

    world.apply_event(_resource_event("dev-1", ChangeKind.REMOVED, kind="gpu", battery=0.0, load=0.1))

    assert "dev-1" not in world.devices
    assert all(d["device_id"] != "dev-1" for d in world.available_devices())


def test_removed_job_no_longer_appears_in_open_jobs():
    world = WorldState()
    world.apply_event(
        _demand_event("job-1", ChangeKind.ADDED, structured=True, job_type="camera_check", priority="normal", requires="gpu")
    )
    assert any(j["job_id"] == "job-1" for j in world.open_jobs())

    world.apply_event(_demand_event("job-1", ChangeKind.REMOVED, reason="completed"))

    assert "job-1" not in world.jobs
    assert all(j["job_id"] != "job-1" for j in world.open_jobs())


def test_rule_changed_event_is_a_harmless_no_op():
    world = WorldState()
    event = Event(type=EventType.RULE_CHANGED, change_kind=ChangeKind.CHANGED, source="test", payload={"rule": "peak_hours", "active": True})
    world.apply_event(event)  # should not raise
    assert world.devices == {}
    assert world.jobs == {}


def test_world_state_stays_live_when_attached_to_a_running_bus():
    bus = EventBus()
    registry = StreamRegistry(bus)
    world = WorldState()
    world.attach(bus)

    telemetry = TelemetryStreamAdapter(num_devices=10, rng=random.Random(1))
    registry.register_stream(telemetry)

    for _ in range(50):
        telemetry.emit(telemetry.next_raw())

    # world state mirrors the adapter's own live device set exactly
    assert set(world.devices) == set(telemetry.devices)
    assert world.available_devices()
    for device in world.available_devices():
        assert 0.0 <= device["battery"] <= 100.0

    # a stream registered after the system is already "running" flows
    # into the same WorldState with no changes needed here
    demand = DemandStreamAdapter(rng=random.Random(2))
    registry.register_stream(demand)
    for _ in range(30):
        demand.emit(demand.next_raw())

    assert world.open_jobs()
    for job in world.open_jobs():
        assert "job_id" in job

    context = ContextStreamAdapter(rng=random.Random(3))
    registry.register_stream(context)
    context.emit(context.next_raw())  # must not raise or pollute devices/jobs
    assert set(world.devices) == set(telemetry.devices)
