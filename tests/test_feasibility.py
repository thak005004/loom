"""Tests for the feasibility layer (fleet/feasibility.py): "what
categories of work could we take on right now," derived purely from
live WorldState, with zero hardcoded device kinds.
"""

from __future__ import annotations

from orchestrator.events.types import ChangeKind, Event, EventType
from orchestrator.fleet.feasibility import feasible_categories
from orchestrator.fleet.state import WorldState


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


def _seed_device(world, device_id, kind, connected=True, capacity=None):
    fields = {"kind": kind, "battery": 80.0, "load": 0.2, "connected": connected}
    if capacity is not None:
        fields["capacity"] = capacity
    world.apply_event(_resource_event(device_id, ChangeKind.ADDED, **fields))


def _seed_job(world, job_id, requires, priority="normal"):
    world.apply_event(_demand_event(job_id, ChangeKind.ADDED, priority=priority, requires=requires))


def test_category_feasible_when_a_capable_device_exists():
    world = WorldState()
    _seed_device(world, "dev-1", "gpu")
    _seed_job(world, "job-A", requires="gpu")

    assert feasible_categories(world)["gpu"] is True


def test_category_becomes_infeasible_when_the_only_capable_device_is_removed():
    world = WorldState()
    _seed_device(world, "dev-1", "gpu")
    _seed_job(world, "job-A", requires="gpu")
    assert feasible_categories(world)["gpu"] is True

    world.apply_event(_resource_event("dev-1", ChangeKind.REMOVED))

    assert feasible_categories(world)["gpu"] is False


def test_category_becomes_feasible_again_once_a_new_capable_device_joins():
    world = WorldState()
    _seed_job(world, "job-A", requires="gpu")
    assert feasible_categories(world)["gpu"] is False

    _seed_device(world, "dev-2", "gpu")

    assert feasible_categories(world)["gpu"] is True


def test_category_with_no_capable_device_anywhere_is_reported_infeasible_not_omitted():
    world = WorldState()
    _seed_device(world, "dev-1", "cpu")
    _seed_job(world, "job-A", requires="fpga")

    result = feasible_categories(world)
    assert "fpga" in result
    assert result["fpga"] is False


def test_disconnected_device_does_not_count_as_capable_supply():
    world = WorldState()
    _seed_device(world, "dev-1", "gpu", connected=False)
    _seed_job(world, "job-A", requires="gpu")

    assert feasible_categories(world)["gpu"] is False


def test_zero_capacity_device_does_not_count_as_capable_supply():
    world = WorldState()
    _seed_device(world, "dev-1", "gpu", capacity=0)
    _seed_job(world, "job-A", requires="gpu")

    assert feasible_categories(world)["gpu"] is False


def test_unconstrained_requires_none_is_feasible_whenever_any_device_has_capacity():
    world = WorldState()
    _seed_device(world, "dev-1", "cpu")

    assert feasible_categories(world)[None] is True


def test_unconstrained_requires_none_is_infeasible_with_an_empty_fleet():
    world = WorldState()
    assert feasible_categories(world)[None] is False


def test_feasibility_never_mutates_world_state():
    world = WorldState()
    _seed_device(world, "dev-1", "gpu")
    _seed_job(world, "job-A", requires="gpu")
    devices_before = dict(world.devices)
    jobs_before = dict(world.jobs)

    feasible_categories(world)

    assert world.devices == devices_before
    assert world.jobs == jobs_before
