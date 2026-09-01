"""Tests for the what-if device-failure simulator (scheduling/whatif.py).

The core guarantee under test: simulating a failure must be provably
side-effect-free — the live world, the live assignment map, and the
bandit's learned state must all come out byte-for-byte identical to
what they were going in, even though the simulation runs the exact
same solve() path a real failure would.
"""

from __future__ import annotations

import copy

from orchestrator.events.types import ChangeKind, Event, EventType
from orchestrator.fleet.state import WorldState
from orchestrator.llm.client import MockLLMClient
from orchestrator.policy.bandit_policy import BanditPolicy
from orchestrator.agents.explainer import explain_what_if
from orchestrator.scheduling.whatif import simulate_device_failure


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


def _seed_device(world, device_id, kind, battery=80.0, load=0.2, connected=True):
    world.apply_event(_resource_event(device_id, ChangeKind.ADDED, kind=kind, battery=battery, load=load, connected=connected))


def _seed_job(world, job_id, requires, priority="normal"):
    world.apply_event(_demand_event(job_id, ChangeKind.ADDED, priority=priority, requires=requires))


WEIGHTS = {"priority_weight": 1.0, "load_penalty_scale": 2.0, "battery_bonus_scale": 1.0, "urgent_unassigned_penalty": 500.0}


def test_orphaned_job_moves_to_the_other_capable_device():
    world = WorldState()
    _seed_device(world, "dev-1", "npu")
    _seed_device(world, "dev-2", "npu")
    _seed_job(world, "job-A", requires="npu")
    assignments = {"job-A": "dev-1"}

    result = simulate_device_failure(world, assignments, "dev-1", WEIGHTS)

    assert result.orphaned_job_ids == ["job-A"]
    assert result.moves["job-A"] == "dev-2"


def test_job_becomes_unassigned_when_no_other_device_qualifies():
    world = WorldState()
    _seed_device(world, "dev-1", "gpu")
    _seed_job(world, "job-A", requires="gpu")
    assignments = {"job-A": "dev-1"}

    result = simulate_device_failure(world, assignments, "dev-1", WEIGHTS)

    assert result.moves["job-A"] is None


def test_device_with_no_jobs_reports_no_orphans():
    world = WorldState()
    _seed_device(world, "dev-1", "npu")
    result = simulate_device_failure(world, {}, "dev-1", WEIGHTS)
    assert result.orphaned_job_ids == []
    assert result.device_existed is True


def test_unknown_device_id_is_reported_not_crashed_on():
    world = WorldState()
    result = simulate_device_failure(world, {}, "dev-ghost", WEIGHTS)
    assert result.device_existed is False


def test_simulation_never_mutates_live_world_or_assignments():
    world = WorldState()
    _seed_device(world, "dev-1", "npu")
    _seed_device(world, "dev-2", "npu")
    _seed_job(world, "job-A", requires="npu")
    assignments = {"job-A": "dev-1"}

    devices_before = copy.deepcopy(world.devices)
    jobs_before = copy.deepcopy(world.jobs)
    assignments_before = dict(assignments)

    simulate_device_failure(world, assignments, "dev-1", WEIGHTS)

    assert world.devices == devices_before
    assert world.jobs == jobs_before
    assert assignments == assignments_before


def test_simulation_never_trains_the_bandit():
    world = WorldState()
    _seed_device(world, "dev-1", "npu")
    _seed_device(world, "dev-2", "npu")
    _seed_job(world, "job-A", requires="npu")
    assignments = {"job-A": "dev-1"}

    policy = BanditPolicy()
    theta_before = copy.deepcopy(policy.theta)
    last_arm_before = policy.last_arm_index

    active_weights = dict(policy.arms[0])
    simulate_device_failure(world, assignments, "dev-1", active_weights)

    assert policy.theta == theta_before
    assert policy.last_arm_index == last_arm_before


def test_mock_explainer_names_the_move_and_the_destination_device():
    world = WorldState()
    _seed_device(world, "dev-1", "npu")
    _seed_device(world, "dev-2", "npu")
    _seed_job(world, "job-A", requires="npu")
    result = simulate_device_failure(world, {"job-A": "dev-1"}, "dev-1", WEIGHTS)

    answer = explain_what_if(result, MockLLMClient())

    assert "job-A" in answer
    assert "dev-2" in answer


def test_mock_explainer_reports_unassignment_honestly():
    world = WorldState()
    _seed_device(world, "dev-1", "gpu")
    _seed_job(world, "job-A", requires="gpu")
    result = simulate_device_failure(world, {"job-A": "dev-1"}, "dev-1", WEIGHTS)

    answer = explain_what_if(result, MockLLMClient())

    assert "job-A" in answer
    assert "unassigned" in answer


def test_explainer_handles_a_device_with_nothing_on_it():
    world = WorldState()
    _seed_device(world, "dev-1", "npu")
    result = simulate_device_failure(world, {}, "dev-1", WEIGHTS)

    answer = explain_what_if(result, MockLLMClient())

    assert "dev-1" in answer
