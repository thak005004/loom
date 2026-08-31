"""Tests for the incremental re-planner (Section 3b/4). The functional
tests build small, hand-precise WorldStates so the expected slice is
unambiguous; the timing tests build larger synthetic fleets to get a
real full-solve-vs-incremental number at a couple of scales, per
Section 11.
"""

from __future__ import annotations

import random
import time

from orchestrator.events.types import ChangeKind, Event, EventType
from orchestrator.fleet.state import WorldState
from orchestrator.policy.bandit_policy import BanditPolicy
from orchestrator.scheduling.replanner import RePlanner


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


def _rule_event(rule, **fields) -> Event:
    return Event(
        type=EventType.RULE_CHANGED,
        change_kind=ChangeKind.CHANGED,
        source="test",
        payload={"rule": rule, **fields},
    )


def _seed_device(world, device_id, kind, battery=80.0, load=0.2, connected=True):
    world.apply_event(_resource_event(device_id, ChangeKind.ADDED, kind=kind, battery=battery, load=load, connected=connected))


def _seed_job(world, job_id, requires, priority="normal", job_type=None):
    world.apply_event(
        _demand_event(job_id, ChangeKind.ADDED, structured=True, job_type=job_type, priority=priority, requires=requires)
    )


def test_device_removed_mid_job_reassigns_just_that_jobs_work():
    world = WorldState()
    _seed_device(world, "dev-1", "npu")
    _seed_device(world, "dev-2", "npu")
    _seed_job(world, "job-A", requires="npu")
    _seed_job(world, "job-B", requires="npu")

    replanner = RePlanner(world)
    replanner.assignments = {"job-A": "dev-1", "job-B": "dev-2"}

    result = replanner.on_event(_resource_event("dev-1", ChangeKind.REMOVED, kind="npu", battery=0.0, load=0.2, connected=False))

    assert "dev-1" not in world.devices
    assert replanner.assignments["job-B"] == "dev-2"  # untouched
    assert replanner.assignments["job-A"] == "dev-2"  # reassigned onto the surviving device
    # the solve call itself only ever considered the orphaned job
    assert {a.job_id for a in result.assignments} <= {"job-A"}


def test_new_device_joining_does_not_touch_existing_assignments():
    world = WorldState()
    _seed_device(world, "dev-1", "npu")
    _seed_job(world, "job-A", requires="npu")
    _seed_job(world, "job-B", requires="npu")
    _seed_job(world, "job-C", requires="npu")  # pending: no capacity for it yet

    replanner = RePlanner(world)
    replanner.assignments = {"job-A": "dev-1", "job-B": "dev-1"}  # dev-1 (capacity 2) already full

    result = replanner.on_event(_resource_event("dev-2", ChangeKind.ADDED, kind="npu", battery=90.0, load=0.0, connected=True))

    assert replanner.assignments["job-A"] == "dev-1"
    assert replanner.assignments["job-B"] == "dev-1"
    assert replanner.assignments["job-C"] == "dev-2"
    assert {a.job_id for a in result.assignments} == {"job-C"}


def test_job_priority_change_only_affects_that_jobs_slice():
    world = WorldState()
    _seed_device(world, "dev-1", "npu")
    _seed_device(world, "dev-2", "npu")
    _seed_job(world, "job-A", requires="npu", priority="normal")
    _seed_job(world, "job-B", requires="npu", priority="normal")

    replanner = RePlanner(world)
    replanner.assignments = {"job-A": "dev-1", "job-B": "dev-2"}

    result = replanner.on_event(_demand_event("job-A", ChangeKind.CHANGED, priority="urgent"))

    assert replanner.assignments["job-B"] == "dev-2"  # never in the slice
    assert replanner.assignments["job-A"] in ("dev-1", "dev-2")  # still validly assigned
    assert {a.job_id for a in result.assignments} <= {"job-A"}


def test_rule_change_only_affects_assignments_it_actually_applies_to():
    # job-quality starts *unconstrained* on a non-GPU device — the rule
    # must actually impose the GPU requirement and move it, not just
    # re-solve it unchanged under the same constraints it already had.
    world = WorldState()
    _seed_device(world, "dev-cpu", "cpu")
    _seed_device(world, "dev-gpu", "gpu")
    _seed_job(world, "job-quality", requires=None, job_type="quality_scan")
    _seed_job(world, "job-camera", requires="gpu", job_type="camera_check")

    replanner = RePlanner(world)
    replanner.assignments = {"job-quality": "dev-cpu", "job-camera": "dev-gpu"}

    result = replanner.on_event(_rule_event("quality_scan_requires_gpu", active=True))

    # the rule left a real, persistent mark on the job's own record
    assert world.jobs["job-quality"]["requires"] == "gpu"
    # ...which the solver's existing capability matching then enforces:
    # the job actually moved onto a GPU-capable device, not the CPU one
    assert replanner.assignments["job-quality"] == "dev-gpu"
    assert replanner.assignments["job-camera"] == "dev-gpu"  # not a quality_scan job — untouched
    assert {a.job_id for a in result.assignments} <= {"job-quality"}


def test_rule_with_no_operational_mapping_is_a_no_op():
    world = WorldState()
    _seed_device(world, "dev-1", "gpu")
    _seed_job(world, "job-camera", requires="gpu", job_type="camera_check")
    replanner = RePlanner(world)
    replanner.assignments = {"job-camera": "dev-1"}

    result = replanner.on_event(_rule_event("peak_hours", active=True))

    assert result is None
    assert replanner.assignments == {"job-camera": "dev-1"}


def test_routine_telemetry_tick_does_not_trigger_a_replan():
    world = WorldState()
    _seed_device(world, "dev-1", "gpu", battery=80.0)
    replanner = RePlanner(world)

    result = replanner.on_event(_resource_event("dev-1", ChangeKind.CHANGED, kind="gpu", battery=79.0, load=0.25, connected=True))

    assert result is None
    assert world.devices["dev-1"]["battery"] == 79.0  # world still updates even though replanner no-ops


def test_closed_job_is_dropped_with_no_solver_call():
    world = WorldState()
    _seed_device(world, "dev-1", "gpu")
    _seed_job(world, "job-A", requires="gpu")
    replanner = RePlanner(world)
    replanner.assignments = {"job-A": "dev-1"}

    result = replanner.on_event(_demand_event("job-A", ChangeKind.REMOVED, reason="completed"))

    assert result is None
    assert "job-A" not in replanner.assignments


def test_incremental_replan_feeds_reward_back_into_the_bandit_policy():
    """The gap that used to exist: RePlanner had no reference to any
    policy, so nothing about an incremental re-plan ever trained the
    bandit unless some caller (the dashboard) remembered to select
    weights and call update() by hand — and nothing enforced that. Now
    RePlanner sources weights from the policy and scores every real
    solve itself. This proves it for two different incremental event
    kinds, not just a full solve()."""
    world = WorldState()
    _seed_device(world, "dev-1", "npu")
    _seed_device(world, "dev-2", "npu")
    _seed_job(world, "job-A", requires="npu")
    _seed_job(world, "job-B", requires="npu")

    policy = BanditPolicy(rng=random.Random(1))
    update_calls = []
    real_update = policy.update

    def spy_update(reward):
        update_calls.append(reward)
        real_update(reward)

    policy.update = spy_update

    replanner = RePlanner(world, policy=policy)
    replanner.assignments = {"job-A": "dev-1", "job-B": "dev-2"}
    theta_before = [row[:] for row in policy.theta]

    # incremental re-plan #1: resource_changed / removed
    result1 = replanner.on_event(
        _resource_event("dev-1", ChangeKind.REMOVED, kind="npu", battery=0.0, load=0.0, connected=False)
    )
    assert result1 is not None  # a real solve happened, not a no-op
    assert len(update_calls) == 1
    assert isinstance(update_calls[0], float)
    assert 0.0 <= update_calls[0] <= 1.0
    assert policy.last_arm_index is not None
    assert replanner.last_reward == update_calls[0]

    # incremental re-plan #2: a different kind of event — demand_changed / changed
    result2 = replanner.on_event(_demand_event("job-B", ChangeKind.CHANGED, priority="urgent"))
    assert result2 is not None
    assert len(update_calls) == 2
    assert isinstance(update_calls[1], float)
    assert 0.0 <= update_calls[1] <= 1.0

    # not just called — actually trained the policy, not a no-op update
    assert policy.theta != theta_before


def test_routine_no_op_event_does_not_call_policy_update():
    """A no-op branch (routine telemetry tick) never selects weights, so
    it must never score/update either — otherwise the outcome would be
    attributed to whatever the *previous* round happened to select."""
    world = WorldState()
    _seed_device(world, "dev-1", "gpu", battery=80.0)

    policy = BanditPolicy(rng=random.Random(2))
    update_calls = []
    policy.update = lambda reward: update_calls.append(reward)

    replanner = RePlanner(world, policy=policy)
    result = replanner.on_event(
        _resource_event("dev-1", ChangeKind.CHANGED, kind="gpu", battery=79.0, load=0.25, connected=True)
    )

    assert result is None
    assert update_calls == []


# -- timing: Section 11's "full re-solve vs incremental re-plan" --


def _build_world(num_devices: int, jobs_per_device: int, rng: random.Random) -> WorldState:
    world = WorldState()
    kinds = ["cpu", "gpu", "npu"]
    for i in range(num_devices):
        world.apply_event(
            _resource_event(
                f"dev-{i:04d}",
                ChangeKind.ADDED,
                kind=kinds[i % len(kinds)],
                battery=rng.uniform(20.0, 100.0),
                load=rng.uniform(0.0, 0.5),
                connected=True,
            )
        )
    for j in range(num_devices * jobs_per_device):
        world.apply_event(
            _demand_event(
                f"job-{j:05d}",
                ChangeKind.ADDED,
                structured=True,
                job_type="inference_batch",
                priority=rng.choice(["normal", "normal", "normal", "urgent"]),
                requires=kinds[j % len(kinds)],
            )
        )
    return world


def _min_time(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def _measure(num_devices: int, repeats: int = 3):
    rng = random.Random(num_devices)
    world = _build_world(num_devices, jobs_per_device=1, rng=rng)
    replanner = RePlanner(world)

    full_time = _min_time(replanner.full_solve, repeats)

    # Pick devices from replanner.assignments.values() — i.e. devices
    # the solver actually used — not by device_id convention (e.g.
    # "dev-0000"). CP-SAT is free to assign a job to *any* capable
    # device with spare capacity, so a fixed device_id isn't guaranteed
    # to hold a job; removing one that doesn't orphans nothing and hits
    # RePlanner's `if not orphaned_ids: return None` fast path near-
    # instantly. That's a real code path (worth having!), but timing
    # *that* isn't a fair "incremental replan" measurement — it silently
    # measures a no-op instead of real reassignment work, which is
    # exactly what made an earlier version of this benchmark look
    # backwards (replan time *dropping* as fleet size grew).
    occupied_devices = sorted(set(replanner.assignments.values()))
    assert len(occupied_devices) >= repeats, "fleet too small/underloaded for this benchmark"

    replan_times = []
    for device_id in occupied_devices[:repeats]:
        start = time.perf_counter()
        replanner.on_event(_resource_event(device_id, ChangeKind.REMOVED, kind="cpu", battery=0.0, load=0.0, connected=False))
        replan_times.append(time.perf_counter() - start)
    replan_time = min(replan_times)

    return full_time, replan_time


def test_incremental_replan_is_faster_than_full_solve_at_two_fleet_sizes():
    full_50, replan_50 = _measure(50)
    full_150, replan_150 = _measure(150)

    assert replan_50 < full_50
    assert replan_150 < full_150

    # the incremental path's advantage should hold up, and not shrink,
    # as the fleet grows — full-solve cost scales with fleet size,
    # incremental cost doesn't (it only ever touches one device's job).
    # floor guards against a replan time near the timer's resolution.
    speedup_50 = full_50 / max(replan_50, 1e-6)
    speedup_150 = full_150 / max(replan_150, 1e-6)
    assert speedup_150 > speedup_50 * 0.7  # slack for timing noise, not a razor-thin bound
