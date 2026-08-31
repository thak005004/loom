from __future__ import annotations

import random

from orchestrator.scheduling.solver import (
    DEFAULT_CAPACITY_BY_KIND,
    solve,
)


def _device(device_id, kind, battery=80.0, load=0.2, connected=True, **extra):
    return {"device_id": device_id, "kind": kind, "battery": battery, "load": load, "connected": connected, **extra}


def _job(job_id, requires=None, priority="normal", **extra):
    return {"job_id": job_id, "requires": requires, "priority": priority, **extra}


def test_solvable_small_case_produces_valid_assignment():
    devices = [_device("dev-gpu", "gpu"), _device("dev-cpu", "cpu")]
    jobs = [_job("job-1", requires="gpu", priority="urgent"), _job("job-2", requires="cpu")]

    result = solve(jobs, devices)

    assert result.status in ("OPTIMAL", "FEASIBLE")
    assert result.unassigned_job_ids == []
    assignment_by_job = {a.job_id: a.device_id for a in result.assignments}
    assert assignment_by_job == {"job-1": "dev-gpu", "job-2": "dev-cpu"}


def test_overloaded_case_degrades_sensibly_instead_of_crashing():
    # only one NPU device (capacity 2 by default) but five NPU-only jobs
    devices = [_device("dev-npu", "npu"), _device("dev-cpu", "cpu")]
    jobs = [_job(f"job-{i}", requires="npu", priority="urgent") for i in range(5)]

    result = solve(jobs, devices)

    assert result.status in ("OPTIMAL", "FEASIBLE")
    npu_capacity = DEFAULT_CAPACITY_BY_KIND["npu"]
    assert len(result.assignments) == npu_capacity
    assert len(result.unassigned_job_ids) == 5 - npu_capacity
    # every unassigned job is one of the ones we submitted, not something invented
    assert set(result.unassigned_job_ids) <= {j["job_id"] for j in jobs}


def test_unschedulable_unparsed_nl_jobs_are_left_unassigned_not_crashed():
    devices = [_device("dev-gpu", "gpu")]
    jobs = [
        _job("job-structured", requires="gpu"),
        {"job_id": "job-nl", "structured": False, "text": "check line 3 please"},
    ]

    result = solve(jobs, devices)

    assert result.status in ("OPTIMAL", "FEASIBLE")
    assigned = {a.job_id for a in result.assignments}
    assert assigned == {"job-structured"}
    assert result.unassigned_job_ids == ["job-nl"]


def test_no_assignment_ever_violates_capability_or_capacity():
    rng = random.Random(42)
    kinds = ["cpu", "gpu", "npu"]
    devices = [_device(f"dev-{i}", rng.choice(kinds), battery=rng.uniform(0, 100), load=rng.uniform(0, 1)) for i in range(12)]
    jobs = [
        _job(f"job-{i}", requires=rng.choice(kinds + [None]), priority=rng.choice(["normal", "urgent"]))
        for i in range(40)
    ]

    result = solve(jobs, devices)

    device_by_id = {d["device_id"]: d for d in devices}
    job_by_id = {j["job_id"]: j for j in jobs}
    counts: dict[str, int] = {}
    for a in result.assignments:
        device = device_by_id[a.device_id]
        job = job_by_id[a.job_id]
        requires = job.get("requires")
        assert requires is None or device["kind"] == requires
        counts[a.device_id] = counts.get(a.device_id, 0) + 1

    for device_id, count in counts.items():
        kind = device_by_id[device_id]["kind"]
        assert count <= DEFAULT_CAPACITY_BY_KIND[kind]


def test_disconnected_devices_are_never_assigned():
    devices = [_device("dev-gpu-off", "gpu", connected=False), _device("dev-gpu-on", "gpu")]
    jobs = [_job("job-1", requires="gpu")]

    result = solve(jobs, devices)

    assert [a.device_id for a in result.assignments] == ["dev-gpu-on"]
