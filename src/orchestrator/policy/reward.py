"""The reward signal the bandit policy learns from (Section 3a).

Kept deliberately simple over clever: three signals, each easy to
compute and reason about on its own, averaged with no fine-tuned
weighting between them.

  - capacity_ok: did any device end up over its capacity? Should never
    happen given solve()'s hard capacity constraint, but this function
    is a standalone check over a (result, jobs, devices) triple, not
    something wired only to solve()'s own output — the re-planner will
    eventually score its own incremental results the same way, so it
    shouldn't silently trust that the constraint held elsewhere.
  - balance: how evenly is the resulting work spread across devices,
    relative to each device's own capacity.
  - priority_served: what fraction of urgent jobs actually got a device.

reward is a weighted average of the three, each already scaled to
roughly [0, 1]. priority_served carries more than a plain 1/3 share
deliberately: capacity_ok and balance are both trivially "perfect" when
nothing gets assigned at all (no overshoot, nothing to balance), so an
equal-weighted average would score total inaction as decent. Weighting
priority_served over 0.5 guarantees that failing to serve urgent demand
pulls the reward below the midpoint no matter how tidy the (empty)
result otherwise looks.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from orchestrator.scheduling.solver import DEFAULT_CAPACITY_BY_KIND, ScheduleResult

_CAPACITY_OK_WEIGHT = 0.2
_BALANCE_WEIGHT = 0.2
_PRIORITY_SERVED_WEIGHT = 0.6


def compute_reward(
    result: ScheduleResult,
    jobs: List[Dict[str, Any]],
    devices: List[Dict[str, Any]],
    *,
    capacity_by_kind: Dict[str, int] = DEFAULT_CAPACITY_BY_KIND,
) -> float:
    device_by_id = {d["device_id"]: d for d in devices}
    counts = Counter(a.device_id for a in result.assignments)

    def capacity_of(device_id: str) -> int:
        device = device_by_id.get(device_id, {})
        return device.get("capacity", capacity_by_kind.get(device.get("kind"), 1))

    overshoot = sum(max(0, count - capacity_of(device_id)) for device_id, count in counts.items())
    capacity_ok = 1.0 if overshoot == 0 else max(0.0, 1.0 - 0.5 * overshoot)

    connected = [d for d in devices if d.get("connected", True)]
    if connected:
        utilization = [counts.get(d["device_id"], 0) / max(1, capacity_of(d["device_id"])) for d in connected]
        mean = sum(utilization) / len(utilization)
        variance = sum((u - mean) ** 2 for u in utilization) / len(utilization)
        stdev = variance**0.5
        balance = max(0.0, 1.0 - min(1.0, stdev * 2))
    else:
        balance = 1.0  # nothing to balance across

    urgent_jobs = [j for j in jobs if j.get("priority") == "urgent"]
    if urgent_jobs:
        assigned_ids = {a.job_id for a in result.assignments}
        priority_served = sum(1 for j in urgent_jobs if j["job_id"] in assigned_ids) / len(urgent_jobs)
    else:
        priority_served = 1.0  # no urgent demand to fail

    return (
        _CAPACITY_OK_WEIGHT * capacity_ok
        + _BALANCE_WEIGHT * balance
        + _PRIORITY_SERVED_WEIGHT * priority_served
    )
