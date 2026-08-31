"""Fairness metric (Section 11): "workload distribution across devices,
before vs. after the policy adapts", reported as its own explicit
number.

This is deliberately a standalone function, not just read off
reward.py's internal balance term — that term is intentionally
down-weighted there (see reward.py's own docstring: priority_served has
to dominate it, or leaving everything unassigned scores deceptively
well). Section 11 asks for fairness reported on its own terms, so it
gets its own function rather than being inferred from a weighted
composite.

Lower stdev = work spread more evenly across the fleet.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from orchestrator.scheduling.solver import DEFAULT_CAPACITY_BY_KIND


def load_distribution_stdev(
    devices: List[Dict[str, Any]],
    assignments: Dict[str, str],
    *,
    capacity_by_kind: Dict[str, int] = DEFAULT_CAPACITY_BY_KIND,
) -> float:
    connected = [d for d in devices if d.get("connected", True)]
    if not connected:
        return 0.0

    counts = Counter(assignments.values())
    utilization = []
    for device in connected:
        capacity = device.get("capacity", capacity_by_kind.get(device.get("kind"), 1))
        utilization.append(counts.get(device["device_id"], 0) / max(1, capacity))

    mean = sum(utilization) / len(utilization)
    variance = sum((u - mean) ** 2 for u in utilization) / len(utilization)
    return variance**0.5
