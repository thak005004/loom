"""Feasibility layer: a pure, read-only query over WorldState answering
"what categories of work could we take on right now" — capability
supply, not committed demand. Unlike solve()/RePlanner, this never
matches jobs to specific devices and never mutates anything; it just
asks whether the fleet, as it exists this instant, has at least one
device capable of a category at all.

A "category" here is a job's `requires` value — the literal thing
solver.py's `_is_capable()` gates capability on (`device.kind ==
requires`, or `requires is None` meaning "any device"). `job_type`
isn't a capability gate anywhere in this codebase (only the
quality_scan_requires_gpu rule maps a job_type to a requires value, and
only once that rule has already fired), so `requires` is the only
category definition that actually matches what the scheduler does.

Eligibility here deliberately mirrors solver.py's own rules exactly —
connected, and capacity > 0 — rather than inventing a separate
battery-based cutoff: battery affects scoring everywhere else in this
codebase, never eligibility, so feasibility would misrepresent the real
system if it quietly added a hard cutoff nothing else enforces.

Nothing here is hardcoded to cpu/gpu/npu: the category universe is
derived entirely from whatever `requires` values and device `kind`s
actually appear in the live WorldState, so a brand-new category (see
scheduling/whatif.py's sibling test, test_pluggable_categories.py)
is reported correctly with zero changes to this file.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

from orchestrator.fleet.state import WorldState
from orchestrator.scheduling.solver import DEFAULT_CAPACITY_BY_KIND


def _has_spare_capacity(device: Dict[str, Any]) -> bool:
    if not device.get("connected", True):
        return False
    capacity = device.get("capacity", DEFAULT_CAPACITY_BY_KIND.get(device.get("kind"), 1))
    return capacity > 0


def feasible_categories(world: WorldState) -> Dict[Optional[str], bool]:
    """Maps every `requires` category currently in play — from open
    demand (a job asking for it) or from the fleet's own device kinds
    (a device offering it, even before any job asks) — to whether at
    least one connected device with spare capacity could take it on
    right now. `None` ("any device", unconstrained work) is always
    included and is feasible whenever any device at all has capacity.

    Pure and read-only: only reads `world.devices` / `world.jobs`,
    never calls solve()/RePlanner, never mutates WorldState.
    """
    supply_kinds: Set[Optional[str]] = {
        device.get("kind") for device in world.devices.values() if _has_spare_capacity(device)
    }
    any_capacity = bool(supply_kinds)

    categories: Set[Optional[str]] = {None}
    categories.update(job.get("requires") for job in world.jobs.values())
    categories.update(device.get("kind") for device in world.devices.values())

    return {category: (any_capacity if category is None else category in supply_kinds) for category in categories}
