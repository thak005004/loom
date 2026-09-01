"""What-if simulation: answers "what would happen if device X failed
right now?" without touching the live schedule, world state, or bandit.

Reuses RePlanner._replan_removed_device verbatim on a deep-copied
WorldState and a throwaway RePlanner instance seeded with a copy of the
live assignment map — the exact same candidate-selection and solve()
logic the real incremental removal path uses, not a separate
reimplementation that could quietly drift from it. The scratch
RePlanner is constructed with a plain `weights` dict (the live policy's
*currently active* arm, snapshotted by the caller) rather than a
`policy` reference, so the simulation can never call `policy.update()`
and never trains the bandit on a hypothetical outcome.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from orchestrator.fleet.state import WorldState
from orchestrator.scheduling.replanner import RePlanner


@dataclass(frozen=True)
class WhatIfResult:
    device_id: str
    device_existed: bool
    orphaned_job_ids: List[str] = field(default_factory=list)
    # job_id -> new device_id, or None if the job would end up unassigned
    moves: Dict[str, Optional[str]] = field(default_factory=dict)


def simulate_device_failure(
    world: WorldState,
    assignments: Dict[str, str],
    device_id: str,
    weights: Dict[str, Any],
) -> WhatIfResult:
    """Pure/read-only: does not mutate `world` or `assignments`, and
    never touches a bandit policy."""
    orphaned_job_ids = [jid for jid, did in assignments.items() if did == device_id]
    if not orphaned_job_ids:
        return WhatIfResult(
            device_id=device_id,
            device_existed=device_id in world.devices,
            orphaned_job_ids=[],
            moves={},
        )

    scratch_world = WorldState()
    scratch_world.devices = copy.deepcopy(world.devices)
    scratch_world.jobs = copy.deepcopy(world.jobs)
    scratch_world.devices.pop(device_id, None)  # simulate the failure

    scratch = RePlanner(scratch_world, weights=dict(weights))
    scratch.assignments = dict(assignments)
    scratch._replan_removed_device(device_id)  # same logic a real REMOVED event triggers

    moves = {jid: scratch.assignments.get(jid) for jid in orphaned_job_ids}
    return WhatIfResult(
        device_id=device_id,
        device_existed=True,
        orphaned_job_ids=orphaned_job_ids,
        moves=moves,
    )
