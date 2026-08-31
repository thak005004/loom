"""CP-SAT scheduler (Section 6/9): assigns open jobs to devices.

Takes a snapshot of open jobs and available devices (from WorldState)
and produces job -> device assignments. Two hard constraints, enforced
structurally rather than checked after the fact:

  - capability: a decision variable only exists for a (job, device)
    pair the device is actually capable of, so an incapable assignment
    is never even representable, let alone chosen.
  - capacity: each device's assigned-job count is bounded by its
    capacity in the model itself.

Assignment is optional per job (each job gets an explicit "unassigned"
boolean rather than a hard `== 1` requirement). That's what makes an
overloaded scenario (more eligible jobs than capacity) degrade
sensibly: the model is never infeasible, it just leaves the
lowest-value jobs unassigned rather than raising.

Four scalar weights drive the objective — priority_weight,
load_penalty_scale, battery_bonus_scale, urgent_unassigned_penalty.
These are the exact knobs Day 1 Step 4's bandit policy learns to set
per fleet context; the defaults here are just the "reasonable starting
point" the plan calls for at this step. `priority_score` used to be a
raw {"urgent": ..., "normal": ...} dict, but a bandit needs to output
plain scalars it can nudge up or down — so it's now a single
`priority_weight` multiplier over a fixed base spread instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ortools.sat.python import cp_model

from orchestrator.fleet.state import WorldState

# Jobs land here without a `requires` key only when they're raw,
# unparsed natural-language requests (Section 3c is Day 2) — the
# solver has no way to know their capability needs yet, so it leaves
# them out rather than guessing.
DEFAULT_CAPACITY_BY_KIND: Dict[str, int] = {"cpu": 3, "gpu": 2, "npu": 2}

# Fixed base spread between priority tiers; priority_weight scales it.
_PRIORITY_BASE: Dict[str, int] = {"urgent": 1000, "normal": 300}

DEFAULT_PRIORITY_WEIGHT = 1.0
DEFAULT_LOAD_PENALTY_SCALE = 2.0
DEFAULT_BATTERY_BONUS_SCALE = 1.0
DEFAULT_URGENT_UNASSIGNED_PENALTY = 500.0


@dataclass(frozen=True)
class Assignment:
    job_id: str
    device_id: str


@dataclass(frozen=True)
class ScheduleResult:
    status: str  # cp_model.CpSolver.StatusName(...): "OPTIMAL", "FEASIBLE", ...
    assignments: List[Assignment] = field(default_factory=list)
    unassigned_job_ids: List[str] = field(default_factory=list)


def _is_schedulable(job: Dict[str, Any]) -> bool:
    return "requires" in job and "priority" in job


def _is_capable(job: Dict[str, Any], device: Dict[str, Any]) -> bool:
    requires = job.get("requires")
    return requires is None or device.get("kind") == requires


def _pair_score(
    job: Dict[str, Any],
    device: Dict[str, Any],
    priority_weight: float,
    load_penalty_scale: float,
    battery_bonus_scale: float,
) -> int:
    priority = _PRIORITY_BASE.get(job.get("priority", "normal"), _PRIORITY_BASE["normal"]) * priority_weight
    load_pct = round(min(1.0, max(0.0, device.get("load", 0.0))) * 100)
    battery_pct = round(min(100.0, max(0.0, device.get("battery", 0.0))))
    # CP-SAT objective coefficients must be integers.
    return round(priority - load_penalty_scale * load_pct + battery_bonus_scale * battery_pct)


def solve(
    jobs: List[Dict[str, Any]],
    devices: List[Dict[str, Any]],
    *,
    capacity_by_kind: Dict[str, int] = DEFAULT_CAPACITY_BY_KIND,
    priority_weight: float = DEFAULT_PRIORITY_WEIGHT,
    load_penalty_scale: float = DEFAULT_LOAD_PENALTY_SCALE,
    battery_bonus_scale: float = DEFAULT_BATTERY_BONUS_SCALE,
    urgent_unassigned_penalty: float = DEFAULT_URGENT_UNASSIGNED_PENALTY,
) -> ScheduleResult:
    schedulable_jobs = [j for j in jobs if _is_schedulable(j)]
    unschedulable_job_ids = [j["job_id"] for j in jobs if not _is_schedulable(j)]
    candidate_devices = [d for d in devices if d.get("connected", True)]

    model = cp_model.CpModel()
    x: Dict[Tuple[str, str], Any] = {}
    score: Dict[Tuple[str, str], int] = {}

    for job in schedulable_jobs:
        for device in candidate_devices:
            if not _is_capable(job, device):
                continue
            key = (job["job_id"], device["device_id"])
            x[key] = model.NewBoolVar(f"x_{job['job_id']}_{device['device_id']}")
            score[key] = _pair_score(job, device, priority_weight, load_penalty_scale, battery_bonus_scale)

    # Each job gets an explicit "unassigned" indicator instead of a bare
    # `sum(job_vars) <= 1`: it's what lets the objective directly
    # penalize leaving an *urgent* job unassigned (below), as its own
    # tunable knob distinct from priority_weight's in-batch tie-breaking.
    unassigned = {}
    for job in schedulable_jobs:
        job_id = job["job_id"]
        job_vars = [var for (jid, _), var in x.items() if jid == job_id]
        u = model.NewBoolVar(f"unassigned_{job_id}")
        if job_vars:
            model.Add(sum(job_vars) + u == 1)
        else:
            model.Add(u == 1)  # no capable device at all
        unassigned[job_id] = u

    for device in candidate_devices:
        capacity = device.get("capacity", capacity_by_kind.get(device.get("kind"), 1))
        device_vars = [var for (_, device_id), var in x.items() if device_id == device["device_id"]]
        if device_vars:
            model.Add(sum(device_vars) <= capacity)

    objective_terms = [var * score[key] for key, var in x.items()]
    penalty = round(urgent_unassigned_penalty)
    for job in schedulable_jobs:
        if job.get("priority") == "urgent":
            objective_terms.append(-penalty * unassigned[job["job_id"]])
    model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    assignments: List[Assignment] = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for (job_id, device_id), var in x.items():
            if solver.Value(var) == 1:
                assignments.append(Assignment(job_id=job_id, device_id=device_id))

    assigned_job_ids = {a.job_id for a in assignments}
    unassigned_job_ids = [
        j["job_id"] for j in schedulable_jobs if j["job_id"] not in assigned_job_ids
    ] + unschedulable_job_ids

    return ScheduleResult(status=status_name, assignments=assignments, unassigned_job_ids=unassigned_job_ids)


def solve_world(world: WorldState, **kwargs: Any) -> ScheduleResult:
    """Convenience wrapper matching the plan's framing ("takes a
    WorldState snapshot") — `solve()` itself stays a pure function over
    plain lists so it's cheap to unit test without standing up a bus."""
    return solve(world.open_jobs(), world.available_devices(), **kwargs)
