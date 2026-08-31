"""Incremental re-planner (Section 3b/4): reacts to any event from any
stream by re-solving only the minimal affected slice, then merges the
result into the current overall assignment — never a full solve() over
the whole WorldState.

One dispatcher, keyed only on (event.type, event.change_kind) — no
per-scenario handlers, no per-stream handlers, matching Section 4's
point that a sixth stream nobody designed for today still works,
because the re-planner was never written against specific sources:

  - RESOURCE_CHANGED / ADDED   -> reconsider this device against
    currently-*unassigned* jobs only (Section 4: "considers it for any
    pending or future jobs immediately"). Already-assigned work is
    outside the slice by construction, so it's structurally impossible
    for a new device to disturb it.
  - RESOURCE_CHANGED / REMOVED -> reassign only the jobs that were on
    this device (the "affected slice" is exactly the plan's own
    phrasing: "the changed device plus any jobs currently assigned to
    it").
  - RESOURCE_CHANGED / CHANGED -> a routine telemetry tick (battery
    ticking down, load drifting). WorldState already reflects it —
    nothing about the current plan is actually invalidated by a normal
    reading, so this is a deliberate no-op rather than a re-plan.
  - RESOURCE_CHANGED / CAPABILITY_CHANGED -> treated like a from-
    scratch re-evaluation of that one device: its current jobs might no
    longer fit, and it might now newly fit some pending ones. (Written
    before the maintenance-log stream existed to produce this — the
    dispatcher needed zero changes once it did.)
  - DEMAND_CHANGED / ADDED, CHANGED -> just that one job, against
    devices with spare capacity of the right kind (the plan's own
    phrasing again).
  - DEMAND_CHANGED / REMOVED -> the job closed or was cancelled; drop
    it from the current assignment. No solver call at all — there's
    nothing to decide.
  - RULE_CHANGED -> only the jobs the rule actually concerns (via
    RULE_EFFECTS below), re-solved against their own spare-capacity
    devices. A rule with no operational mapping yet is a no-op, not a
    guess.

Ordering note: RePlanner.on_event() applies the event to its own
`world` reference itself before reacting (`world.apply_event` is
idempotent — a repeated ADDED/CHANGED is just the same overwrite twice,
a repeated REMOVED is just a second no-op pop), so it's safe to attach
RePlanner and WorldState to the same bus in either order. It doesn't
need to be the *same* WorldState instance a caller happens to be
reading elsewhere, but in practice it is.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set

from orchestrator.events.bus import EventBus
from orchestrator.events.types import ChangeKind, Event, EventType
from orchestrator.fleet.state import WorldState
from orchestrator.scheduling.solver import DEFAULT_CAPACITY_BY_KIND, ScheduleResult, solve, solve_world

# Rule name -> the job_type it constrains, and (optionally) the
# capability it imposes on matching jobs going forward. Section 4's
# worked example is "this job type now requires two devices" — but the
# solver's model is job -> *one* device; there's no notion of a job
# consuming two devices at once. A rule whose "effect" was just a
# re-solve under the exact same constraints would trigger a real re-plan
# that changes nothing, which is a gap the moment anyone checks the
# resulting assignment rather than just the fact that a re-solve fired.
# So this rule is narrowed to a capability constraint the existing
# single-device model actually enforces: quality_scan jobs now require
# a GPU-class device. `requires` is set directly on the affected job
# records before re-solving, so the existing capability-matching logic
# in solve() picks it up with zero solver changes — a job that wasn't
# GPU-bound before may now move or become unassigned, which is a real,
# checkable effect.
#
# Rules with no entry here (peak_hours, night_shift,
# battery_conservation) don't yet have an operational mapping to a
# specific job slice — a fuller build would feed them into the bandit's
# context instead of filtering jobs directly, which is beyond this
# step's scope.
RULE_EFFECTS: Dict[str, Dict[str, Optional[str]]] = {
    "quality_scan_requires_gpu": {"job_type": "quality_scan", "requires": "gpu"},
}


class RePlanner:
    def __init__(self, world: WorldState, weights: Optional[Dict[str, Any]] = None) -> None:
        self.world = world
        self.weights: Dict[str, Any] = dict(weights or {})
        self.assignments: Dict[str, str] = {}

        self.last_full_solve_seconds: Optional[float] = None
        self.last_replan_seconds: Optional[float] = None

    def attach(self, bus: EventBus) -> None:
        bus.subscribe(self.on_event)

    # -- full solve, for the Section 11 timing comparison and initial seeding --

    def full_solve(self) -> ScheduleResult:
        start = time.perf_counter()
        result = solve_world(self.world, **self.weights)
        self.last_full_solve_seconds = time.perf_counter() - start
        self.assignments = {a.job_id: a.device_id for a in result.assignments}
        return result

    # -- incremental path --

    def on_event(self, event: Event) -> Optional[ScheduleResult]:
        start = time.perf_counter()
        self.world.apply_event(event)  # idempotent; safe even if world already saw this event
        result = self._dispatch(event)
        self.last_replan_seconds = time.perf_counter() - start
        return result

    def _dispatch(self, event: Event) -> Optional[ScheduleResult]:
        if event.type == EventType.RESOURCE_CHANGED:
            return self._replan_resource(event)
        if event.type == EventType.DEMAND_CHANGED:
            return self._replan_demand(event)
        if event.type == EventType.RULE_CHANGED:
            return self._replan_rule(event)
        return None

    def _replan_resource(self, event: Event) -> Optional[ScheduleResult]:
        device_id = event.payload["device_id"]
        if event.change_kind == ChangeKind.ADDED:
            return self._replan_new_device(device_id)
        if event.change_kind == ChangeKind.REMOVED:
            return self._replan_removed_device(device_id)
        if event.change_kind == ChangeKind.CAPABILITY_CHANGED:
            return self._replan_reevaluate_device(device_id)
        return None  # routine CHANGED tick: world already updated, nothing to re-plan

    def _replan_new_device(self, device_id: str) -> Optional[ScheduleResult]:
        device = self.world.devices.get(device_id)
        if device is None:
            return None
        pending_jobs = [j for j in self.world.open_jobs() if j["job_id"] not in self.assignments]
        if not pending_jobs:
            return None
        result = solve(pending_jobs, [device], **self.weights)
        self._merge(result, {j["job_id"] for j in pending_jobs})
        return result

    def _replan_removed_device(self, device_id: str) -> Optional[ScheduleResult]:
        orphaned_ids = [jid for jid, did in self.assignments.items() if did == device_id]
        if not orphaned_ids:
            return None
        slice_jobs = []
        for jid in orphaned_ids:
            job = self.world.jobs.get(jid)
            if job is None:
                self.assignments.pop(jid, None)  # closed in the meantime; nothing to reassign
            else:
                slice_jobs.append(job)
        if not slice_jobs:
            return None
        slice_ids = {j["job_id"] for j in slice_jobs}
        candidates = self._candidate_devices_for(slice_jobs, exclude_job_ids=slice_ids)
        result = solve(slice_jobs, candidates, **self.weights)
        self._merge(result, slice_ids)
        return result

    def _replan_reevaluate_device(self, device_id: str) -> Optional[ScheduleResult]:
        device = self.world.devices.get(device_id)
        current_ids = {jid for jid, did in self.assignments.items() if did == device_id}
        pending_ids = {j["job_id"] for j in self.world.open_jobs() if j["job_id"] not in self.assignments}
        slice_ids = current_ids | pending_ids
        slice_jobs = [self.world.jobs[jid] for jid in slice_ids if jid in self.world.jobs]
        if not slice_jobs:
            return None
        candidates = self._candidate_devices_for(slice_jobs, exclude_job_ids=slice_ids)
        if device is not None and device not in candidates:
            candidates = candidates + [device]
        result = solve(slice_jobs, candidates, **self.weights)
        self._merge(result, {j["job_id"] for j in slice_jobs})
        return result

    def _replan_demand(self, event: Event) -> Optional[ScheduleResult]:
        job_id = event.payload["job_id"]
        if event.change_kind == ChangeKind.REMOVED:
            self.assignments.pop(job_id, None)
            return None
        job = self.world.jobs.get(job_id)
        if job is None:
            self.assignments.pop(job_id, None)
            return None
        candidates = self._candidate_devices_for([job], exclude_job_ids={job_id})
        result = solve([job], candidates, **self.weights)
        self._merge(result, {job_id})
        return result

    def _replan_rule(self, event: Event) -> Optional[ScheduleResult]:
        effect = RULE_EFFECTS.get(event.payload.get("rule"))
        if effect is None:
            return None
        job_type = effect["job_type"]
        affected = [j for j in self.world.open_jobs() if j.get("job_type") == job_type]
        if not affected:
            return None
        new_requires = effect.get("requires")
        if new_requires is not None:
            # mutate the job records WorldState already owns — the
            # rule's constraint is now a real, persistent fact about
            # these jobs, not just a one-off input to this solve() call
            for job in affected:
                job["requires"] = new_requires
        affected_ids = {j["job_id"] for j in affected}
        candidates = self._candidate_devices_for(affected, exclude_job_ids=affected_ids)
        result = solve(affected, candidates, **self.weights)
        self._merge(result, affected_ids)
        return result

    # -- shared helpers --

    def _candidate_devices_for(self, jobs: List[Dict[str, Any]], exclude_job_ids: Set[str]) -> List[Dict[str, Any]]:
        """Devices with spare capacity of the right kind for at least one
        of `jobs` — the plan's own phrasing for the demand_changed slice,
        reused for every other event type's device-side candidates too."""
        needed_kinds = {j.get("requires") for j in jobs}
        unconstrained = None in needed_kinds
        candidates = []
        for device in self.world.available_devices():
            if not device.get("connected", True):
                continue
            if not (unconstrained or device.get("kind") in needed_kinds):
                continue
            capacity = device.get("capacity", DEFAULT_CAPACITY_BY_KIND.get(device.get("kind"), 1))
            used = sum(
                1
                for jid, did in self.assignments.items()
                if did == device["device_id"] and jid not in exclude_job_ids
            )
            if used < capacity:
                candidates.append(device)
        return candidates

    def _merge(self, result: ScheduleResult, slice_job_ids: Set[str]) -> None:
        for job_id in slice_job_ids:
            self.assignments.pop(job_id, None)
        for assignment in result.assignments:
            self.assignments[assignment.job_id] = assignment.device_id
