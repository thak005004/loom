"""WorldState: the live picture the scheduler queries and the
re-planner will later diff against.

The event bus and the streams behind it only carry a log of *changes*.
Nothing before this point maintains "what's true right now" — a
consumer that wants the current set of devices or open jobs would
otherwise have to replay the whole event history itself. WorldState
subscribes to the bus once and folds every event into two dicts
(`devices`, `jobs`), updated incrementally as events arrive, so the
solver (and later the re-planner) can just read `world.devices` /
`world.jobs` instead of touching the event stream at all.

Deliberately dumb: it mirrors what the streams report without applying
policy (e.g. it doesn't drop disconnected devices or filter unparsed
jobs) — that judgment belongs to whoever queries the state, not to the
thing maintaining it.
"""

from __future__ import annotations

from typing import Any, Dict, List

from orchestrator.events.bus import EventBus
from orchestrator.events.types import ChangeKind, Event, EventType


class WorldState:
    def __init__(self) -> None:
        self.devices: Dict[str, Dict[str, Any]] = {}
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def attach(self, bus: EventBus) -> None:
        """Subscribe to `bus` so this instance stays live as events
        arrive — including a bus already in use by other subscribers,
        and adapters registered before or after this call."""
        bus.subscribe(self.apply_event)

    def apply_event(self, event: Event) -> None:
        if event.type == EventType.RESOURCE_CHANGED:
            self._apply_resource_event(event)
        elif event.type == EventType.DEMAND_CHANGED:
            self._apply_demand_event(event)
        elif event.type == EventType.RULE_CHANGED:
            pass  # rules affect scoring, not "what exists" — not this component's concern

    def _apply_resource_event(self, event: Event) -> None:
        device_id = event.payload["device_id"]
        if event.change_kind == ChangeKind.REMOVED:
            self.devices.pop(device_id, None)
        elif event.change_kind == ChangeKind.ADDED:
            self.devices[device_id] = dict(event.payload)
        else:
            # CHANGED and CAPABILITY_CHANGED events aren't guaranteed to
            # carry the device's full state — telemetry's CHANGED ticks
            # happen to (so this is a no-op replace for them either
            # way), but a source that only knows about one signal (e.g.
            # the maintenance stream reporting just a reliability
            # change) must not be allowed to wipe out the device's
            # kind/battery/load by overwriting the whole record.
            existing = self.devices.get(device_id, {})
            self.devices[device_id] = {**existing, **event.payload}

    def _apply_demand_event(self, event: Event) -> None:
        job_id = event.payload["job_id"]
        if event.change_kind == ChangeKind.REMOVED:
            self.jobs.pop(job_id, None)
        elif event.change_kind == ChangeKind.ADDED:
            self.jobs[job_id] = dict(event.payload)
        else:
            # CHANGED events (e.g. a priority update) carry only the
            # fields that changed, so merge onto the existing record
            # instead of overwriting it.
            existing = self.jobs.get(job_id, {})
            self.jobs[job_id] = {**existing, **event.payload}

    def available_devices(self) -> List[Dict[str, Any]]:
        return list(self.devices.values())

    def open_jobs(self) -> List[Dict[str, Any]]:
        return list(self.jobs.values())
