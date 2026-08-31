"""Manual Override Stream (Section 2).

Stands in for a human operator stepping in: pulling a device for
inspection, or forcing a job's priority up to bump it to the front of
the queue. The point of this stream, per the plan, is that it *doesn't*
need a special "human override" code path anywhere downstream — it
just emits the exact same event shapes the automated streams already
produce (RESOURCE_CHANGED/REMOVED, DEMAND_CHANGED/CHANGED), so
WorldState and the re-planner handle an operator's action identically
to a device dying on its own or a job's priority drifting. The only
difference visible anywhere is `Event.source == "override"` instead of
"telemetry" or "demand" — informational only, never branched on.

This is also the first adapter whose events span two different
EventTypes from the same stream, which the registry/bus were always
built to allow (Section 2 never assumed one stream -> one event type):
`parse()` just reads which kind this particular raw record is.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterator, List, Optional

from orchestrator.events.types import ChangeKind, Event, EventType
from orchestrator.streams.base import StreamAdapter

_PULL_DEVICE_PROB = 0.5  # vs. forcing a job's priority up


class OverrideStreamAdapter(StreamAdapter):
    def __init__(
        self,
        name: str = "override",
        device_ids: Optional[List[str]] = None,
        job_ids: Optional[List[str]] = None,
        num_devices: int = 10,
        num_jobs: int = 10,
        rng: Optional[random.Random] = None,
    ) -> None:
        super().__init__(name)
        self.rng = rng or random.Random()
        self.device_ids: List[str] = list(device_ids) if device_ids is not None else [f"dev-{i:04d}" for i in range(num_devices)]
        self.job_ids: List[str] = list(job_ids) if job_ids is not None else [f"job-{i:04d}" for i in range(num_jobs)]

    def next_raw(self) -> Dict[str, Any]:
        if self.job_ids and (not self.device_ids or self.rng.random() >= _PULL_DEVICE_PROB):
            job_id = self.rng.choice(self.job_ids)
            return {
                "_event_type": "demand",
                "job_id": job_id,
                "priority": "urgent",
                "reason": "operator_override",
                "change_kind": "changed",
            }

        device_id = self.rng.choice(self.device_ids)
        return {
            "_event_type": "resource",
            "device_id": device_id,
            "reason": "pulled_for_inspection",
            "change_kind": "removed",
        }

    def generate(self, n: int) -> Iterator[Dict[str, Any]]:
        for _ in range(n):
            yield self.next_raw()

    def parse(self, raw: Dict[str, Any]) -> Event:
        event_type = EventType.RESOURCE_CHANGED if raw["_event_type"] == "resource" else EventType.DEMAND_CHANGED
        payload = {k: v for k, v in raw.items() if k not in ("change_kind", "_event_type")}
        return Event(
            type=event_type,
            change_kind=ChangeKind(raw["change_kind"]),
            source=self.name,
            payload=payload,
        )
