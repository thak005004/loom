"""Maintenance/History Log Stream (Section 2).

Simulates a device's own repair/failure history feeding a predictive
reliability signal: a device that racks up enough recent issues gets a
downgraded "reliability" capability signal, and one that's gone quiet
for a while gets marked nominal again. This is the first real producer
of CAPABILITY_CHANGED — the re-planner has had a handler for it since
Day 1 Step 5 (`RePlanner._replan_reevaluate_device`) without ever having
seen one; wiring this stream up requires zero changes to replanner.py.

Unlike telemetry, this stream doesn't own the canonical record for the
devices it reports on — it only knows about reliability, not
kind/battery/load — so its events are deliberately partial payloads
(`{"device_id": ..., "reliability": ...}`). That's exactly why
WorldState merges CHANGED/CAPABILITY_CHANGED resource events onto the
existing record instead of overwriting it wholesale.

Only actual reliability *transitions* are emitted (nominal->degraded,
degraded->nominal) — not every internal issue-tick — so this stream
doesn't spam the re-planner with re-evaluations for devices whose
eligibility hasn't actually changed, mirroring why telemetry's routine
battery/load ticks don't trigger a re-plan either.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterator, List, Optional

from orchestrator.events.types import ChangeKind, Event, EventType
from orchestrator.streams.base import StreamAdapter

_ISSUE_PROB = 0.3  # chance a tick logs a new issue against the picked device
_DEGRADE_THRESHOLD = 3  # issues before a device's reliability downgrades
_RECOVER_PROB = 0.1  # chance a degraded device is reported repaired on a given tick


class MaintenanceStreamAdapter(StreamAdapter):
    def __init__(
        self,
        name: str = "maintenance",
        device_ids: Optional[List[str]] = None,
        num_devices: int = 20,
        rng: Optional[random.Random] = None,
    ) -> None:
        super().__init__(name)
        self.rng = rng or random.Random()
        # Standalone-demoable with a synthetic pool by default, but
        # accepts the real fleet's device_ids so its CAPABILITY_CHANGED
        # events land on devices WorldState actually knows about.
        self.device_ids: List[str] = list(device_ids) if device_ids is not None else [f"dev-{i:04d}" for i in range(num_devices)]
        self._issue_counts: Dict[str, int] = {d: 0 for d in self.device_ids}
        self._degraded: Dict[str, bool] = {d: False for d in self.device_ids}

    def next_raw(self) -> Dict[str, Any]:
        while True:
            device_id = self.rng.choice(self.device_ids)

            if self._degraded[device_id]:
                if self.rng.random() < _RECOVER_PROB:
                    self._issue_counts[device_id] = 0
                    self._degraded[device_id] = False
                    return {
                        "device_id": device_id,
                        "reliability": "nominal",
                        "recent_failure_count": 0,
                        "change_kind": "capability_changed",
                    }
                continue  # still degraded, nothing new to report this tick

            if self.rng.random() < _ISSUE_PROB:
                self._issue_counts[device_id] += 1
                if self._issue_counts[device_id] >= _DEGRADE_THRESHOLD:
                    self._degraded[device_id] = True
                    return {
                        "device_id": device_id,
                        "reliability": "degraded",
                        "recent_failure_count": self._issue_counts[device_id],
                        "change_kind": "capability_changed",
                    }
            # no threshold crossed this tick — loop and try again

    def generate(self, n: int) -> Iterator[Dict[str, Any]]:
        for _ in range(n):
            yield self.next_raw()

    def parse(self, raw: Dict[str, Any]) -> Event:
        payload = {k: v for k, v in raw.items() if k != "change_kind"}
        return Event(
            type=EventType.RESOURCE_CHANGED,
            change_kind=ChangeKind(raw["change_kind"]),
            source=self.name,
            payload=payload,
        )
