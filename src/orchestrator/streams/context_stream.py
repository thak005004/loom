"""External Context Stream (Section 2).

Lower-frequency than telemetry or demand: shift-schedule signals,
peak-hours flags, and policy updates that change how jobs get
prioritized, not what jobs exist or what devices are available. Always
RULE_CHANGED — a rule doesn't have a meaningful "removed" or
"capability_changed" state in this design, so `change_kind` is always
CHANGED here (toggling `active` True/False covers a rule being turned
on or off).
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterator, Optional

from orchestrator.events.types import ChangeKind, Event, EventType
from orchestrator.streams.base import StreamAdapter

RULES = (
    {"rule": "peak_hours", "description": "prioritize urgent jobs during peak hours"},
    {"rule": "night_shift", "description": "night-shift devices deprioritized for new jobs"},
    {"rule": "quality_scan_requires_gpu", "description": "quality_scan jobs now require a GPU-class device"},
    {"rule": "battery_conservation", "description": "avoid assigning jobs to devices under 20% battery"},
)


class ContextStreamAdapter(StreamAdapter):
    def __init__(self, name: str = "context", rng: Optional[random.Random] = None) -> None:
        super().__init__(name)
        self.rng = rng or random.Random()

    def next_raw(self) -> Dict[str, Any]:
        rule = self.rng.choice(RULES)
        return {
            "rule": rule["rule"],
            "description": rule["description"],
            "active": self.rng.choice([True, False]),
            "change_kind": "changed",
        }

    def generate(self, n: int) -> Iterator[Dict[str, Any]]:
        for _ in range(n):
            yield self.next_raw()

    def parse(self, raw: Dict[str, Any]) -> Event:
        payload = {k: v for k, v in raw.items() if k != "change_kind"}
        return Event(
            type=EventType.RULE_CHANGED,
            change_kind=ChangeKind(raw["change_kind"]),
            source=self.name,
            payload=payload,
        )
