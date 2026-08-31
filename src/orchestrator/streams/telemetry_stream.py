"""Device/Resource Telemetry Stream (Section 2).

Simulates a fleet of 50-150 heterogeneous devices (CPU/GPU/NPU) with
battery decay, occasional connectivity drops, and load drift. Most
ticks are routine updates (CHANGED); a device whose battery reaches
zero is REMOVED from the fleet; new devices occasionally join (ADDED).
Capability changes (e.g. a device's reliability degrading) are left to
the maintenance/history-log stream in Section 2 — that's the source the
plan assigns them to, and it keeps this adapter's job narrow: telemetry
reports state, not history-derived judgments about a device.

This is a simulator standing in for a real telemetry source, so it
generates its own raw records via `next_raw()` in addition to the
required `parse()`. That generation method is a convention shared by
the other simulated streams (demand, context) — not part of the
StreamAdapter contract itself, since a real production adapter (e.g.
one reading a Kafka topic) wouldn't generate anything, only parse what
arrives.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterator, Optional

from orchestrator.events.types import ChangeKind, Event, EventType
from orchestrator.streams.base import StreamAdapter

DEVICE_KINDS = ("cpu", "gpu", "npu")

# Tuning constants for the simulation.
_BATTERY_DRAIN_RANGE = (0.05, 1.5)  # per-tick battery drop, percentage points
_LOAD_DRIFT_RANGE = (-0.1, 0.1)
_DISCONNECT_PROB = 0.03
_RECONNECT_PROB = 0.5
_NEW_DEVICE_PROB = 0.02


class TelemetryStreamAdapter(StreamAdapter):
    def __init__(
        self,
        name: str = "telemetry",
        num_devices: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        super().__init__(name)
        self.rng = rng or random.Random()
        n = num_devices if num_devices is not None else self.rng.randint(50, 150)
        self.devices: Dict[str, Dict[str, Any]] = {}
        self._next_device_num = 0
        for _ in range(n):
            device = self._make_device(battery_range=(40.0, 100.0))
            self.devices[device["device_id"]] = device

    def _make_device(self, battery_range=(60.0, 100.0)) -> Dict[str, Any]:
        device_id = f"dev-{self._next_device_num:03d}"
        self._next_device_num += 1
        return {
            "device_id": device_id,
            "kind": self.rng.choice(DEVICE_KINDS),
            "battery": self.rng.uniform(*battery_range),
            "connected": True,
            "load": self.rng.uniform(0.0, 0.3),
        }

    def next_raw(self) -> Dict[str, Any]:
        """Advance the simulation by one tick and return the raw record."""
        if not self.devices or self.rng.random() < _NEW_DEVICE_PROB:
            device = self._make_device()
            self.devices[device["device_id"]] = device
            return {**device, "change_kind": "added"}

        device_id = self.rng.choice(list(self.devices))
        device = self.devices[device_id]
        device["battery"] = max(0.0, device["battery"] - self.rng.uniform(*_BATTERY_DRAIN_RANGE))
        device["load"] = min(1.0, max(0.0, device["load"] + self.rng.uniform(*_LOAD_DRIFT_RANGE)))
        if device["connected"] and self.rng.random() < _DISCONNECT_PROB:
            device["connected"] = False
        elif not device["connected"] and self.rng.random() < _RECONNECT_PROB:
            device["connected"] = True

        if device["battery"] <= 0.0:
            del self.devices[device_id]
            return {**device, "battery": 0.0, "connected": False, "change_kind": "removed"}
        return {**device, "change_kind": "changed"}

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
