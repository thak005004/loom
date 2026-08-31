"""Task/Demand Stream (Section 2).

Simulates incoming job requests: mostly structured, some deliberately
messy natural language. NL requests are passed through untouched as
`{"structured": False, "text": ...}` — the LLM parser that turns them
into structured fields is Day 2 (Section 3c); this stream's only job is
to produce the raw demand, structured or not.

Job requests are ADDED events. Open jobs can also be CHANGED (priority
shifting mid-flight, Section 4's "a job's priority changes") or REMOVED
(the job completes or gets cancelled) — without a REMOVED case, the
open-job pool would only ever grow, and anything querying "what's
currently open" (the world state, later the scheduler) would keep
assigning against jobs that finished long ago.
"Bursts" (arriving irregularly, sometimes in clusters) has no real-time
transport yet to be bursty *in*, so it's modeled at the call site by
varying batch size per `generate()` call rather than inside the
adapter — genuine timing-based burstiness is a Section 8 concern once
this is wired into asyncio.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterator, List, Optional

from orchestrator.events.types import ChangeKind, Event, EventType
from orchestrator.streams.base import StreamAdapter

JOB_TYPES = ("camera_check", "inference_batch", "firmware_update", "quality_scan", "sensor_diagnostic")
PRIORITIES = ("normal", "normal", "normal", "urgent")
REQUIRES = ("cpu", "gpu", "npu", None)

NL_TEMPLATES = (
    "urgent camera check on line 3, needs GPU",
    "can someone run inference on the batch from this morning, not time sensitive",
    "line 7 sensor's been acting weird, check it out when free",
    "need a firmware update pushed to all NPU units before end of day",
    "quality scan backlog is piling up on the east line, can we get to it soon",
)

_JOB_CLOSE_PROB = 0.15
_PRIORITY_CHANGE_PROB = 0.2
_NL_PROB = 0.25
_CLOSE_REASONS = ("completed", "completed", "completed", "cancelled")


class DemandStreamAdapter(StreamAdapter):
    def __init__(self, name: str = "demand", rng: Optional[random.Random] = None) -> None:
        super().__init__(name)
        self.rng = rng or random.Random()
        self._next_job_num = 0
        self._open_job_ids: List[str] = []

    def _new_job_id(self) -> str:
        job_id = f"job-{self._next_job_num:04d}"
        self._next_job_num += 1
        return job_id

    def next_raw(self) -> Dict[str, Any]:
        if self._open_job_ids:
            r = self.rng.random()
            if r < _JOB_CLOSE_PROB:
                job_id = self.rng.choice(self._open_job_ids)
                self._open_job_ids.remove(job_id)
                return {
                    "job_id": job_id,
                    "reason": self.rng.choice(_CLOSE_REASONS),
                    "change_kind": "removed",
                }
            if r < _JOB_CLOSE_PROB + _PRIORITY_CHANGE_PROB:
                job_id = self.rng.choice(self._open_job_ids)
                return {
                    "job_id": job_id,
                    "priority": self.rng.choice(PRIORITIES),
                    "change_kind": "changed",
                }

        job_id = self._new_job_id()
        self._open_job_ids.append(job_id)

        if self.rng.random() < _NL_PROB:
            return {
                "job_id": job_id,
                "structured": False,
                "text": self.rng.choice(NL_TEMPLATES),
                "change_kind": "added",
            }

        return {
            "job_id": job_id,
            "structured": True,
            "job_type": self.rng.choice(JOB_TYPES),
            "priority": self.rng.choice(PRIORITIES),
            "requires": self.rng.choice(REQUIRES),
            "change_kind": "added",
        }

    def generate(self, n: int) -> Iterator[Dict[str, Any]]:
        for _ in range(n):
            yield self.next_raw()

    def parse(self, raw: Dict[str, Any]) -> Event:
        payload = {k: v for k, v in raw.items() if k != "change_kind"}
        return Event(
            type=EventType.DEMAND_CHANGED,
            change_kind=ChangeKind(raw["change_kind"]),
            source=self.name,
            payload=payload,
        )
