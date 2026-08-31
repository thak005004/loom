"""Section 11 throughput benchmark: events/sec at a couple of dataset
sizes, processing the historical log incrementally through
WorldState.apply_event() — reading the JSONL file line by line, never
holding the whole dataset in memory, consistent with Section 8's
"process as a stream, don't load a giant list upfront."

Run with: python scripts/benchmark_throughput.py
(regenerate the log first if data/history_log.jsonl doesn't exist yet:
python -m orchestrator.streams.history_log --n 50000)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from orchestrator.events.types import ChangeKind, Event, EventType
from orchestrator.fleet.state import WorldState

LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "history_log.jsonl"
SCALES = (1_000, 50_000)


def _event_from_record(record: dict) -> Event:
    return Event(
        type=EventType(record["type"]),
        change_kind=ChangeKind(record["change_kind"]),
        source=record["source"],
        payload=record["payload"],
        timestamp=record["timestamp"],
    )


def benchmark(path: Path, n: int) -> float:
    """Returns events/sec processing the first n events of the log
    through a fresh WorldState. Timing covers JSON parsing + Event
    construction + apply_event() — the real per-event cost, not just
    the file read."""
    world = WorldState()
    count = 0
    start = time.perf_counter()
    with path.open() as f:
        for line in f:
            if count >= n:
                break
            world.apply_event(_event_from_record(json.loads(line)))
            count += 1
    elapsed = time.perf_counter() - start
    return count / elapsed if elapsed > 0 else float("inf")


if __name__ == "__main__":
    if not LOG_PATH.exists():
        raise SystemExit(f"{LOG_PATH} not found — generate it first: python -m orchestrator.streams.history_log --n 50000")

    total_lines = sum(1 for _ in LOG_PATH.open())
    print(f"Benchmarking against {LOG_PATH} ({total_lines} events on disk)\n")

    for n in SCALES:
        if n > total_lines:
            print(f"{n:>7} events: skipped (log only has {total_lines})")
            continue
        rate = benchmark(LOG_PATH, n)
        print(f"{n:>7} events: {rate:>10,.0f} events/sec  ({n / rate * 1000:.1f} ms total)")
