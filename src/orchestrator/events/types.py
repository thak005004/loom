"""The shared event language every stream adapter emits into.

Everything downstream of the registry (the bus, the re-planner, the
policy) is written only against these three types — never against a
specific stream. That's what makes adding a new stream a non-event.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any


class EventType(str, Enum):
    RESOURCE_CHANGED = "resource_changed"
    DEMAND_CHANGED = "demand_changed"
    RULE_CHANGED = "rule_changed"


class ChangeKind(str, Enum):
    """The sub-kind of change within an EventType (Section 4's worked
    examples table: "resource removed" vs "resource added" vs "resource
    capability changed" are all resource_changed events, distinguished
    only by this field). One shared enum across all three EventTypes,
    not a different set per type — not every combination is meaningful
    (a rule is normally just CHANGED, never CAPABILITY_CHANGED), but
    keeping it a single vocabulary is what lets the re-planner branch on
    (type, change_kind) with one general mechanism instead of a parallel
    per-type enum."""

    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"
    CAPABILITY_CHANGED = "capability_changed"


@dataclass(frozen=True)
class Event:
    type: EventType
    change_kind: ChangeKind
    source: str
    payload: Any
    timestamp: float = field(default_factory=time)
