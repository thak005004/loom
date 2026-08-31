"""Adaptive scheduling policy (Section 3a): a contextual bandit that
picks which of `solve()`'s four objective weights to use, given the
current fleet context, and keeps updating that choice from observed
outcomes — continuously, with no separate train/freeze split.

Design: a small set of fixed weight profiles ("arms"), each a
reasonable named strategy (balanced / priority-aggressive /
load-balance-aggressive / battery-conservative). For each arm, the
policy keeps a linear estimate of "how good has this arm been in a
context like this one" (a weight vector over the context features,
updated toward the observed reward after every round via the
Widrow-Hoff / LMS rule — the textbook simple linear bandit, chosen over
a fancier method because it's the whole point to stay debuggable: you
can print `policy.theta` and read the learned preference directly).

Selection is epsilon-greedy: usually exploit the arm with the highest
predicted value for the current context, occasionally explore a random
arm. Exploration never turns off — there's no "training phase" that
ends; every call to `update()` nudges the estimates a little further,
which is what "continuously adapts, doesn't train once and freeze"
means concretely here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from orchestrator.fleet.state import WorldState

WeightProfile = Dict[str, float]

# Named, fixed weight profiles the bandit chooses between. Keys match
# solve()'s scalar kwargs exactly, so a chosen profile can be handed to
# solve(**profile) directly.
ARMS: List[WeightProfile] = [
    {  # balanced — the plan's own reasonable default
        "priority_weight": 1.0,
        "load_penalty_scale": 2.0,
        "battery_bonus_scale": 1.0,
        "urgent_unassigned_penalty": 500.0,
    },
    {  # priority-aggressive — push hard to serve urgent jobs first
        "priority_weight": 2.0,
        "load_penalty_scale": 1.0,
        "battery_bonus_scale": 0.5,
        "urgent_unassigned_penalty": 1000.0,
    },
    {  # load-balance-aggressive — spread work evenly even at some
        # priority cost
        "priority_weight": 0.5,
        "load_penalty_scale": 4.0,
        "battery_bonus_scale": 1.0,
        "urgent_unassigned_penalty": 200.0,
    },
    {  # battery-conservative — protect low-battery devices
        "priority_weight": 1.0,
        "load_penalty_scale": 1.0,
        "battery_bonus_scale": 3.0,
        "urgent_unassigned_penalty": 500.0,
    },
]

ARM_NAMES = ["balanced", "priority_aggressive", "load_balance_aggressive", "battery_conservative"]


@dataclass(frozen=True)
class Context:
    """Fleet state fed to the bandit before each scheduling round.
    All three features are 0-1 scaled so no one dimension dominates the
    linear model just because of its raw units."""

    avg_battery: float  # mean device battery, 0-1 (100% -> 1.0)
    avg_load: float  # mean device load, already 0-1
    recent_failure_rate: float  # 0-1; from the maintenance-log stream once
    # it exists (Day 2) — until then, 0.0 or a caller-estimated value


def context_features(context: Context) -> List[float]:
    # leading 1.0 is the bias term
    return [1.0, context.avg_battery, context.avg_load, context.recent_failure_rate]


def context_from_world(world: WorldState, recent_failure_rate: float = 0.0) -> Context:
    devices = world.available_devices()
    if not devices:
        return Context(avg_battery=1.0, avg_load=0.0, recent_failure_rate=recent_failure_rate)
    avg_battery = sum(d.get("battery", 0.0) for d in devices) / len(devices) / 100.0
    avg_load = sum(d.get("load", 0.0) for d in devices) / len(devices)
    return Context(avg_battery=avg_battery, avg_load=avg_load, recent_failure_rate=recent_failure_rate)


class BanditPolicy:
    def __init__(
        self,
        arms: List[WeightProfile] = ARMS,
        epsilon: float = 0.15,
        learning_rate: float = 0.1,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.arms = list(arms)
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.rng = rng or random.Random()
        n_features = len(context_features(Context(0.0, 0.0, 0.0)))
        self.theta: List[List[float]] = [[0.0] * n_features for _ in self.arms]

        self.last_arm_index: Optional[int] = None
        self.last_context_features: Optional[List[float]] = None

    def _predict(self, arm_index: int, features: List[float]) -> float:
        return sum(w * f for w, f in zip(self.theta[arm_index], features))

    def select_weights(self, context: Context) -> WeightProfile:
        features = context_features(context)
        if self.rng.random() < self.epsilon:
            arm_index = self.rng.randrange(len(self.arms))
        else:
            values = [self._predict(i, features) for i in range(len(self.arms))]
            arm_index = max(range(len(self.arms)), key=lambda i: values[i])

        self.last_arm_index = arm_index
        self.last_context_features = features
        return dict(self.arms[arm_index])

    def update(self, reward: float) -> None:
        """Nudge the chosen arm's value estimate toward the observed
        reward from the round `select_weights()` just set up. Call this
        once per round, right after `select_weights()` — there's no
        separate training mode, this *is* the whole learning loop."""
        if self.last_arm_index is None or self.last_context_features is None:
            raise RuntimeError("update() called before select_weights()")

        i = self.last_arm_index
        features = self.last_context_features
        predicted = self._predict(i, features)
        error = reward - predicted
        self.theta[i] = [w + self.learning_rate * error * f for w, f in zip(self.theta[i], features)]
