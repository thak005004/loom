"""Tests for the adaptive policy (Section 3a). The most important one
is `test_bandit_weights_measurably_shift_toward_consistently_rewarded_pattern`:
it's the proof the "AI" here is genuinely adaptive rather than a fixed
weighting dressed up as one.
"""

from __future__ import annotations

import random

import pytest

from orchestrator.policy.bandit_policy import ARMS, BanditPolicy, Context, context_features
from orchestrator.policy.reward import compute_reward
from orchestrator.scheduling.solver import Assignment, ScheduleResult, solve


def test_context_features_include_bias_and_all_three_signals():
    context = Context(avg_battery=0.7, avg_load=0.3, recent_failure_rate=0.05)
    assert context_features(context) == [1.0, 0.7, 0.3, 0.05]


def test_update_before_select_weights_raises():
    policy = BanditPolicy(rng=random.Random(0))
    with pytest.raises(RuntimeError):
        policy.update(1.0)


def test_theta_moves_toward_observed_reward():
    policy = BanditPolicy(rng=random.Random(0), epsilon=0.0, learning_rate=0.5)
    context = Context(avg_battery=0.5, avg_load=0.5, recent_failure_rate=0.0)

    policy.select_weights(context)  # theta all zero -> arm 0 picked by tie-break
    before = policy._predict(0, context_features(context))
    policy.update(1.0)
    after = policy._predict(0, context_features(context))

    assert after > before  # moved toward the reward it just observed


def test_bandit_weights_measurably_shift_toward_consistently_rewarded_pattern():
    """Train two otherwise-identical policies on the same context, one
    consistently rewarded for picking arm 1, the other for arm 2. If the
    policy is genuinely adaptive (not decorative), each should converge
    to preferring the arm it was rewarded for, and the two trained
    policies should disagree with each other on the same input."""

    context = Context(avg_battery=0.6, avg_load=0.4, recent_failure_rate=0.1)
    rounds = 500

    def train_toward(target_index: int, seed: int) -> BanditPolicy:
        policy = BanditPolicy(rng=random.Random(seed), epsilon=0.3, learning_rate=0.3)
        for _ in range(rounds):
            policy.select_weights(context)
            reward = 1.0 if policy.last_arm_index == target_index else 0.0
            policy.update(reward)
        return policy

    policy_favoring_1 = train_toward(target_index=1, seed=1)
    policy_favoring_2 = train_toward(target_index=2, seed=2)

    # force pure exploitation to read out the learned preference cleanly
    policy_favoring_1.epsilon = 0.0
    policy_favoring_2.epsilon = 0.0

    weights_1 = [policy_favoring_1.select_weights(context) for _ in range(10)]
    weights_2 = [policy_favoring_2.select_weights(context) for _ in range(10)]

    assert all(w == ARMS[1] for w in weights_1)
    assert all(w == ARMS[2] for w in weights_2)
    # same context, opposite reward histories -> different learned choice;
    # the shift is driven by the reward signal, not a fixed preference
    assert weights_1[0] != weights_2[0]


def test_compute_reward_is_high_when_urgent_jobs_served_and_load_balanced():
    devices = [{"device_id": "dev-a", "kind": "gpu", "connected": True}, {"device_id": "dev-b", "kind": "gpu", "connected": True}]
    jobs = [{"job_id": "job-1", "priority": "urgent"}, {"job_id": "job-2", "priority": "normal"}]
    result = ScheduleResult(
        status="OPTIMAL",
        assignments=[Assignment(job_id="job-1", device_id="dev-a"), Assignment(job_id="job-2", device_id="dev-b")],
        unassigned_job_ids=[],
    )

    reward = compute_reward(result, jobs, devices)
    assert reward > 0.9


def test_compute_reward_is_low_when_urgent_jobs_go_unassigned():
    devices = [{"device_id": "dev-a", "kind": "gpu", "connected": True}]
    jobs = [{"job_id": "job-1", "priority": "urgent"}, {"job_id": "job-2", "priority": "urgent"}]
    result = ScheduleResult(status="OPTIMAL", assignments=[], unassigned_job_ids=["job-1", "job-2"])

    reward = compute_reward(result, jobs, devices)
    assert reward < 0.5


def test_compute_reward_end_to_end_with_real_solver_output():
    devices = [{"device_id": "dev-a", "kind": "gpu", "battery": 90.0, "load": 0.1, "connected": True}]
    jobs = [{"job_id": "job-1", "requires": "gpu", "priority": "urgent"}]

    result = solve(jobs, devices)
    reward = compute_reward(result, jobs, devices)

    assert len(result.assignments) == 1
    assert reward > 0.9
