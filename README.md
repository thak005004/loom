# Loom

A scheduler that assigns jobs to a fleet of devices, learns from its own decisions, and reacts to change in milliseconds instead of recomputing everything.

## Purpose

Loom decides who does what, in real time, as things keep changing. Devices drop offline, urgent jobs jump the line, rules change. Instead of fixed rules that never improve, or special-case code that breaks on anything unplanned, Loom learns from outcomes and handles every kind of change through one general mechanism.

## How it works

A contextual bandit picks the scheduler's priority weights (urgency vs. battery vs. load) based on current conditions, watches how each decision plays out, and adjusts. It never stops learning. Proof: two identical policies, rewarded for opposite choices, learned to make opposite decisions.

Every change (device dies, job arrives, rule changes) resolves to one of three shapes: resource changed, demand changed, rule changed. The scheduler only re-solves the part actually affected, not the whole plan. A full recompute slows down as the fleet grows; incremental replanning stays fast regardless of fleet size.

New data sources plug in without touching existing code, just implement one interface and register. Verified live: adding a source mid-run, zero other code touched.

Job requests can arrive as plain sentences. An LLM structures them, and can say "I don't know" instead of guessing on nonsense. You can also ask why any decision was made and get an answer grounded in real numbers. Both fall back to simple rule-based logic with no API key set.

## How it fits together

```
                    Multiple data sources
                  (devices, jobs, rules, more)
                              |
                              v
                          Event bus
               (normalizes every kind of change)
                              |
                              v
                         World state
                    (live devices and jobs)
                              |
                  +-----------+-----------+
                  v                       v
          Adaptive policy  <----->    Scheduler
        (learns from outcomes)  (assigns jobs to devices)
                  |                       |
                  +-----------+-----------+
                              v
                   Incremental re-planner
                    (reacts to any change)
                              |
                              v
                  Explainer + live dashboard
                (shows state, answers "why")
```

## An honest result

Fairness (how evenly work spreads across devices) doesn't improve as the policy learns, it actually got slightly worse across test runs. The reward weights urgent work far above balance, so the policy is optimizing correctly, just not for fairness. A periodic full rebalance would fix it. I didn't build that, and would rather report the real result than a rosier one.

## The numbers

- 76 automated tests
- Incremental replan: 5-8.5ms, regardless of fleet size
- Full recompute: ~0.10s at 50 devices, over 1s at 150
- Event throughput: 80,000-86,000 events/sec, stable from 1k to 50k events

## Stack

Python, Google OR-Tools (CP-SAT), a hand-rolled contextual bandit, Anthropic's API with a rule-based fallback, Streamlit, pytest.

## Running it

Requires Python 3.10+.

```bash
pip install -e ".[dev]"
pytest
streamlit run dashboard/app.py
```

No API key needed. Without `ANTHROPIC_API_KEY`, parsing and explanations fall back to rule-based logic, with a banner saying so.
