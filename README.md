# Loom

A scheduler that assigns jobs to a fleet of devices, learns from its own decisions, and reacts to change in milliseconds instead of recomputing everything.

Devices drop offline, urgent jobs jump the line, rules change. Most schedulers use fixed rules that never improve, or special-case every scenario and break the first time something unplanned happens. Loom does neither, and it doesn't wait on a person to notice and patch in a fix: the moment something changes, it's already adjusting, on its own.

## How it works

- **It learns.** A contextual bandit picks the scheduler's priority weights (urgency vs. battery vs. load) based on current conditions, watches how each decision plays out, and adjusts. It never stops learning. Proof, not just a claim: two identical policies, rewarded for opposite choices, learned to make opposite decisions.

- **It reacts to any kind of change, not just the ones someone thought of.** Every change resolves to one of three shapes: resource changed, demand changed, rule changed. The scheduler only re-solves the part actually affected. A full recompute slows down as the fleet grows; incremental replanning stays fast no matter the fleet size.

- **New data sources plug in without touching existing code.** Any source just implements one interface and registers itself. Verified live: adding a source mid-run, zero other code touched.

- **New categories work the same way.** Device kinds and job requirements are just string matches, not a fixed list, so a brand-new category the code has never seen gets scheduled correctly with zero changes elsewhere. There's also a live feasibility check: what could the current fleet actually handle right now, not just what's already assigned.

- **It understands messy input and explains itself.** Job requests can arrive as plain sentences; an LLM structures them, and can say "I don't know" instead of guessing on nonsense. You can ask why any decision was made and get an answer grounded in real numbers.

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

Fairness (how evenly work spreads across devices) doesn't improve as the policy learns, it got slightly worse across test runs. The reward weights urgent work far above balance, so the policy is optimizing correctly, just not for fairness. A periodic full rebalance would fix it. Didn't build that yet, wanted to report the real result instead.

## The numbers

- 99 automated tests
- Incremental replan: 5-8.5ms, regardless of fleet size
- Full recompute: ~0.10s at 50 devices, over 1s at 150
- Event throughput: 80,000-86,000 events/sec, stable from 1k to 50k events

## Stack, and why

- **Google OR-Tools (CP-SAT)** for the scheduling solver
- **A hand-rolled contextual bandit**, not a heavier RL setup, easier to debug and how similar problems get solved in production elsewhere
- **Anthropic's API** for parsing and explanations, with a rule-based fallback so it runs with zero setup and no key
- **Streamlit** for the dashboard, **pytest** for the tests

## Running it

Requires Python 3.10+.

```bash
pip install -e ".[dev]"
pytest
streamlit run dashboard/app.py
```

No API key needed. Without `ANTHROPIC_API_KEY`, parsing and explanations fall back to rule-based logic, with a banner saying so.
