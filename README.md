# Loom

A scheduler that assigns jobs to a fleet of devices, learns from its own decisions, and reacts to change in milliseconds instead of recomputing everything.

Devices drop offline, urgent jobs jump the line, rules change. Most schedulers either use fixed rules that never improve, or special-case every scenario and break the first time something unplanned happens. Loom does neither: it learns from outcomes, and it handles every kind of change through one general mechanism instead of a pile of special cases.

## How it works

A contextual bandit picks the scheduler's priority weights (urgency vs. battery vs. load) based on current conditions, watches how each decision plays out, and adjusts. It never stops learning. Proof, not just a claim: two identical policies, rewarded for opposite choices, learned to make opposite decisions.

Every change, whatever it is, resolves to one of three shapes: resource changed, demand changed, rule changed. The scheduler only re-solves the part actually affected, not the whole plan. A full recompute slows down as the fleet grows; incremental replanning stays fast no matter the fleet size.

New data sources plug in without touching existing code, just implement one interface and register. Verified live: adding a source mid-run, with zero other code touched.

Categories are pluggable the same way sources are: device kinds and job requirements were never a fixed enum, just a plain string match, so a brand-new category the code has never seen gets scheduled correctly by the real solver and replanner with zero changes elsewhere — proven with a device kind that appears nowhere else in the codebase (`tests/test_pluggable_categories.py`). Loom also checks what it *could* do, not just what's assigned: a live feasibility layer reports which categories the current fleet can actually handle right now, updating the instant a device joins, drops, or fills up (`fleet/feasibility.py`, `tests/test_feasibility.py`).

Job requests can arrive as plain sentences. An LLM structures them, and can say "I don't know" instead of guessing on nonsense input. You can also ask why any decision was made and get an answer grounded in real numbers, not a plausible-sounding guess.

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

- 99 automated tests
- Incremental replan: 5-8.5ms, regardless of fleet size
- Full recompute: ~0.10s at 50 devices, over 1s at 150
- Event throughput: 80,000-86,000 events/sec, stable from 1k to 50k events

## Stack, and why

- **Google OR-Tools (CP-SAT)** for the actual scheduling solver
- **A hand-rolled contextual bandit**, not a heavier RL setup, easier to debug and reason about, and this is genuinely how similar problems get solved in production elsewhere
- **Anthropic's API** for parsing and explanations, with a rule-based fallback so the whole thing runs with zero setup and no key
- **Streamlit** for the dashboard, **pytest** for the tests

## Running it

Requires Python 3.10+.

```bash
pip install -e ".[dev]"
pytest
streamlit run dashboard/app.py
```

No API key needed. Without `ANTHROPIC_API_KEY`, parsing and explanations fall back to rule-based logic, with a banner saying so.
