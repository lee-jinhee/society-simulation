# Event-Driven Congestion Pricing Pilot

Date: 2026-06-19

## Status

Mock-policy pilot completed. This is not a claim about real human opinion. It verifies that the event-driven scenario can produce auditable traces with agents, relationships, staged events, exposures, messages, private/public stance, memory, and metrics.

## Purpose

The purpose is to replace the previous binary neighbor-action toy with a richer trace-validity experiment. The key question is whether the simulator can represent people encountering events and conversations over time, not whether a population converges to a label.

## Configuration

- Scenario: congestion pricing in a fictional mid-sized city.
- Agents: 8.
- Days: 7 plus day 0 initial state.
- Policy: mock persona.
- Output: `runs/event_driven_congestion_pricing`.

## Verification Run

The mock run completed through the CLI with:

- final_private_stance_mean: `-0.5375`
- final_public_stance_mean: `-0.24375`
- final_private_public_gap: `0.29375`
- message_count: `56`
- llm_calls: `56`
- llm_estimated_cost_usd: `0.00000000`

## What To Inspect

- `exposures.jsonl`: what each agent saw.
- `messages.jsonl`: what each agent said.
- `agent_states.jsonl`: private stance, public stance, confidence, salience, emotion, memory.
- `llm_decisions.jsonl`: prompt and structured decision audit.
- `metrics.json`: aggregate public/private stance and message metrics.

## Next Paid Pilot

Run the same config with `update_policy.type = "llm"` and a strict cost cap after reviewing mock traces. The paid pilot should use 8 agents, 7 days, max completion tokens around 160, and one seed.
