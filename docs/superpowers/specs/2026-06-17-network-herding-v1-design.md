# Network Herding v1 Design

Date: 2026-06-17

## Status

Approved direction: proceed in order through:

1. network crowd psychology;
2. news/event-driven opinion movement;
3. LLM-based update policies.

This document defines the next implementation scope, `network_herding`. It also fixes the extension points needed for news events and LLM policies so the later phases can build on the same engine instead of becoming separate prototypes.

## Goal

Build a graph-local opinion dynamics scenario that can study crowd behavior such as herding, echo chambers, consensus, fragmentation, and polarization.

v1 should answer a narrow question:

> Given a fixed social graph, local observations, and a transparent update rule, how does group opinion evolve over repeated rounds?

The result should be a deterministic, replayable baseline before adding news inputs or LLM cognition.

## Non-Goals

v1 will not include:

- LLM calls;
- natural-language messages;
- real news ingestion;
- public-opinion polling prediction;
- persona generation;
- browser visualization;
- city, mobility, economy, or daily-life simulation.

Those features belong to later phases after graph-local dynamics are tested.

## Roadmap Boundaries

### Phase A: Network Herding v1

The next implementation adds graph topology, repeated rounds, local neighbor observations, network update policies, time-series metrics, and replay.

### Phase B: News/Event Opinion v2

After v1 works, external events enter as structured shocks. A news/event object should have fields such as topic, stance, credibility, salience, and source. Exposure is controlled by an `ExposurePolicy`, not by free-form agent improvisation.

v1 must leave room for this by making observations and replay entries extensible.

### Phase C: LLM Policy v3

After deterministic baselines exist, an LLM policy can replace or augment numeric update rules. It must use structured output, provider configuration, prompt replay, response replay, caching, and cost accounting.

v1 must leave room for this by keeping update policies behind a shared interface.

## Architecture

The v0 cascade runner is a single sequential scenario. v1 introduces a scenario dispatcher while keeping v0 behavior intact.

Target shape:

```text
ExperimentConfig
  -> ScenarioRunner
  -> Topology
  -> Scheduler
  -> ObservationPolicy
  -> UpdatePolicy
  -> Metrics
  -> ReplayWriter
```

The simulator still owns the social structure. Agents do not decide who they can see, when they act, or which global state is available. The experiment configuration controls graph topology, scheduling, observation scope, update policy, and interventions.

## Configuration

The existing `ExperimentConfig` is narrow and tied to `sequential_information_cascade`. v1 should preserve compatibility while adding fields for network runs.

Recommended config shape:

```json
{
  "experiment_name": "network_herding",
  "seed": 42,
  "num_agents": 100,
  "initial_opinion": {
    "type": "bernoulli",
    "probability_a": 0.5
  },
  "topology": {
    "type": "small_world",
    "degree": 6,
    "rewiring_probability": 0.1
  },
  "scheduler": {
    "type": "synchronous_rounds",
    "rounds": 25
  },
  "observation_policy": {
    "type": "neighbor_actions"
  },
  "update_policy": {
    "type": "threshold",
    "adoption_threshold": 0.6
  },
  "output_dir": "runs/network_herding"
}
```

v1 should support only a small validated subset. Unknown topology, scheduler, observation, or update policy names must fail before a run starts.

## Opinion Model

v1 should support a binary public action, `A` or `B`, because that matches the current v0 action type and keeps the first graph experiment simple.

Internally, each agent should also carry a numeric `belief_probability`, interpreted as probability or support for `A`. This leaves a path to continuous opinions without changing replay format.

Per-agent dynamic state:

- `agent_id`;
- `belief_probability`;
- `confidence`;
- `action`;
- `round_index`;
- `observed_neighbor_actions`;
- `observed_neighbor_ids`.

The first v1 implementation does not need rich agent personas. Static profile fields such as susceptibility and label can be reused later.

## Graph Topology

v1 should implement graph generation without runtime third-party dependencies if practical. Since NetworkX is not yet in the project, the first implementation can use standard-library adjacency sets.

Supported topology types for v1:

1. `complete`
   - Useful sanity check. Every agent sees every other agent.

2. `cycle`
   - Useful low-complexity local-neighborhood baseline.

3. `erdos_renyi`
   - Random graph with edge probability.

4. `small_world`
   - Ring lattice with random rewiring. This is the most important v1 topology for echo-chamber-like local clustering.

`scale_free` and stochastic block models are valuable, but can be deferred if v1 would otherwise become too large.

Graph object requirements:

- deterministic from seed and topology config;
- no self-edges;
- undirected for v1;
- stable adjacency ordering for replay and tests;
- validation that all node ids are in `0..num_agents-1`;
- graph metadata written to replay.

## Scheduler

v1 should support one scheduler first:

- `synchronous_rounds`: all agents observe the previous round and update simultaneously.

This is easier to reason about than asynchronous updates and avoids hidden ordering effects in the first network scenario.

Future schedulers:

- asynchronous random node updates;
- random pair interactions;
- event-driven exposure scheduler;
- feed/recommendation scheduler.

## Observation Policy

v1 observation policy:

- `neighbor_actions`: each agent sees only the previous round actions of graph neighbors.

It must not expose:

- global action counts;
- non-neighbor actions;
- private signals from v0;
- true state;
- future updates from the same round.

For deterministic debugging, the observation should include neighbor ids and observed neighbor actions in stable sorted neighbor order.

## Update Policies

v1 should implement transparent non-LLM policies first.

Required v1 policies:

- majority rule;
- threshold policy;
- DeGroot policy.

Deferred policies:

- voter model;
- Friedkin-Johnsen policy.

### Majority Rule

Choose the local neighbor majority. Tie breaks toward the agent's current action. This creates a simple social conformity baseline.

### Threshold Policy

Switch to `A` if the local fraction of `A` neighbors is at or above `adoption_threshold`. Switch to `B` if the local fraction of `B` neighbors is at or above the threshold. Otherwise keep the current action.

This is useful for contagion and adoption dynamics.

### Voter Model

Copy one randomly selected neighbor from the previous round. This gives a stochastic baseline. It must use the run seed.

This is not required for the first v1 implementation. It is a follow-up policy after the deterministic policies are stable.

### DeGroot Policy

Update numeric belief as a weighted average of the agent's current belief and neighbor beliefs. Convert belief to action using `belief >= 0.5`.

This is the bridge to continuous opinion dynamics.

### Friedkin-Johnsen Policy

Blend social influence with an anchored initial belief. This introduces stubbornness and is useful for polarization/echo-chamber experiments.

This is not required for the first v1 implementation. It is the next continuous-opinion policy after DeGroot.

## Metrics

v1 metrics should be computed as time series and final summary.

Required final metrics:

- `final_action_counts`;
- `final_a_fraction`;
- `consensus_reached`;
- `consensus_action`;
- `time_to_consensus`;
- `polarization_index`;
- `opinion_variance`;
- `mean_belief`;
- `edge_disagreement_rate`;
- `component_count`.

Required time-series metrics per round:

- `round_index`;
- `a_fraction`;
- `belief_mean`;
- `belief_variance`;
- `edge_disagreement_rate`;
- `action_changes`.

Definitions:

- `consensus_reached`: all agents have the same action at final round.
- `time_to_consensus`: first round where all actions are identical and remain identical through the final round.
- `polarization_index`: absolute distance between the mean belief above 0.5 and the mean belief below 0.5, multiplied by the fraction of agents not near the center. This is operational, not a universal polarization theorem.
- `edge_disagreement_rate`: fraction of graph edges whose endpoints have different actions.

## Replay

v1 replay should write:

- `config.json`;
- `graph.json`;
- `steps.jsonl`;
- `metrics.json`;
- `timeseries.jsonl`;
- `summary.txt`.

Each `steps.jsonl` row should represent one agent update in one round:

- `round_index`;
- `agent_id`;
- `previous_belief_probability`;
- `belief_probability`;
- `previous_action`;
- `action`;
- `confidence`;
- `observed_neighbor_ids`;
- `observed_neighbor_actions`;
- `update_policy`;
- `random_seed`.

`graph.json` should include topology config and adjacency lists. Adjacency lists should use string keys or sorted integer-key-compatible JSON structures, but tests must load them and compare deterministically.

## CLI

The existing command should keep working:

```bash
python -m society_simulation run examples/sequential_cascade.json
```

v1 should add an example:

```bash
python -m society_simulation run examples/network_herding.json
```

The CLI does not need a new subcommand. The experiment name in the config selects the scenario.

## Error Handling

Configuration errors fail before simulation begins:

- unsupported topology type;
- invalid probability;
- invalid round count;
- invalid threshold;
- impossible graph parameters;
- unsupported update policy.

Runtime errors should include scenario, round, and agent id where possible.

Replay write failures should report the output path and should not be mislabeled as config-read failures in new code. If the existing CLI wrapper remains broad, v1 should split load-time errors from run-time errors.

## Testing Strategy

Tests should focus on graph determinism, local information boundaries, synchronous update semantics, metric correctness, and replay integrity.

Required tests:

1. Topology generation is deterministic for fixed seed.
2. Topology validation rejects invalid graph parameters.
3. Neighbor observations expose only graph neighbors.
4. Synchronous rounds use previous-round state, not partially updated current-round state.
5. Majority and threshold policies handle ties deterministically.
6. DeGroot updates numeric belief as expected on a tiny graph.
7. Network metrics match hand-computed values on small graphs.
8. Replay writes `graph.json`, `steps.jsonl`, `timeseries.jsonl`, `metrics.json`, and `summary.txt`.
9. CLI runs `examples/network_herding.json`.
10. Existing sequential cascade tests continue to pass.

## Implementation Notes

Keep v1 small and inspectable.

Recommended module additions:

- `graph.py`: graph model and topology generators;
- `network_models.py`: network-specific state and observation dataclasses if the v0 models become too narrow;
- `network_policies.py`: graph-local update policies;
- `network_metrics.py`: network time-series and final metrics;
- `network_runner.py`: `network_herding` scenario runner.

Recommended refactors:

- Introduce scenario dispatch in `runner.py`.
- Keep the existing v0 sequential code behind a dedicated function.
- Keep config validation explicit rather than permissive.

Avoid introducing NetworkX in v1 unless standard-library graph generation becomes awkward. The smaller dependency surface is better for the first graph baseline.

## Success Criteria

v1 is successful when:

- `python -m society_simulation run examples/network_herding.json` completes without LLM calls;
- the run writes graph, step, time-series, final metrics, and summary artifacts;
- two runs with the same seed produce identical replay artifacts;
- tests cover small hand-checkable graphs and policies;
- old sequential cascade behavior still passes all tests;
- the design leaves clear extension points for news events and LLM policies.

## Deferred v2: News/Event Opinion

v2 should add structured external events, not raw unbounded news text.

Likely objects:

- `NewsEvent`: topic, stance, credibility, salience, source, round index;
- `ExposurePolicy`: who sees which event;
- `SourceTrust`: per-agent or per-group trust in sources;
- `Intervention`: fact-check, counter-message, removal, downranking.

The first v2 claim should be modest: compare opinion movement under controlled event exposure, not predict real polls.

## Deferred v3: LLM Policy

v3 should add an OpenAI-compatible `LLMUpdatePolicy`.

Requirements:

- provider config separate from experiment config secrets;
- structured JSON response schema;
- strict parsing and validation;
- prompt, response, model, token usage, and estimated cost in replay;
- cache by normalized prompt and model config;
- deterministic non-LLM baseline run for every LLM experiment;
- support for low-cost OpenAI-compatible models where available.

LLM agents are the cognitive layer. They should not own graph generation, exposure rules, scheduler behavior, metrics, or replay.
