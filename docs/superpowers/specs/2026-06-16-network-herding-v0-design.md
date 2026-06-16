# Network Herding v0 Design

Date: 2026-06-16

## Status

Approved direction: build a common experiment engine that can eventually support network herding, while the first runnable scenario is a sequential information cascade.

This spec defines the first implementation scope. It intentionally avoids LLM calls, chat simulation, news ingestion, polling prediction, and city-scale simulation.

## Goal

Build a small, reproducible experiment runner for studying social learning and herding dynamics.

The v0 system must:

- run a sequential information cascade experiment without any LLM calls;
- make state, observations, update policy, and metrics explicit;
- save a replay log that can be inspected after a run;
- expose interfaces that later support graph-local network herding and LLM-based update policies.

The scientific purpose is to create a trustworthy experimental skeleton before adding expensive or noisy LLM cognition.

## Non-Goals

v0 will not include:

- LLM provider integrations;
- agent-to-agent natural language chat;
- social media feeds or recommendation algorithms;
- public news ingestion;
- public-opinion or polling prediction;
- rich demographic/persona generation;
- a browser UI;
- city, mobility, economy, or daily-life simulation.

These are deferred until the core experiment loop is inspectable and tested.

## Recommended Stack

Use Python for the first implementation.

Reasons:

- social simulation libraries such as NetworkX are Python-native;
- scientific metrics and data export are straightforward;
- pytest gives a simple path for deterministic simulation tests;
- later LLM provider adapters can be added behind Python interfaces.

Initial tooling should be minimal:

- Python 3.11 or newer;
- `pytest` for tests;
- `networkx` only when graph-based experiments begin;
- standard-library `json`, `jsonl`-style line writing, `dataclasses`, and `random` for v0.

## Architecture

The architecture is an experiment engine with replaceable policies.

Core idea:

```text
ExperimentConfig
  -> Scenario
  -> Scheduler
  -> SignalModel
  -> ObservationPolicy
  -> UpdatePolicy
  -> Metrics
  -> ReplayLog
```

The engine owns the run loop. Agents do not freely act. Each step is scheduled, each observation is constructed by policy, each decision is produced by an update policy, and each state transition is logged.

This design follows the prior-art lesson: the simulator should not let LLMs or agents improvise the social structure. The environment controls what can be seen and when.

## Components

### ExperimentConfig

Defines a run.

Fields:

- `experiment_name`: e.g. `sequential_information_cascade`;
- `seed`: random seed;
- `num_agents`: number of agents;
- `true_state`: optional fixed state, otherwise sampled;
- `signal_accuracy`: probability that a private signal matches true state;
- `prior_probability`: prior probability for true state A;
- `scheduler`: e.g. `sequential`;
- `observation_policy`: e.g. `previous_actions`;
- `update_policy`: e.g. `bayesian_cascade` or `simple_heuristic`;
- `output_dir`: replay and metrics output location.

Validation rules:

- `num_agents` must be positive;
- `signal_accuracy` must be between 0.5 and 1.0 for the canonical cascade experiment;
- `prior_probability` must be greater than 0 and less than 1;
- unsupported policy names fail before the run starts.

### AgentProfile

Stable per-agent attributes.

v0 fields:

- `agent_id`;
- `prior_probability`;
- `susceptibility`, reserved for later heuristic/network policies;
- `label`, optional debug label.

v0 should not model rich personas. Rich profile design comes later after the mechanism is stable.

### AgentState

Dynamic per-agent state.

Fields:

- `agent_id`;
- `private_signal`;
- `belief_probability`;
- `confidence`;
- `action`;
- `step_index`;
- `observed_actions`;

`belief_probability` should represent probability of true state A. `action` should be a small enum-like value, such as `A` or `B`.

### SignalModel

Generates the true state and each agent's private signal.

For v0:

- true state is binary: `A` or `B`;
- each private signal is binary: `A` or `B`;
- signal matches true state with probability `signal_accuracy`;
- all draws are made from the seeded random generator owned by the run.

### Scheduler

Controls execution order.

v0 scheduler:

- `SequentialScheduler`: agents act once in fixed order `0..n-1`.

Future schedulers:

- random pair scheduler;
- synchronous graph rounds;
- feed-based exposure scheduler;
- event-driven intervention scheduler.

### ObservationPolicy

Determines what an agent can see at a decision point.

v0 policy:

- `PreviousActionsObservation`: agent sees its private signal and all previous public actions.

It does not see:

- previous private signals;
- previous beliefs;
- future actions;
- global truth.

This separation is important. The information structure is the experiment.

### UpdatePolicy

Converts an observation into belief, confidence, and action.

v0 policies:

1. `BayesianCascadePolicy`
   - Computes posterior probability from prior, private signal, and previous actions under the canonical binary cascade model.
   - Infers the likelihood of observed prior actions under each possible true state, rather than treating prior actions as hidden signals that are freely visible.
   - Chooses action `A` if posterior probability is at least 0.5, otherwise `B`.
   - Breaks exact 0.5 ties toward `A` for deterministic replay.
   - This is the rational baseline.

2. `SimpleHeuristicPolicy`
   - Uses a transparent rule, such as private signal plus majority of previous actions.
   - Useful as a non-Bayesian comparison baseline.

Future policies:

- DeGroot update;
- Friedkin-Johnsen update;
- threshold model;
- voter model;
- structured LLM decision policy;
- hybrid policy combining numeric state and LLM-generated rationale.

### Metrics

Computed at the end of each run.

v0 metrics:

- `final_accuracy`: fraction of actions matching true state;
- `correct_cascade`: whether a stable cascade forms in the true direction;
- `wrong_cascade`: whether a stable cascade forms in the false direction;
- `cascade_start_step`: first step where remaining actions no longer change direction under the policy's available observations;
- `private_signal_ignored_count`: number of agents whose action differs from their private signal;
- `action_counts`: counts of `A` and `B`;
- `belief_summary`: min, max, mean final belief probability.

Operational cascade detection for v0:

- find the earliest suffix of length at least 2 where all actions are identical;
- require at least one agent in that suffix to choose an action different from their private signal;
- mark the cascade as correct if the suffix action equals true state;
- mark it as wrong if the suffix action differs from true state;
- if no suffix satisfies these conditions, set `cascade_start_step` to null and both cascade flags to false.

This finite-run rule is intentionally operational. It is not a claim that the infinite-horizon theoretical cascade has been proven.

`confidence` is derived as `abs(belief_probability - 0.5) * 2`, yielding a value from 0 to 1.

### ReplayLog

Every run writes enough information to reproduce and inspect behavior.

Artifacts:

- `config.json`: resolved experiment configuration;
- `steps.jsonl`: one line per agent decision;
- `metrics.json`: final metrics;
- `summary.txt` or CLI output: human-readable short summary.

Each step log must include:

- `step_index`;
- `agent_id`;
- `true_state`;
- `private_signal`;
- `observed_actions`;
- `belief_probability`;
- `confidence`;
- `action`;
- `update_policy`;
- `random_seed`.

No hidden state should be required to understand why a run produced its result.

## First Scenario: Sequential Information Cascade

The first runnable scenario is `sequential_information_cascade`.

Run flow:

1. Load and validate `ExperimentConfig`.
2. Initialize seeded random generator.
3. Sample or set true state.
4. Create `num_agents` profiles.
5. Generate private signals for all agents.
6. Iterate agents in sequential order.
7. For each agent:
   - construct observation from private signal and previous public actions;
   - call selected `UpdatePolicy`;
   - update `AgentState`;
   - append step to replay log.
8. Compute metrics.
9. Write replay artifacts.
10. Return a concise run summary.

This scenario becomes the minimal scientific testbed. If it is wrong or uninspectable, larger network herding experiments will also be untrustworthy.

## Future Extension: Network Herding

The common engine should later support a graph-local scenario without rewriting the core loop.

Network herding adds:

- a `Graph` object;
- graph topology generators;
- per-round neighbor observations;
- synchronous or asynchronous rounds;
- tie weights and homophily;
- seed-node interventions;
- correction or fact-check interventions.

The main difference is observation scope. Instead of seeing all previous public actions, an agent sees actions or messages from its graph neighborhood according to `ObservationPolicy`.

## Error Handling

Configuration errors should fail before a run starts.

Examples:

- invalid probability range;
- unknown policy name;
- missing output directory permission;
- unsupported scenario name.

Runtime errors should preserve partial debugging context where possible:

- if writing a replay file fails, surface the path and operation;
- if a policy returns an invalid action, fail with agent id and step index;
- if metrics cannot be computed, keep the step log and report the metric failure.

v0 has no LLM parsing errors because v0 has no LLM calls.

## Testing Strategy

Tests should focus on determinism, information boundaries, and metric correctness.

Required tests:

1. Config validation rejects invalid probabilities and unknown policies.
2. Signal generation is deterministic for a fixed seed.
3. Private signals match true state at approximately expected rates in aggregate tests.
4. Sequential scheduler emits agents in order.
5. Observation policy exposes previous public actions but not private signals or future actions.
6. Bayesian policy returns known decisions for hand-checked small examples.
7. Heuristic policy returns known decisions for hand-checked small examples.
8. Replay log contains one step per agent and required fields.
9. Metrics identify a simple correct cascade.
10. Metrics identify a simple wrong cascade.

The first implementation should not proceed to LLM integration until these tests exist.

## CLI Shape

The first user-facing command can be small:

```bash
society-sim run examples/sequential_cascade.json
```

If packaging is deferred, the equivalent module command is acceptable:

```bash
python -m society_simulation run examples/sequential_cascade.json
```

The command should print:

- experiment name;
- true state;
- action counts;
- cascade result;
- output path.

## Acceptance Criteria

v0 is complete when:

- a sequential information cascade run can execute from a config file;
- the same seed produces identical replay logs;
- both Bayesian and heuristic policies are available;
- replay artifacts are written;
- core tests pass;
- no LLM API key is required;
- the architecture leaves a clear extension point for network-local observations and LLM update policies.

## Open Decisions Deferred

These decisions are intentionally deferred:

- exact LLM provider interface;
- OpenAI-compatible model pricing and budget estimates;
- real persona schema;
- empirical network imports;
- public-opinion data sources;
- visualization/UI design.

Deferring these keeps v0 focused on the experiment machinery rather than the whole product vision.
