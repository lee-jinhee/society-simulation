# Sweep Runner Design

Date: 2026-06-17

## Status

Approved direction: build a no-LLM sweep runner before adding news shocks or LLM update policies.

This design defines a small experiment manager that repeatedly runs existing simulator configs under controlled parameter changes. It does not change the behavior of `network_herding`; it turns single-run scenarios into comparable batches.

## Goal

Build a deterministic, no-LLM sweep runner that answers questions such as:

> Across many seeds and network conditions, which topology and update parameters lead to consensus, fragmentation, or polarization?

The result should make the current simulator useful as a research tool instead of a one-config demo runner.

## Non-Goals

The first sweep runner will not include:

- LLM calls;
- model-provider configuration;
- token or API cost estimation;
- parallel execution;
- charts or browser visualization;
- statistical hypothesis testing;
- external news/event shocks;
- resumable distributed jobs.

Those are later features. The first version should produce clean machine-readable outputs that those features can consume.

## User-Facing Shape

Add a new CLI subcommand:

```bash
python -m society_simulation sweep experiments/network_topology_sweep.json
```

The existing command remains unchanged:

```bash
python -m society_simulation run examples/network_herding.json
```

The sweep command prints a concise summary:

```text
sweep=network_topology_sweep
runs=480
completed=480
failed=0
output_dir=runs/sweeps/network_topology_sweep
summary_csv=runs/sweeps/network_topology_sweep/summary.csv
```

## Sweep Config

A sweep config has four top-level responsibilities:

- identify the sweep;
- provide a base experiment config;
- define parameter factors;
- define the sweep output directory.

Recommended shape:

```json
{
  "sweep_name": "network_topology_sweep",
  "base_config": {
    "experiment_name": "network_herding",
    "seed": 1,
    "num_agents": 30,
    "initial_opinion": {
      "type": "bernoulli",
      "probability_a": 0.5
    },
    "topology": {
      "type": "small_world",
      "degree": 4,
      "rewiring_probability": 0.1
    },
    "scheduler": {
      "type": "synchronous_rounds",
      "rounds": 12
    },
    "observation_policy": {
      "type": "neighbor_actions"
    },
    "update_policy": {
      "type": "threshold",
      "adoption_threshold": 0.6
    },
    "output_dir": "runs/ignored-by-sweep"
  },
  "factors": [
    {
      "name": "seed",
      "path": "seed",
      "values": [1, 2, 3, 4, 5]
    },
    {
      "name": "initial_a",
      "path": "initial_opinion.probability_a",
      "values": [0.45, 0.5, 0.55]
    },
    {
      "name": "topology",
      "values": [
        {
          "label": "complete",
          "overrides": {
            "topology": {
              "type": "complete"
            }
          }
        },
        {
          "label": "cycle",
          "overrides": {
            "topology": {
              "type": "cycle"
            }
          }
        },
        {
          "label": "small_world_k4_p01",
          "overrides": {
            "topology": {
              "type": "small_world",
              "degree": 4,
              "rewiring_probability": 0.1
            }
          }
        },
        {
          "label": "erdos_renyi_p02",
          "overrides": {
            "topology": {
              "type": "erdos_renyi",
              "edge_probability": 0.2
            }
          }
        }
      ]
    },
    {
      "name": "threshold",
      "path": "update_policy.adoption_threshold",
      "values": [0.55, 0.6, 0.7]
    }
  ],
  "output_dir": "runs/sweeps/network_topology_sweep"
}
```

The `path` form is for one-field changes. The `overrides` form is for correlated changes, such as topology type plus required topology parameters. This prevents invalid configs such as `complete` topology carrying `degree`.

## Validation Rules

The sweep runner validates before launching the batch:

- `sweep_name` must be a non-empty string;
- `base_config` must be a valid existing experiment config after applying each run's overrides;
- `factors` must be a non-empty list;
- each factor must have a non-empty `name`;
- factor names must be unique;
- a factor value must use either scalar path replacement or labeled overrides;
- labels in one factor must be unique after string conversion;
- generated run ids must be unique;
- generated `output_dir` values must live under the sweep `output_dir`;
- every generated config must pass the existing `load_config`/config validation path before execution.

Validation should fail before any run starts when the sweep definition is structurally invalid. If a run fails during execution despite preflight validation, the sweep should record the failure and continue by default.

## Run Expansion

The runner creates the Cartesian product of factor values in the order listed. Each generated run receives:

- a stable `run_id`;
- a factor label map;
- a fully materialized experiment config;
- a unique per-run `output_dir`.

Run ids should be deterministic and filesystem-safe:

```text
seed-1__initial_a-0_45__topology-complete__threshold-0_55
```

Each per-run output directory should be:

```text
<sweep_output_dir>/runs/<run_id>
```

The generated experiment config's own `output_dir` is overwritten with that per-run directory. This prevents factor changes from accidentally writing over each other.

## Artifacts

A successful sweep writes:

- `sweep_config.json`: normalized sweep config;
- `manifest.jsonl`: one row per planned run, including run id, labels, output dir, status, and error if any;
- `summary.csv`: one row per planned run with factor labels, status, errors, and selected metrics when available;
- `summary.json`: aggregate counts and basic metric means grouped by factor labels;
- `runs/<run_id>/...`: normal single-run artifacts from the existing simulator.

`summary.csv` should include at least:

- `run_id`;
- factor columns;
- `experiment_name`;
- `output_dir`;
- `status`;
- `error`;
- `final_action_counts_A`;
- `final_action_counts_B`;
- `final_a_fraction`;
- `consensus_reached`;
- `consensus_action`;
- `time_to_consensus`;
- `polarization_index`;
- `opinion_variance`;
- `mean_belief`;
- `edge_disagreement_rate`;
- `component_count`.

For non-network experiments, the summary writer should include available generic metrics and leave missing network-specific metrics empty. The first example sweep should target `network_herding`.

## Components

### `sweep_config.py`

Loads and validates sweep definitions. It should keep parsing strict and use structured errors that the CLI can print cleanly.

### `sweep_runner.py`

Expands factors, materializes configs, dispatches each generated config through `run_experiment`, and records outcomes. It should call the existing single-run path rather than duplicating scenario logic.

### `sweep_artifacts.py`

Writes manifest, summary CSV, aggregate JSON, and normalized sweep config. Artifact writing should be deterministic so git diffs and regression tests are stable.

### CLI

Extends the parser with `sweep <config>`. The existing `run <config>` path should keep its current output and behavior.

## Data Flow

```text
sweep JSON
  -> SweepConfig
  -> factor product
  -> materialized experiment config dicts
  -> existing config loader
  -> existing run_experiment
  -> per-run artifacts
  -> manifest and summary artifacts
```

The sweep layer does not know how network opinions update. It only knows how to generate valid configs, run them, and summarize returned metrics.

## Error Handling

Structural sweep errors should stop the entire sweep before execution. Examples:

- missing `base_config`;
- duplicate factor names;
- a path that cannot be applied;
- a factor value with both scalar and overrides forms;
- generated configs that fail validation.

Runtime run errors should be captured in the manifest and summary with `status=failed`. The CLI should return non-zero if any run failed. This makes failures visible in automation while still preserving partial outputs for debugging.

## Testing

Tests should cover:

- loading a valid sweep config;
- rejecting malformed factor definitions;
- expanding factors deterministically;
- applying scalar paths and override bundles;
- validating all generated configs before execution;
- generating unique output directories;
- writing manifest, summary CSV, and aggregate JSON;
- CLI `sweep` success path;
- CLI `sweep` invalid config error path;
- existing CLI `run` regression.

The implementation should stay standard-library only.

## First Example Sweep

Add:

```text
experiments/network_topology_sweep.json
```

Keep it small enough for a quick local run:

- 4 topologies;
- 3 seeds;
- 2 initial opinion probabilities;
- 2 thresholds.

That produces 48 runs, which is enough to verify the whole pipeline without creating a huge artifact set.

## Later Extensions

Once no-LLM sweeps are stable, add:

- `max_runs` and `dry_run`;
- optional parallel execution;
- richer grouped summaries;
- plotting scripts;
- budget estimation for future LLM policies;
- cache-aware LLM sweep support;
- resume from manifest.
