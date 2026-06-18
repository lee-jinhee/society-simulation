# Sweep Analyzer Design

Date: 2026-06-18

## Status

Approved direction: build a no-LLM sweep result analyzer before adding agent heterogeneity, news shocks, or LLM update policies.

The sweep runner can now execute repeated network experiments and write machine-readable artifacts. The next step is to turn those artifacts into a compact research report that supports decisions about what experiment to run next.

## Goal

Build a deterministic analyzer for completed sweep outputs. It should answer questions such as:

> Which topology and threshold settings produce consensus, fragmentation, or polarization most often?

The analyzer should make a sweep directory useful without opening raw CSV files by hand.

## Non-Goals

The first analyzer will not include:

- plotting or browser visualization;
- statistical significance tests;
- regression modeling;
- causal inference;
- notebook generation;
- LLM-written interpretation;
- external data ingestion;
- polling or news prediction;
- parallel execution.

Those features can build on this analyzer later. The first version should produce simple, auditable summaries.

## User-Facing Shape

Add a new CLI subcommand:

```bash
python -m society_simulation analyze runs/sweeps/network_topology_sweep
```

The command reads an existing sweep output directory and prints:

```text
analysis=network_topology_sweep
runs=48
completed=48
failed=0
output_dir=runs/sweeps/network_topology_sweep/analysis
report=runs/sweeps/network_topology_sweep/analysis/report.md
```

The command writes:

- `analysis/report.md`: human-readable Markdown report;
- `analysis/group_summary.csv`: one row per factor label;
- `analysis/group_summary.json`: machine-readable grouped metrics;
- `analysis/failure_summary.csv`: failed or incomplete runs, if any.

## Input Contract

The analyzer reads these existing sweep artifacts:

- `summary.csv`;
- `summary.json`;
- `manifest.jsonl`;
- `sweep_config.json`.

`summary.csv` is the primary source for per-run metrics. `summary.json` provides aggregate counts. `sweep_config.json` provides factor order and sweep name. `manifest.jsonl` is used for failure details and consistency checks.

The analyzer should fail cleanly if required files are missing or malformed.

## Metrics

The first analyzer should compute grouped summaries for every factor in the sweep config.

For each factor label, compute:

- `runs`;
- `completed`;
- `failed`;
- `consensus_rate`;
- `mean_final_a_fraction`;
- `mean_polarization_index`;
- `mean_edge_disagreement_rate`;
- `mean_time_to_consensus`;
- `mean_opinion_variance`;
- `mean_component_count`.

Rows with `status == completed` should count toward `completed`. Rows with `status == failed` should count toward `failed`. Other non-completed statuses, such as `pending`, should count toward `runs` but should not count as failed. Non-completed rows should not contribute to metric means.

Missing metric values should be ignored for means. If no completed rows have a metric, the mean should be empty in CSV and `null` in JSON.

`consensus_rate` should be computed over completed rows only: completed rows with `consensus_reached == true` divided by completed rows with a parseable `consensus_reached` value. If a group has no completed rows with consensus data, the value should be empty in CSV and `null` in JSON.

## Grouping Semantics

The analyzer groups one factor at a time. For the current example sweep, this yields summaries such as:

- topology = complete;
- topology = cycle;
- topology = small_world_k4_p01;
- topology = erdos_renyi_p02;
- threshold = 0_55;
- threshold = 0_7;
- initial_a = 0_45;
- initial_a = 0_55;
- seed = 1.

It does not compute multi-factor interaction tables in the first version. Those can be added later after the single-factor report proves useful.

## Report Structure

`report.md` should be deterministic and compact:

```markdown
# Sweep Analysis: network_topology_sweep

## Overview

- Runs: 48
- Completed: 48
- Failed: 0

## Topline

- Highest consensus rate: topology=complete ...
- Highest polarization: topology=cycle ...
- Highest edge disagreement: topology=...

## Factor Summaries

### topology

| value | runs | completed | failed | consensus_rate | mean_final_a_fraction | mean_polarization_index | mean_edge_disagreement_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| complete | ... |

## Failures

No failed runs.
```

The topline section should use simple rankings based on computed metrics. It should not claim causality.

Topline rankings should ignore a factor named `seed` when at least one non-seed factor has metric data. Seed remains in the full factor summary table, but it should not be presented as a substantive treatment effect.

## Components

### `sweep_analysis.py`

Owns parsing existing sweep artifacts and computing grouped summaries.

Public API:

```python
def analyze_sweep(output_dir: str | Path) -> SweepAnalysisResult:
    ...
```

Responsibilities:

- load and validate artifact files;
- derive factor names from `sweep_config.json`;
- compute grouped metric summaries;
- compute topline rankings;
- preserve deterministic ordering.

### `sweep_analysis_artifacts.py`

Owns writing analyzer outputs.

Responsibilities:

- write `report.md`;
- write `group_summary.csv`;
- write `group_summary.json`;
- write `failure_summary.csv`;
- create the `analysis/` directory.

### CLI

Extend `src/society_simulation/cli.py` with:

```bash
society-sim analyze <sweep_output_dir>
```

The existing `run` and `sweep` commands must keep their current output.

## Data Flow

```text
sweep output directory
  -> load sweep_config.json, summary.csv, summary.json, manifest.jsonl
  -> validate consistency
  -> compute per-factor grouped summaries
  -> compute simple topline rankings
  -> write analysis artifacts
  -> print concise CLI summary
```

## Validation and Error Handling

The analyzer should produce clean CLI errors for:

- missing sweep output directory;
- missing required artifact file;
- malformed JSON or CSV;
- `summary.csv` missing required columns;
- factor names in `sweep_config.json` missing from `summary.csv`;
- inconsistent run counts between `summary.csv` and `summary.json`.

The analyzer should not fail just because some metric cells are empty. Empty metric cells should be treated as missing values.

## Determinism

All outputs should be stable for the same input artifacts:

- factors appear in sweep config order;
- labels appear in first-seen `summary.csv` order within each factor;
- JSON artifacts use `sort_keys=True`;
- CSV headers are fixed;
- Markdown tables use fixed column order;
- numeric formatting is consistent.

## Testing

Add focused tests for:

- loading a representative sweep output fixture assembled in `tmp_path`;
- computing grouped consensus and mean metrics;
- ignoring failed and incomplete rows for metric means;
- preserving factor and label ordering;
- writing all analysis artifacts;
- Markdown report structure;
- CLI success output;
- CLI clean errors for missing or malformed artifacts;
- preserving existing `run` and `sweep` CLI behavior.

Use small temporary fixtures in tests rather than checking in generated `runs/` directories.

## Acceptance Criteria

After implementation:

```bash
python -m society_simulation sweep experiments/network_topology_sweep.json
python -m society_simulation analyze runs/sweeps/network_topology_sweep
```

should create:

```text
runs/sweeps/network_topology_sweep/analysis/report.md
runs/sweeps/network_topology_sweep/analysis/group_summary.csv
runs/sweeps/network_topology_sweep/analysis/group_summary.json
runs/sweeps/network_topology_sweep/analysis/failure_summary.csv
```

The full test suite should pass.

## Future Extensions

After this analyzer exists, the next likely features are:

- multi-factor interaction summaries;
- simple plots;
- agent heterogeneity sweeps;
- information shock sweeps;
- LLM agent comparison reports;
- backtesting against real-world polling or event timelines.
