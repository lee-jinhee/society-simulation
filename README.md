# Society Simulation

Society Simulation is a baseline for sequential information cascades and network herding dynamics, with an optional no-cost mock LLM policy for exercising LLM-shaped experiment workflows.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Python 3.11+ is required.

## Run

Sequential information cascade:

```bash
python -m society_simulation run examples/sequential_cascade.json
```

Network herding:

```bash
python -m society_simulation run examples/network_herding.json
```

Mock LLM network herding smoke test:

```bash
python -m society_simulation run examples/network_herding_mock_llm.json
```

The mock LLM path is deterministic and does not call any API. It records estimated prompt tokens, completion tokens, calls, and cost accounting in `metrics.json`; with the included mock config, API cost is `0`.

Network topology sweep:

```bash
python -m society_simulation sweep experiments/network_topology_sweep.json
python -m society_simulation analyze runs/sweeps/network_topology_sweep
```

Artifacts are written to `output_dir` in the config. Sequential cascade runs write:

- `config.json`
- `steps.jsonl`
- `metrics.json`
- `summary.txt`

Network herding runs write:

- `config.json`
- `graph.json`
- `steps.jsonl`
- `timeseries.jsonl`
- `metrics.json`
- `summary.txt`

Sweep runs write:

- `sweep_config.json`
- `manifest.jsonl`
- `summary.csv`
- `summary.json`
- `runs/<run_id>/...`

Sweep analysis writes:

- `analysis/report.md`
- `analysis/group_summary.csv`
- `analysis/group_summary.json`
- `analysis/failure_summary.csv`

## Test

```bash
pytest -v
```
