# Society Simulation

Society Simulation is a no-LLM baseline for sequential information cascades and network herding dynamics.

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
