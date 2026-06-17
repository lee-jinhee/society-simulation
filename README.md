# Society Simulation

Network Herding v0 is a no-LLM simulation of sequential information cascades and network herding dynamics.

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

## Test

```bash
pytest -v
```
