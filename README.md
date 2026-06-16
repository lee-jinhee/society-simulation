# Society Simulation

Network Herding v0 is a no-LLM simulation of sequential information cascades.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Run

```bash
python -m society_simulation run examples/sequential_cascade.json
```

Artifacts are written to `output_dir` in the config:

- `config.json`
- `steps.jsonl`
- `metrics.json`
- `summary.txt`

## Test

```bash
pytest -v
```
