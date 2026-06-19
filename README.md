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

The mock LLM path is deterministic and does not call any API. It records estimated prompt tokens, completion tokens, calls, and cost accounting in `metrics.json`; with the included mock config, API cost is `0`. LLM-backed network runs also write `llm_decisions.jsonl` with one row per agent decision, including the prompt, raw response, parsed action, parsed belief, token counts, call-level cost, and latency.

OpenAI-compatible LLM network herding:

```bash
export SOCIETY_SIM_LLM_API_KEY="..."
python -m society_simulation run examples/network_herding_openai_compatible.json
```

This path calls a real `/chat/completions` compatible API only when you run a config with `update_policy.type` set to `llm` and the configured `api_key_env` is present. Edit `base_url`, `model`, `token_limit_parameter`, and the per-1M-token price fields for your provider. The example uses 3 agents and 1 round with a small `max_estimated_cost_usd` cap so the first paid smoke test stays tiny.

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
- `llm_decisions.jsonl` when `update_policy.type` is `mock_llm` or `llm`
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
