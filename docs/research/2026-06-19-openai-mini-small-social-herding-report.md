# Local Social Observation Is Sufficient to Elicit LLM-Mediated Herding in a Small Agent Society

Date: 2026-06-19

## Status

Registered report draft. The paid OpenAI run has not executed because neither `SOCIETY_SIM_LLM_API_KEY` nor `OPENAI_API_KEY` is present in the current shell environment.

## Abstract

This pilot asks whether a population of LLM-controlled agents, each observing only its network neighbors' previous actions and beliefs, converges to collective consensus under repeated synchronous interaction. We instantiate 30 agents on a small-world graph, run 12 update rounds, and repeat the experiment over 10 random seeds using `gpt-5.4-mini`. The simulator records every model decision, prompt, raw response, parsed action, token count, and estimated call-level cost. The central measurement is not whether the model is individually rational, but whether locally conditioned LLM responses produce macroscopic crowd dynamics such as consensus, polarization, or persistent disagreement. Results are pending API-key availability.

## 1. Introduction

LLM agent societies are useful only if their macro-level behavior can be audited back to micro-level interactions. This pilot therefore couples a minimal network herding environment with a decision-level LLM audit trail. Each agent receives a compact textual observation of its own current action and belief plus the actions and beliefs of its graph neighbors. The model returns only JSON containing the next binary action and belief probability.

The experiment is intentionally small. Its goal is to validate whether real LLM calls behave stably enough for larger sweeps, not to claim population-level external validity. A useful pilot should answer three questions: whether the model follows the requested response schema, whether token and cost accounting remain bounded, and whether the resulting society exhibits measurable collective dynamics.

## 2. Experimental Design

### 2.1 Environment

- Simulator: `network_herding`.
- Agents: 30.
- Graph: small-world topology with degree 4 and rewiring probability 0.1.
- Initial opinion: Bernoulli action assignment with `probability_a = 0.5`.
- Scheduler: synchronous rounds.
- Rounds: 12.
- Seeds: 1 through 10.
- Maximum LLM decisions: `30 agents * 12 rounds * 10 seeds = 3,600`.

### 2.2 Agent Policy

Each update is delegated to an OpenAI-compatible chat completion policy:

- Provider: OpenAI-compatible endpoint.
- Base URL: `https://api.openai.com/v1`.
- Model: `gpt-5.4-mini`.
- Temperature: 0.0.
- Maximum completion tokens: 32.
- Expected response: compact JSON with `action` and `belief_probability`.

The policy records both aggregate usage in `metrics.json` and per-decision traces in `llm_decisions.jsonl`.

### 2.3 Budget Control

The configured prices are the OpenAI standard prices checked on 2026-06-19:

- Input: `$0.75 / 1M tokens`.
- Output: `$4.50 / 1M tokens`.

The expected call size is approximately 100 input tokens and 20 output tokens. This gives an expected total pilot cost near:

```text
3,600 * (100 * 0.75 + 20 * 4.50) / 1,000,000 = $0.594
```

Safety limits:

- `max_completion_tokens = 32` prevents long completions.
- `max_estimated_cost_usd = 0.3` stops a single seed-run if its cumulative cost becomes abnormal.
- The operator-level total budget is `$3.00`; runs must stop before launching another seed if cumulative estimated cost reaches or exceeds `$3.00`.

## 3. Execution Protocol

Run this only after setting `SOCIETY_SIM_LLM_API_KEY`:

```bash
export SOCIETY_SIM_LLM_API_KEY="..."
./.venv/bin/python - <<'PY'
from pathlib import Path

from society_simulation.runner import run_experiment
from society_simulation.sweep_artifacts import SweepRunRecord, write_sweep_artifacts
from society_simulation.sweep_config import build_experiment_config, expand_sweep, load_sweep_config

BUDGET_USD = 3.0
SWEEP_PATH = Path("experiments/openai_mini_small_social_herding.json")
sweep = load_sweep_config(SWEEP_PATH)
planned_runs = expand_sweep(sweep)
records = []
spent = 0.0

for planned_run in planned_runs:
    if spent >= BUDGET_USD:
        print(f"budget_stop spent=${spent:.6f} budget=${BUDGET_USD:.2f}")
        break

    config = build_experiment_config(planned_run.config)
    try:
        result = run_experiment(config)
    except Exception as exc:
        records.append(
            SweepRunRecord(
                run_id=planned_run.run_id,
                labels=planned_run.labels,
                experiment_name=str(planned_run.config["experiment_name"]),
                output_dir=planned_run.config["output_dir"],
                status="failed",
                error=str(exc),
                metrics={},
            )
        )
        write_sweep_artifacts(sweep, planned_runs, tuple(records))
        print(f"failed {planned_run.run_id} error={exc}")
        break

    usage = result.metrics.get("llm_usage", {})
    run_cost = float(usage.get("total_cost_usd", 0.0))
    spent += run_cost
    status = "completed"
    error = None
    if spent >= BUDGET_USD:
        status = "failed"
        error = f"budget exceeded after run: spent ${spent:.6f} >= ${BUDGET_USD:.2f}"

    records.append(
        SweepRunRecord(
            run_id=planned_run.run_id,
            labels=planned_run.labels,
            experiment_name=config.experiment_name,
            output_dir=result.output_dir,
            status=status,
            error=error,
            metrics=result.metrics,
        )
    )
    write_sweep_artifacts(sweep, planned_runs, tuple(records))
    print(
        f"completed {planned_run.run_id} "
        f"calls={usage.get('calls')} "
        f"prompt_tokens={usage.get('prompt_tokens')} "
        f"completion_tokens={usage.get('completion_tokens')} "
        f"run_cost=${run_cost:.6f} cumulative=${spent:.6f}"
    )

    if spent >= BUDGET_USD:
        print(f"budget_stop spent=${spent:.6f} budget=${BUDGET_USD:.2f}")
        break

print(f"finished completed={len(records)} planned={len(planned_runs)} spent=${spent:.6f}")
PY
```

## 4. Planned Analysis

Primary outcomes:

- `consensus_reached`: whether all agents converge to one action.
- `consensus_action`: whether consensus favors A or B.
- `time_to_consensus`: first round at which consensus appears.
- `final_a_fraction`: final population share choosing A.
- `polarization_index`: final belief mass near extremes.
- `edge_disagreement_rate`: fraction of graph edges connecting disagreeing actions.

Audit-level checks:

- JSON parse success rate.
- Distribution of `prompt_tokens` and `completion_tokens`.
- Per-call latency distribution.
- Relationship between neighbor-majority exposure and parsed action.
- Cases where the LLM preserves its current action against local majority pressure.

## 5. Results

Pending. No paid API calls were made in this session because the required API key was not available.

Preflight outcome:

```text
ValueError SOCIETY_SIM_LLM_API_KEY is required for llm provider
```

This failure occurs before any OpenAI request is constructed, so it does not incur API cost.

## 6. Discussion

This pilot is designed as a falsifiable systems check. A publishable result would require stronger baselines, larger sweeps, prompt variants, model-family comparisons, and external validation against empirical social data. The immediate scientific value is narrower: it tests whether a real LLM can be embedded as an auditable local decision rule without uncontrolled cost or schema instability.

If the pilot completes within budget and produces stable decision records, the next study should compare real LLM agents against mock-neighbor-majority, threshold, and DeGroot baselines across topology, initial polarization, and exogenous news shocks.

## 7. Reproducibility Artifacts

- Config: `experiments/openai_mini_small_social_herding.json`.
- Expected sweep output: `runs/sweeps/openai_mini_small_social_herding`.
- Per-run artifacts: `config.json`, `graph.json`, `steps.jsonl`, `timeseries.jsonl`, `metrics.json`, `llm_decisions.jsonl`, `summary.txt`.
- Sweep artifacts: `sweep_config.json`, `manifest.jsonl`, `summary.csv`, `summary.json`.
