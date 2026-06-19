# Local Social Observation Is Sufficient to Elicit LLM-Mediated Herding in a Small Agent Society

Date: 2026-06-19

## Status

Registered pilot with attempted execution. `OPENAI_API_KEY` was loaded from `/home/jhlee/repo/.env` and mapped to `SOCIETY_SIM_LLM_API_KEY`, but OpenAI rejected the first request with `HTTP 429 insufficient_quota`. No successful LLM decision was produced, no token usage was recorded by the simulator, and the estimated experiment cost remained `$0.000000`.

## Abstract

This pilot asks whether a population of LLM-controlled agents, each observing only its network neighbors' previous actions and beliefs, converges to collective consensus under repeated synchronous interaction. We instantiate 30 agents on a small-world graph, run 12 update rounds, and repeat the experiment over 10 random seeds using `gpt-5.4-mini`. The simulator records every model decision, prompt, raw response, parsed action, token count, and estimated call-level cost. The central measurement is not whether the model is individually rational, but whether locally conditioned LLM responses produce macroscopic crowd dynamics such as consensus, polarization, or persistent disagreement. Execution was attempted, but the provider quota blocked the first request before any decision-level data could be collected.

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

Run this only after setting `SOCIETY_SIM_LLM_API_KEY`, or after loading `OPENAI_API_KEY` and mapping it to `SOCIETY_SIM_LLM_API_KEY`:

```bash
set -a
source /home/jhlee/repo/.env
set +a
export SOCIETY_SIM_LLM_API_KEY="$OPENAI_API_KEY"
./.venv/bin/python -u - <<'PY'
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

print(
    f"finished completed={sum(1 for r in records if r.status == 'completed')} "
    f"failed={sum(1 for r in records if r.status == 'failed')} "
    f"planned={len(planned_runs)} spent=${spent:.6f}"
)
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

### 5.1 Execution Attempt

The experiment was launched on 2026-06-19 with `OPENAI_API_KEY` loaded from `/home/jhlee/repo/.env`. The run stopped on the first seed before completing any LLM-backed agent decision.

```text
starting sweep=openai_mini_small_social_herding planned_runs=10 budget=$3.00
failed seed-1 error=ValueError: llm provider request failed with HTTP 429: insufficient_quota
finished completed=0 failed=1 planned=10 spent=$0.000000
```

### 5.2 Outcome

- Completed runs: 0.
- Failed runs: 1.
- Pending runs: 9.
- Successful LLM decisions: 0.
- Recorded prompt tokens: 0.
- Recorded completion tokens: 0.
- Estimated simulator cost: `$0.000000`.
- Stopping reason: provider quota rejection before the first completed response.

The resulting sweep artifacts were still written:

- `runs/sweeps/openai_mini_small_social_herding/sweep_config.json`
- `runs/sweeps/openai_mini_small_social_herding/manifest.jsonl`
- `runs/sweeps/openai_mini_small_social_herding/summary.csv`
- `runs/sweeps/openai_mini_small_social_herding/summary.json`

No per-run `llm_decisions.jsonl` artifact exists because the first LLM decision did not complete.

## 6. Discussion

This pilot is designed as a falsifiable systems check. The current attempt does not answer the scientific question because provider quota blocked inference before any agent decision completed. It does validate one operational property: the budgeted runner fails closed, writes sweep-level failure artifacts, and does not continue launching additional seeds after a provider-level error.

A publishable result would require successful provider access, stronger baselines, larger sweeps, prompt variants, model-family comparisons, and external validation against empirical social data. If the pilot completes within budget and produces stable decision records after quota is resolved, the next study should compare real LLM agents against mock-neighbor-majority, threshold, and DeGroot baselines across topology, initial polarization, and exogenous news shocks.

## 7. Reproducibility Artifacts

- Config: `experiments/openai_mini_small_social_herding.json`.
- Expected sweep output: `runs/sweeps/openai_mini_small_social_herding`.
- Per-run artifacts: `config.json`, `graph.json`, `steps.jsonl`, `timeseries.jsonl`, `metrics.json`, `llm_decisions.jsonl`, `summary.txt`.
- Sweep artifacts: `sweep_config.json`, `manifest.jsonl`, `summary.csv`, `summary.json`.
