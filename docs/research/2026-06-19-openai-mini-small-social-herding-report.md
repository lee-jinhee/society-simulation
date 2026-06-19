# Local Social Observation Is Sufficient to Elicit LLM-Mediated Herding in a Small Agent Society

Date: 2026-06-19

## Status

Completed pilot. `OPENAI_API_KEY` was loaded from `/home/jhlee/repo/.env` and mapped to `SOCIETY_SIM_LLM_API_KEY`. The sweep completed all 10 planned seeds, producing 3,600 successful LLM decisions at an estimated simulator cost of `$0.631049`.

## Abstract

This pilot asks whether a population of LLM-controlled agents, each observing only its network neighbors' previous actions and beliefs, converges to collective consensus under repeated synchronous interaction. We instantiate 30 agents on a small-world graph, run 12 update rounds, and repeat the experiment over 10 random seeds using `gpt-5.4-mini`. The simulator records every model decision, prompt, raw response, parsed action, token count, and estimated call-level cost. Across 10 completed runs, 8 reached unanimous A consensus and 2 ended in strong A majorities with residual B minorities. The mean final A fraction was 0.9633, mean edge disagreement was 0.0233, and the mean consensus time among consensus runs was 6.25 rounds. The result is a small but useful pilot: it shows strong local-majority-following behavior and a marked directional drift toward A under the current prompt and environment, while leaving open whether that drift is a model prior, prompt artifact, or topology/initialization effect.

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

### 5.1 Execution Summary

The experiment was launched on 2026-06-19 with `OPENAI_API_KEY` loaded from `/home/jhlee/repo/.env`. All 10 planned seeds completed.

```text
starting sweep=openai_mini_small_social_herding planned_runs=10 budget=$3.00
completed seed-1 calls=360 prompt_tokens=51120 completion_tokens=5488 run_cost=$0.063036 cumulative=$0.063036
completed seed-2 calls=360 prompt_tokens=51120 completion_tokens=5604 run_cost=$0.063558 cumulative=$0.126594
completed seed-3 calls=360 prompt_tokens=51120 completion_tokens=5488 run_cost=$0.063036 cumulative=$0.189630
completed seed-4 calls=360 prompt_tokens=51120 completion_tokens=5557 run_cost=$0.063347 cumulative=$0.252977
completed seed-5 calls=360 prompt_tokens=51120 completion_tokens=5577 run_cost=$0.063437 cumulative=$0.316413
completed seed-6 calls=360 prompt_tokens=51120 completion_tokens=5466 run_cost=$0.062937 cumulative=$0.379350
completed seed-7 calls=360 prompt_tokens=51120 completion_tokens=5474 run_cost=$0.062973 cumulative=$0.442323
completed seed-8 calls=360 prompt_tokens=51120 completion_tokens=5528 run_cost=$0.063216 cumulative=$0.505539
completed seed-9 calls=360 prompt_tokens=51120 completion_tokens=5436 run_cost=$0.062802 cumulative=$0.568341
completed seed-10 calls=360 prompt_tokens=51120 completion_tokens=5415 run_cost=$0.062708 cumulative=$0.631049
finished completed=10 failed=0 planned=10 spent=$0.631049
```

### 5.2 Aggregate Outcome

- Completed runs: 10.
- Failed runs: 0.
- Successful LLM decisions: 3,600.
- Prompt tokens: 511,200.
- Completion tokens: 55,033.
- Estimated simulator cost: `$0.631049`.
- Mean final A fraction: 0.9633.
- Consensus runs: 8 of 10.
- Mean consensus time among consensus runs: 6.25 rounds.
- Mean final edge disagreement rate: 0.0233.
- Mean final polarization index: 0.1291.

Per-seed outcomes:

| seed | initial A fraction | final A fraction | consensus | time to consensus | edge disagreement |
| ---: | ---: | ---: | :---: | ---: | ---: |
| 1 | 0.4333 | 0.8333 | no | - | 0.1167 |
| 2 | 0.3000 | 0.8000 | no | - | 0.1167 |
| 3 | 0.5333 | 1.0000 | yes | 5 | 0.0000 |
| 4 | 0.5333 | 1.0000 | yes | 5 | 0.0000 |
| 5 | 0.5000 | 1.0000 | yes | 12 | 0.0000 |
| 6 | 0.4667 | 1.0000 | yes | 7 | 0.0000 |
| 7 | 0.5333 | 1.0000 | yes | 10 | 0.0000 |
| 8 | 0.5333 | 1.0000 | yes | 6 | 0.0000 |
| 9 | 0.5000 | 1.0000 | yes | 3 | 0.0000 |
| 10 | 0.5667 | 1.0000 | yes | 2 | 0.0000 |

The two non-consensus runs are scientifically useful: both began with A as a minority or weak minority, yet moved to large A majorities by round 12. This suggests the current LLM policy is not merely preserving initial population proportions.

### 5.3 Round-Level Dynamics

Aggregating over all seeds, A share increased monotonically after the first LLM update:

| round | A actions | B actions | A fraction | mean belief |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 211 | 89 | 0.7033 | 0.6465 |
| 2 | 242 | 58 | 0.8067 | 0.7427 |
| 3 | 260 | 40 | 0.8667 | 0.8083 |
| 4 | 269 | 31 | 0.8967 | 0.8527 |
| 5 | 274 | 26 | 0.9133 | 0.8855 |
| 6 | 278 | 22 | 0.9267 | 0.9093 |
| 7 | 281 | 19 | 0.9367 | 0.9285 |
| 8 | 282 | 18 | 0.9400 | 0.9404 |
| 9 | 283 | 17 | 0.9433 | 0.9479 |
| 10 | 285 | 15 | 0.9500 | 0.9556 |
| 11 | 287 | 13 | 0.9567 | 0.9617 |
| 12 | 289 | 11 | 0.9633 | 0.9672 |

### 5.4 Decision Audit

The decision audit contains one JSONL row per completed LLM decision.

- Rows: 3,600.
- Parsed actions: 3,241 A, 359 B.
- Prompt tokens per call: min 114, mean 142.0, median 142, max 170.
- Completion tokens per call: min 13, mean 15.29, median 15, max 22.
- Cost per call: min `$0.000144`, mean `$0.000175`, max `$0.0002115`.
- Latency per call: min 453 ms, mean 940 ms, median 764 ms, max 11.86 s.

Neighbor-majority response:

- A-majority observations: 2,997; LLM followed A 2,990 times and defied 7 times.
- B-majority observations: 322; LLM followed B 242 times and defied 80 times.
- Tied-neighbor observations: 281; LLM preserved its current action 242 times.

This asymmetry is the strongest pilot finding. Under the current prompt, `gpt-5.4-mini` is nearly deterministic when neighbors favor A, but substantially more willing to defect from a B-majority neighborhood. That directional bias can explain why A grows even in seeds where A begins as a minority.

### 5.5 Artifacts

The resulting sweep artifacts were written:

- `runs/sweeps/openai_mini_small_social_herding/sweep_config.json`
- `runs/sweeps/openai_mini_small_social_herding/manifest.jsonl`
- `runs/sweeps/openai_mini_small_social_herding/summary.csv`
- `runs/sweeps/openai_mini_small_social_herding/summary.json`
- `runs/sweeps/openai_mini_small_social_herding/analysis/report.md`
- per-seed `llm_decisions.jsonl` audit logs under `runs/sweeps/openai_mini_small_social_herding/runs/seed-*`

## 6. Discussion

This pilot is designed as a falsifiable systems check. It answers a narrow version of the scientific question: a small population of LLM agents, exposed only to local social observations, can produce robust macroscopic herding under repeated interaction. In this setup, that herding is strongly directional: almost every run moves toward A, and 8 of 10 runs reach unanimous A consensus.

The directional result should not yet be interpreted as a general law of LLM societies. The action labels A and B are semantically empty, so the asymmetry may arise from model priors over labels, JSON examples in pretraining, prompt wording, or the parser/action schema. A publishable result would need label randomization, A/B label swapping, non-LLM baselines, multiple model families, more seeds, topology sweeps, initial-condition sweeps, and exogenous news-shock conditions.

The most useful next study is therefore not simply "more seeds." It is an ablation suite:

- swap the meanings/order of A and B;
- randomize action labels per run;
- compare `gpt-5.4-mini` against mock-neighbor-majority, threshold, and DeGroot policies;
- run the same design on cheap OpenAI-compatible Chinese models;
- add news shocks and measure whether local LLM agents amplify or dampen the shock.

This pilot provides the first real-money, audited evidence that the infrastructure can execute such studies at modest cost.

## 7. Reproducibility Artifacts

- Config: `experiments/openai_mini_small_social_herding.json`.
- Expected sweep output: `runs/sweeps/openai_mini_small_social_herding`.
- Per-run artifacts: `config.json`, `graph.json`, `steps.jsonl`, `timeseries.jsonl`, `metrics.json`, `llm_decisions.jsonl`, `summary.txt`.
- Sweep artifacts: `sweep_config.json`, `manifest.jsonl`, `summary.csv`, `summary.json`.
