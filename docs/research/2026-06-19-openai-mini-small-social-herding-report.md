# Local Social Observation Elicits Directional Herding in a Small LLM Agent Society

Date: 2026-06-19

## Abstract

We study whether a population of LLM-controlled agents can produce macro-level social dynamics from only local observations. Thirty agents are placed on a small-world network. At each synchronous update, every agent observes only its own current action and belief plus the previous actions and beliefs of its graph neighbors. The agent then queries `gpt-5.4-mini` for a binary action and belief probability. Across 10 random seeds and 3,600 audited LLM decisions, 8 runs reached unanimous A consensus and 2 runs ended with strong A majorities but residual B minorities. The mean final A fraction was 0.9633, mean edge disagreement was 0.0233, and mean consensus time among consensus runs was 6.25 rounds. Decision-level audits reveal a strong asymmetry: the model followed A-majority neighborhoods in 2,990 of 2,997 cases, but followed B-majority neighborhoods in only 242 of 322 cases. This pilot therefore demonstrates that local LLM interactions can generate robust aggregate herding, while also exposing a likely label- or prompt-induced directional bias that must be controlled before making broader claims about social prediction.

## 1. Research Question

The purpose of this experiment was not to test whether an individual LLM can answer a social-science prompt. The purpose was to test whether many locally situated LLM agents, interacting repeatedly through a network, can generate measurable crowd-level phenomena.

The central question was:

> If each agent sees only its neighbors' actions and beliefs, can repeated LLM-mediated local updates produce consensus, polarization, or persistent disagreement at the society level?

The experiment was designed to obtain three kinds of evidence:

1. **Macro dynamics:** whether the society converges, polarizes, or remains fragmented.
2. **Micro-to-macro auditability:** whether each aggregate outcome can be traced back to individual prompts, raw model responses, parsed actions, beliefs, costs, and latencies.
3. **Operational feasibility:** whether a real paid LLM sweep can run within a small budget without token explosion or schema instability.

## 2. Method

### 2.1 Environment

The experiment uses the `network_herding` simulator.

- Agents: 30.
- Network: small-world graph.
- Degree: 4.
- Rewiring probability: 0.1.
- Initial action assignment: Bernoulli with `probability_a = 0.5`.
- Scheduler: synchronous update rounds.
- Rounds per run: 12.
- Seeds: 10.
- Maximum LLM decisions: `30 agents * 12 rounds * 10 seeds = 3,600`.

### 2.2 Agent Policy

Each agent update is delegated to `gpt-5.4-mini`. The prompt contains:

- the agent id;
- the round index;
- the agent's current action;
- the agent's current belief probability;
- the ids, actions, and belief probabilities of observed neighbors.

The model is constrained to return compact JSON with:

- `action`: `A` or `B`;
- `belief_probability`: a number in `[0, 1]`.

The simulator validates and parses the response before updating the agent state.

### 2.3 Audit Trail

Every successful LLM decision writes one row to `llm_decisions.jsonl`. Each row includes the prompt, raw provider response, parsed action, parsed belief, token counts, estimated cost, latency, model id, agent id, and round index. This makes the experiment inspectable at both aggregate and individual-decision levels.

### 2.4 Budget Controls

The experiment used fixed low-output settings to prevent runaway cost:

- maximum completion tokens: 32;
- per-run estimated cost cap: `$0.30`;
- operator-level budget stop: `$3.00`;
- actual estimated cost after completion: `$0.631049`.

No token explosion occurred. Completion tokens per decision ranged from 13 to 22.

## 3. Results

### 3.1 Completion and Cost

All planned runs completed.

| quantity | value |
| --- | ---: |
| completed runs | 10 |
| failed runs | 0 |
| LLM decisions | 3,600 |
| prompt tokens | 511,200 |
| completion tokens | 55,033 |
| estimated cost | `$0.631049` |

### 3.2 Aggregate Outcomes

| metric | value |
| --- | ---: |
| consensus runs | 8 / 10 |
| mean final A fraction | 0.9633 |
| mean edge disagreement rate | 0.0233 |
| mean polarization index | 0.1291 |
| mean time to consensus among consensus runs | 6.25 rounds |

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

The two non-consensus runs are informative. Both began with A as a minority or weak minority, yet ended with A as a strong majority. The process is therefore not merely preserving the initial population split.

### 3.3 Round-Level Dynamics

Aggregated across all seeds, A share increased monotonically after the first LLM update.

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

### 3.4 Decision-Level Findings

The model's local response rule was asymmetric.

| neighborhood condition | cases | followed local majority | defied local majority |
| --- | ---: | ---: | ---: |
| A-majority neighborhood | 2,997 | 2,990 | 7 |
| B-majority neighborhood | 322 | 242 | 80 |

Tied-neighbor observations occurred 281 times. In 242 of those cases, the model preserved the agent's current action.

This asymmetry is the main pilot finding. Under the current prompt and action schema, the model is nearly deterministic when local social evidence favors A, but much more willing to defect from B-majority neighborhoods. This provides a plausible mechanism for the observed A drift.

### 3.5 Token and Latency Profile

| quantity | min | mean | median | max |
| --- | ---: | ---: | ---: | ---: |
| prompt tokens per decision | 114 | 142.00 | 142 | 170 |
| completion tokens per decision | 13 | 15.29 | 15 | 22 |
| cost per decision | `$0.000144` | `$0.000175` | `$0.000174` | `$0.0002115` |
| latency per decision | 453 ms | 940 ms | 764 ms | 11.86 s |

The cost profile is stable enough for larger pilot sweeps.

## 4. Interpretation

This experiment provides evidence that local LLM interactions can generate macro-level herding dynamics. The system does not need global information, a central planner, or explicit consensus instructions to move toward aggregate agreement.

However, the observed outcome is not yet evidence of a general social law. The action labels A and B are semantically empty. A directional preference for A may come from model priors, prompt wording, action ordering, JSON formatting conventions, or parser/schema artifacts. Therefore the most important result is not "LLM societies converge to A." The stronger and more defensible result is:

> In this audited setting, local LLM decision rules can create strong aggregate herding, and the audit trail reveals a label-asymmetric micro-mechanism that plausibly drives the macro outcome.

## 5. Limitations

This is a pilot study, not an ICLR-ready final result.

Major limitations:

- only one model family;
- only one topology class;
- only 10 seeds;
- no action-label randomization;
- no A/B label swap;
- no prompt ablation;
- no comparison against non-LLM baselines in the same report;
- no external empirical validation;
- no news shocks or time-varying media environment.

These limitations are not cosmetic. Without label randomization and baseline comparison, the observed A drift cannot be interpreted as a robust social-scientific phenomenon.

## 6. Next Experiments

The next experiment should not simply add more seeds. It should directly test the mechanism exposed by the audit trail.

Priority ablations:

1. Swap the order and semantic presentation of A and B.
2. Randomize action labels per run.
3. Compare against mock-neighbor-majority, threshold, and DeGroot policies.
4. Repeat the same setting with cheaper OpenAI-compatible models.
5. Add exogenous news shocks and measure whether local LLM agents amplify, dampen, or polarize around them.
6. Run topology sweeps over cycle, complete, small-world, and Erdős-Rényi graphs.

These experiments would turn the pilot into a defensible study of how local LLM cognition scales into crowd-level dynamics.

## 7. Reproducibility

The registered experiment configuration is:

- `experiments/openai_mini_small_social_herding.json`

The sweep artifacts are:

- `runs/sweeps/openai_mini_small_social_herding/sweep_config.json`
- `runs/sweeps/openai_mini_small_social_herding/manifest.jsonl`
- `runs/sweeps/openai_mini_small_social_herding/summary.csv`
- `runs/sweeps/openai_mini_small_social_herding/summary.json`
- `runs/sweeps/openai_mini_small_social_herding/analysis/report.md`
- per-seed replay and `llm_decisions.jsonl` files under `runs/sweeps/openai_mini_small_social_herding/runs/seed-*`
