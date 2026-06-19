# 지역적 사회 관찰은 작은 LLM 에이전트 사회에서 방향성 있는 군집화를 유도한다

날짜: 2026-06-19

## 초록

우리는 LLM으로 제어되는 에이전트 집단이 지역적 관찰만으로 거시적 사회 동역학을 만들어낼 수 있는지 연구한다. 30개의 에이전트를 small-world 네트워크 위에 배치했다. 각 동기식 업데이트에서 모든 에이전트는 자신의 현재 행동과 믿음, 그리고 그래프 이웃들의 이전 행동과 믿음만 관찰한다. 그런 다음 에이전트는 `gpt-5.4-mini`에 이진 행동과 믿음 확률을 질의한다. 10개의 무작위 seed와 3,600개의 감사된 LLM 결정에서, 8개 run은 만장일치 A 합의에 도달했고 2개 run은 잔류 B 소수파가 있는 강한 A 다수로 끝났다. 평균 최종 A 비율은 0.9633, 평균 edge disagreement는 0.0233, 합의 run들의 평균 합의 시간은 6.25 round였다. 결정 단위 감사는 강한 비대칭성을 드러낸다. 모델은 A-majority 이웃 상황에서는 2,997건 중 2,990건에서 A를 따랐지만, B-majority 이웃 상황에서는 322건 중 242건에서만 B를 따랐다. 따라서 이 pilot은 지역적 LLM 상호작용이 강한 집단 수준의 군집화를 만들어낼 수 있음을 보여주는 동시에, 사회 예측에 대한 더 넓은 주장을 하기 전에 통제해야 할 가능성 높은 label 또는 prompt 유도 방향성 bias를 드러낸다.

## 1. 연구 질문

이 실험의 목적은 개별 LLM이 사회과학 prompt에 답할 수 있는지를 테스트하는 것이 아니었다. 목적은 네트워크를 통해 반복적으로 상호작용하는, 지역적으로 위치한 다수의 LLM 에이전트가 측정 가능한 군중 수준 현상을 만들어낼 수 있는지를 테스트하는 것이었다.

중심 질문은 다음과 같았다.

> 각 에이전트가 이웃의 행동과 믿음만 볼 때, 반복되는 LLM 매개 지역 업데이트가 사회 수준에서 합의, 양극화, 또는 지속적 불일치를 만들어낼 수 있는가?

이 실험은 세 종류의 증거를 얻도록 설계되었다.

1. **거시 동역학:** 사회가 수렴하는지, 양극화되는지, 또는 분절된 채 남는지.
2. **미시-거시 감사 가능성:** 각 aggregate outcome을 개별 prompt, 원시 모델 응답, 파싱된 행동, 믿음, 비용, latency까지 추적할 수 있는지.
3. **운영 가능성:** 실제 유료 LLM sweep이 작은 예산 안에서 token explosion이나 schema instability 없이 실행될 수 있는지.

## 2. 방법

### 2.1 환경

실험은 `network_herding` simulator를 사용한다.

- 에이전트: 30.
- 네트워크: small-world graph.
- Degree: 4.
- Rewiring probability: 0.1.
- 초기 행동 할당: `probability_a = 0.5`인 Bernoulli.
- Scheduler: 동기식 update rounds.
- Run당 rounds: 12.
- Seeds: 10.
- 최대 LLM decisions: `30 agents * 12 rounds * 10 seeds = 3,600`.

### 2.2 에이전트 정책

각 에이전트 업데이트는 `gpt-5.4-mini`에 위임된다. Prompt에는 다음이 포함된다.

- agent id;
- round index;
- agent의 현재 action;
- agent의 현재 belief probability;
- 관찰된 neighbors의 ids, actions, belief probabilities.

모델은 다음을 포함하는 compact JSON을 반환하도록 제한된다.

- `action`: `A` 또는 `B`;
- `belief_probability`: `[0, 1]` 범위의 숫자.

Simulator는 에이전트 상태를 업데이트하기 전에 응답을 검증하고 파싱한다.

### 2.3 감사 추적

성공한 모든 LLM decision은 `llm_decisions.jsonl`에 한 row를 쓴다. 각 row에는 prompt, 원시 provider response, 파싱된 action, 파싱된 belief, token counts, estimated cost, latency, model id, agent id, round index가 포함된다. 이로써 실험은 aggregate 수준과 개별 decision 수준 모두에서 inspect 가능하다.

### 2.4 예산 통제

실험은 runaway cost를 방지하기 위해 고정된 low-output 설정을 사용했다.

- maximum completion tokens: 32;
- per-run estimated cost cap: `$0.30`;
- operator-level budget stop: `$3.00`;
- 완료 후 실제 estimated cost: `$0.631049`.

Token explosion은 발생하지 않았다. Decision당 completion tokens는 13에서 22 사이였다.

## 3. 결과

### 3.1 완료 및 비용

계획된 모든 run이 완료되었다.

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

Seed별 outcomes:

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

두 non-consensus run은 정보가 있다. 둘 다 A가 minority 또는 약한 minority로 시작했지만, round 12에서는 A가 강한 majority가 되었다. 따라서 이 과정은 단순히 초기 population split을 보존하는 것이 아니다.

### 3.3 Round-Level Dynamics

모든 seed를 aggregate하면, 첫 LLM update 이후 A share는 단조롭게 증가했다.

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

모델의 지역 response rule은 비대칭적이었다.

| neighborhood condition | cases | followed local majority | defied local majority |
| --- | ---: | ---: | ---: |
| A-majority neighborhood | 2,997 | 2,990 | 7 |
| B-majority neighborhood | 322 | 242 | 80 |

Tied-neighbor observations는 281번 발생했다. 그중 242건에서 모델은 agent의 현재 action을 보존했다.

이 비대칭성이 이 pilot의 핵심 발견이다. 현재 prompt와 action schema 아래에서 모델은 지역 사회적 증거가 A를 지지할 때 거의 deterministic하지만, B-majority neighborhood에서는 훨씬 더 자주 이탈한다. 이는 관찰된 A drift의 그럴듯한 메커니즘을 제공한다.

### 3.5 Token and Latency Profile

| quantity | min | mean | median | max |
| --- | ---: | ---: | ---: | ---: |
| prompt tokens per decision | 114 | 142.00 | 142 | 170 |
| completion tokens per decision | 13 | 15.29 | 15 | 22 |
| cost per decision | `$0.000144` | `$0.000175` | `$0.000174` | `$0.0002115` |
| latency per decision | 453 ms | 940 ms | 764 ms | 11.86 s |

비용 profile은 더 큰 pilot sweeps를 실행하기에 충분히 안정적이다.

## 4. 해석

이 실험은 지역적 LLM 상호작용이 거시 수준의 herding dynamics를 만들어낼 수 있다는 증거를 제공한다. 이 시스템은 aggregate agreement를 향해 움직이기 위해 global information, central planner, 또는 explicit consensus instructions를 필요로 하지 않는다.

하지만 관찰된 outcome은 아직 일반적인 사회 법칙의 증거가 아니다. Action labels A와 B는 의미적으로 비어 있다. A에 대한 방향성 preference는 model priors, prompt wording, action ordering, JSON formatting conventions, 또는 parser/schema artifacts에서 비롯될 수 있다. 따라서 가장 중요한 결과는 "LLM societies converge to A"가 아니다. 더 강하고 방어 가능한 결과는 다음이다.

> 이 감사 가능한 설정에서, 지역적 LLM decision rules는 강한 aggregate herding을 만들어낼 수 있으며, audit trail은 macro outcome을 plausibly drive하는 label-asymmetric micro-mechanism을 드러낸다.

## 5. 한계

이것은 pilot study이지, ICLR-ready final result가 아니다.

주요 한계:

- 단 하나의 model family;
- 단 하나의 topology class;
- 10 seeds뿐임;
- action-label randomization 없음;
- A/B label swap 없음;
- prompt ablation 없음;
- 같은 report 안에서 non-LLM baselines와 비교 없음;
- 외부 empirical validation 없음;
- news shocks 또는 time-varying media environment 없음.

이 한계들은 cosmetic하지 않다. Label randomization과 baseline comparison 없이는 관찰된 A drift를 robust social-scientific phenomenon으로 해석할 수 없다.

## 6. 다음 실험

다음 실험은 단순히 seeds를 더 추가하는 것이 아니어야 한다. Audit trail이 드러낸 메커니즘을 직접 테스트해야 한다.

우선순위 ablations:

1. A와 B의 순서 및 semantic presentation을 바꾼다.
2. Run마다 action labels를 randomize한다.
3. Mock-neighbor-majority, threshold, DeGroot policies와 비교한다.
4. 같은 설정을 더 저렴한 OpenAI-compatible models로 반복한다.
5. Exogenous news shocks를 추가하고, 지역적 LLM agents가 그것을 amplify, dampen, 또는 polarize하는지 측정한다.
6. Cycle, complete, small-world, Erdős-Rényi graphs에 대해 topology sweeps를 실행한다.

이 실험들은 pilot을 지역적 LLM cognition이 crowd-level dynamics로 어떻게 확장되는지에 대한 방어 가능한 연구로 바꿀 것이다.

## 7. 재현성

등록된 실험 configuration은 다음이다.

- `experiments/openai_mini_small_social_herding.json`

Sweep artifacts는 다음이다.

- `runs/sweeps/openai_mini_small_social_herding/sweep_config.json`
- `runs/sweeps/openai_mini_small_social_herding/manifest.jsonl`
- `runs/sweeps/openai_mini_small_social_herding/summary.csv`
- `runs/sweeps/openai_mini_small_social_herding/summary.json`
- `runs/sweeps/openai_mini_small_social_herding/analysis/report.md`
- `runs/sweeps/openai_mini_small_social_herding/runs/seed-*` 아래의 seed별 replay 및 `llm_decisions.jsonl` 파일
