# GPT-5.4 Mini Social Memory-On Pilot

날짜: 2026-06-20

## 초록

event-driven congestion-pricing 시나리오에서 social memory layer를 켠 paid
GPT-5.4 mini pilot을 실행했다. 계획된 56회의 LLM decision이 모두 완료됐고,
parse 또는 validation error는 없었다. 총 추정 비용은 `$0.08677875`였으며,
`$0.20` cap보다 낮았다. 이전 memory-off paid pilot과 비교하면, memory-on
agent들은 평균적으로 덜 찬성적인 최종 상태에 도달했고, 비용과 exemption 문제에
직접 노출된 persona들이 우려를 더 강하게 유지했다. 이것은 아직 clean causal
ablation은 아니다. memory-off 비교가 이전 paid pilot을 재사용하기 때문이다.
하지만 retrieved social memory가 LLM society trajectory를 실질적으로 바꿀 수
있다는 첫 증거다.

## 연구 질문

event-grounded LLM agent에게 retrieved social memory를 추가하면 multi-day policy
debate에서 public/private opinion dynamics가 바뀌는가?

좁은 pilot 질문은 다음과 같다.

1. Memory-on runner가 작은 cost cap 아래에서 paid LLM run을 완료할 수 있는가?
2. Retrieved memory가 이전 memory-off paid run과 비교해 aggregate stance를
   바꾸는가?
3. 어떤 종류의 memory가 검색되며, 그것이 행동적으로 유용해 보이는가?

## 방법

시나리오: 가상 도시의 congestion pricing 논쟁.

- Agents: 8 persona profiles.
- 기간: day 0 initial states + 7 update days.
- Model: GPT-5.4 mini.
- Calls: `8 agents * 7 days = 56`.
- Memory retrieval: enabled.
- Retrieval limit: decision당 최대 3 memories.
- Cost cap: `$0.20`.

가격은 2026-06-20에 확인한 OpenAI GPT-5.4 mini 모델 페이지 기준이다. 입력은
1M tokens당 `$0.75`, 출력은 1M tokens당 `$4.50`이다.
출처: <https://developers.openai.com/api/docs/models/gpt-5.4-mini>

## 운영 결과

| quantity | value |
| --- | ---: |
| planned LLM calls | 56 |
| completed LLM calls | 56 |
| parse or validation errors | 0 |
| prompt tokens | 65,125 |
| completion tokens | 8,430 |
| estimated input cost | `$0.04884375` |
| estimated output cost | `$0.037935` |
| estimated total cost | `$0.08677875` |
| max prompt tokens in one call | 1,317 |
| max completion tokens in one call | 166 |
| memory count | 504 |
| retrieval count | 56 |
| mean retrieved memories per decision | 2.5714 |
| mean retrieval score | 0.6220 |

실행은 `$0.20` cap 아래에 머물렀다.

## Memory-Off vs Memory-On

| metric | memory off paid pilot | memory on paid pilot |
| --- | ---: | ---: |
| final private stance mean | 0.3625 | 0.2000 |
| final public stance mean | 0.35125 | 0.2025 |
| final private-public gap | 0.05875 | 0.08250 |
| final private stance variance | 0.12637 | 0.15170 |
| final public stance variance | 0.09269 | 0.09964 |
| final mean confidence | 0.8975 | 0.84625 |
| final mean salience | 0.9875 | 0.97125 |
| message count | 56 | 56 |
| prompt tokens | 59,614 | 65,125 |
| completion tokens | 8,487 | 8,430 |
| estimated cost | `$0.082902` | `$0.08677875` |

Memory-on은 prompt tokens를 5,511개 늘렸고 비용은 약 `$0.00388` 증가했다.

## Final Agent States

| agent | memory off private | memory on private | memory off public | memory on public |
| --- | ---: | ---: | ---: | ---: |
| amara | 0.78 | 0.44 | 0.74 | 0.38 |
| carlos | 0.02 | -0.12 | 0.00 | -0.02 |
| jisoo | 0.55 | 0.22 | 0.48 | 0.16 |
| mei | 0.61 | 0.28 | 0.54 | 0.20 |
| minho | -0.18 | -0.48 | -0.05 | -0.34 |
| nora | 0.50 | 0.66 | 0.44 | 0.58 |
| owen | -0.08 | -0.12 | -0.02 | -0.02 |
| sara | 0.70 | 0.72 | 0.68 | 0.68 |

변화는 균일하지 않다. Nora와 Sara는 지지를 유지하거나 더 지지적으로 변했지만,
Jisoo, Amara, Mei, Carlos, Minho는 memory-off run보다 덜 지지적이었다. 특히
taxi-driver persona인 Minho는 memory-on 조건에서 훨씬 더 강하게 반대에 남았다.

## Retrieved Memory 진단

Retrieved memory kind counts:

| kind | count |
| --- | ---: |
| event exposure | 8 |
| self message | 78 |
| self reasoning | 58 |

가장 높은 점수로 검색된 memories는 더 이상 generic mock message가 아니었다.
Fact-check, unresolved exemption details, automatic discounts, workers, families,
revenue accountability 같은 구체적인 우려가 포함되어 있었다.

고득점 retrieved memory 예:

> The fact-check confirms the revenue formula and discounts, but exemption details are still unresolved.

## 해석

이번 run은 memory가 행동적으로 의미 있어 보인 첫 실행이다. Memory-on agent들은
마지막 fact-check에 단순히 더 설득된 것이 아니었다. Retrieval은 unresolved
concerns와 과거 self-commitments를 다시 불러왔다. 그 결과 memory-off paid
pilot보다 mean support는 낮아졌고, public/private gap은 약간 커졌으며,
confidence는 낮아지고 stance variance는 높아졌다.

메커니즘 관점에서 중요하다. Memory layer는 agent를 매번 새 prompt에 반응하는
존재가 아니라 path dependence를 가진 actor에 더 가깝게 만든다. 이것이 이
프로젝트의 연구 목표에 더 가깝다. 여론은 사람들이 이미 알아차린 것, 말한 것,
걱정한 것, 그리고 공개적으로 commit한 것에 의존해야 한다.

## 한계

이것은 아직 pilot이다.

주요 한계:

- 하나의 모델;
- 하나의 seed;
- 하나의 정책 도메인;
- memory-off 비교는 같은 commit에서 새로 rerun한 paired ablation이 아니라 이전
  paid pilot을 재사용했다;
- human benchmark 없음;
- prompt ablation 없음;
- alternative retrieval weights 없음;
- reflection generation 없음;
- 저가 중국계 모델 비교 없음.

## 다음 단계

현재 코드에서 clean paired ablation을 실행한다.

1. GPT-5.4 mini memory off.
2. GPT-5.4 mini memory on.
3. 같은 seed, 같은 scenario, 같은 model, 같은 cost cap.

예상 총비용은 약 `$0.17`이며, `$0.50` cap이면 충분하다. 그다음 연구 단계는 이
paired ablation을 작은 human panel 또는 다른 model family와 비교하는 것이다.

## 재현성

설정:

- `experiments/event_driven_congestion_pricing_gpt54_mini_memory_on_pilot.json`

Artifacts:

- `runs/event_driven_congestion_pricing_gpt54_mini_memory_on_20260620/config.json`
- `runs/event_driven_congestion_pricing_gpt54_mini_memory_on_20260620/metrics.json`
- `runs/event_driven_congestion_pricing_gpt54_mini_memory_on_20260620/llm_decisions.jsonl`
- `runs/event_driven_congestion_pricing_gpt54_mini_memory_on_20260620/memories.jsonl`
- `runs/event_driven_congestion_pricing_gpt54_mini_memory_on_20260620/retrievals.jsonl`
- `runs/event_driven_congestion_pricing_gpt54_mini_memory_on_20260620/messages.jsonl`
- `runs/event_driven_congestion_pricing_gpt54_mini_memory_on_20260620/agent_states.jsonl`
