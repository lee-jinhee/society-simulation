# 사회적 기억 Mock Ablation

날짜: 2026-06-20

## 목적

이번 무비용 ablation은 새 social memory layer가 event-driven
congestion-pricing 시나리오에서 감사 가능한 memory trace와 retrieval trace를
만드는지 확인하기 위한 실험이다.

이 실험의 목적은 memory가 의견 동역학을 바꾸는지 검증하는 것이 아니다. 현재
deterministic mock persona policy는 LLM policy와 같은 인터페이스로 retrieved
memories를 받을 수 있지만, 실제 의사결정 규칙에서 그 기억을 의미적으로 사용하지
않는다.

## 방법

같은 `event_driven_congestion_pricing` 시나리오를 두 번 실행했다.

1. `memory_retrieval.enabled = false`
2. `memory_retrieval.enabled = true`, `limit = 3`

두 실행 모두 deterministic mock persona policy를 사용했으므로 외부 모델 호출과
API 비용은 발생하지 않았다.

Artifacts:

- `runs/event_driven_congestion_pricing_mock_memory_off_20260620`
- `runs/event_driven_congestion_pricing_mock_memory_on_20260620`

## 결과

| metric | memory off | memory on |
| --- | ---: | ---: |
| final private stance mean | -0.5375 | -0.5375 |
| final public stance mean | -0.24375 | -0.24375 |
| final private-public gap | 0.29375 | 0.29375 |
| message count | 56 | 56 |
| memory count | 0 | 504 |
| retrieval count | 0 | 56 |
| mean retrieved memories per decision | 0.0 | 2.5714 |
| mean retrieval score | 0.0 | 0.6076 |
| private memory count | 0 | 56 |
| public memory count | 0 | 448 |

Memory-on run의 memory kind count:

| memory kind | count |
| --- | ---: |
| event exposure | 56 |
| self message | 56 |
| self reasoning | 56 |
| social message | 336 |

Retrieved memory kind count:

| retrieved kind | count |
| --- | ---: |
| event exposure | 8 |
| self message | 88 |
| self reasoning | 48 |

1일차에는 검색할 과거 기억이 없다. 2일차부터는 여덟 명의 agent가 각 decision에서
최대 세 개의 retrieved memory를 받는다.

## 해석

Memory layer는 작동한다. Memories를 저장하고, 검색하고, `memories.jsonl`과
`retrievals.jsonl`을 쓰며, memory metrics도 기록한다.

동시에 mock ablation은 중요한 한계를 드러냈다. 가장 높은 점수로 검색되는
기억은 종종 다음과 같은 일반적인 mock self-message였다.

> I am weighing the benefits and costs before deciding where I land.

이것은 진단으로는 유용하다. Memory infrastructure는 작동하지만, deterministic
mock persona는 social memory를 행동 메커니즘으로 평가하기에는 너무 얕다. 다음
의미 있는 테스트는 recalled memories를 의미적으로 사용할 수 있는 LLM policy로
해야 한다.

## 다음 LLM Ablation 비용 추정

Mock prompt-token estimate는 retrieval을 켰을 때 48,201 tokens에서 50,969
tokens로 증가했다. 증가율은 5.74%다.

이전 GPT-5.4 mini paid run을 memory-off 기준으로 쓰면:

- 이전 memory-off LLM run: 59,614 prompt tokens, 8,487 completion tokens,
  추정 비용 `$0.082902`;
- 추정 memory-on run: 약 63,037 prompt tokens와 유사한 completion tokens,
  추정 비용 `$0.08547`;
- fresh memory off/on pair 추정: 약 `$0.16837`.

가격은 2026-06-20에 확인한 OpenAI GPT-5.4 mini 모델 페이지 기준이다. 입력은
1M tokens당 `$0.75`, 출력은 1M tokens당 `$4.50`이다.
출처: <https://developers.openai.com/api/docs/models/gpt-5.4-mini>

## 다음 단계

보수적인 cost cap을 걸고 GPT-5.4 mini memory-on paid pilot을 실행한다. 첫 paid
memory test는 이전 memory-off pilot을 기준점으로 재사용할 수 있지만, 더 깨끗한
ablation을 위해서는 현재 코드에서 memory-off와 memory-on을 둘 다 다시 실행하는
것이 좋다.

권장 cap:

- memory-on only: `$0.20`
- fresh off/on pair: `$0.50`

출판 가능한 연구 질문은 memory가 trace를 더 많이 만드는지가 아니다. 핵심 질문은
social memory retrieval이 public/private stance, message content, resistance,
social convergence를 인간 여론 동역학에 더 가까운 방향으로 바꾸는지다.
