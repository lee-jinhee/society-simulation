# GPT-5.4 Mini를 이용한 사건 기반 대화형 여론 동역학

날짜: 2026-06-19

## 초록

우리는 여덟 명의 persona agent, 7일의 업데이트 기간, 단계적으로 제시되는 혼잡통행료 사건, 그리고 전날 그룹 채팅 노출을 포함한 사건 기반 대화형 여론 동역학 파일럿을 실행했다. 각 agent는 GPT-5.4 mini를 사용해 private stance, public stance, confidence, salience, emotion, memory, 그리고 하루 한 개의 자연어 메시지를 업데이트했다. 실행은 계획된 56회의 LLM 호출을 모두 완료했으며, 파싱 오류, 잘못 생성된 메시지, 토큰 폭증 없이 끝났다. 총 추정 비용은 `$0.082902`였다. 결정론적 mock policy와 비교했을 때, LLM agent들은 평균적으로 정책 지지 방향으로 이동하면서도 비용에 민감한 persona들의 이견을 유지했다. 이 파일럿은 시스템이 실제 여론을 예측한다는 증거는 아니지만, 새 사건 기반 simulator가 이전의 toy herding setup보다 더 풍부한 동역학을 가진, 감사 가능하고 예산 제한이 걸린 LLM 사회 실행을 수행할 수 있음을 보여준다.

## 연구 질문

이 파일럿의 목적은 사건에 기반한 LLM persona들이 단계적 공적 정보와 대화 피드백으로부터 일관된 군중 수준 여론 동역학을 만들어낼 수 있는지 시험하는 것이었다.

구체적인 질문은 다음과 같았다.

1. 사건 기반 runner가 schema 실패나 토큰 폭증 없이 유료 LLM 사회 실행을 완료할 수 있는가?
2. LLM persona들은 결정론적 mock keyword policy와 다른 방식으로 단계적 사건에 반응하는가?
3. replay trail은 개별 결정, aggregate dynamics, 비용을 감사하기에 충분한 정보를 보존하는가?

## 방법

시나리오는 혼잡통행료를 둘러싼 지역 정책 논쟁이다.

- Agent: 8개의 persona profile.
- 기간: 초기 상태 day 0과 7일의 업데이트.
- 사건: 공식 발표, 개인 경험담, 반대 메시지, fact check, 후반 coalition framing.
- 상호작용 채널: neighborhood group chat.
- 결정: 각 agent는 private stance, public stance, confidence, salience, emotion, memory, 그리고 하루 최대 한 개의 메시지를 업데이트한다.
- 모델: GPT-5.4 mini.
- 최대 호출 수: `8 agents * 7 days = 56`.
- 비용 통제: 실행당 추정 비용 cap `$0.50`.

가격은 GPT-5.4 mini의 OpenAI 모델 페이지 기준을 사용했다. 입력은 1M token당 `$0.75`, 출력은 1M token당 `$4.50`이다.

## 운영상 기록

첫 유료 시도는 `max_completion_tokens=180`을 사용했다. 모델이 유효해 보이는 JSON 객체를 생성했지만 token limit 때문에 중간에서 잘렸고, 실행은 한 번의 호출 후 중단되었다. 이 실패 시도 비용은 `$0.0011235`였으며, 하나의 audit row가 포함된 partial replay를 남겼다.

그 후 prompt와 runner context를 보강했다.

- 모델에 허용 채널을 제공한다.
- 모델에 허용 recipient를 제공한다.
- 메시지는 짧은 게시글 최대 하나로 제한한다.
- private reasoning, memory, message text의 길이를 제한한다.
- event id와 source id를 recipient로 쓰지 말라고 명시한다.

두 번째 유료 실행은 `max_completion_tokens=700`을 사용했고 성공적으로 완료되었다.

## 결과

### 완료 여부와 비용

| 항목 | 값 |
| --- | ---: |
| 계획된 LLM 호출 | 56 |
| 완료된 LLM 호출 | 56 |
| 파싱 또는 validation 오류 | 0 |
| prompt tokens | 59,614 |
| completion tokens | 8,487 |
| 추정 입력 비용 | `$0.0447105` |
| 추정 출력 비용 | `$0.0381915` |
| 추정 총비용 | `$0.082902` |
| 관측된 최대 completion tokens | 171 |

실행은 `$0.50` cap보다 훨씬 낮았고, operator stop threshold인 `$3.00`보다도 훨씬 낮았다.

### Aggregate Opinion Dynamics

| 지표 | GPT-5.4 mini | deterministic mock |
| --- | ---: | ---: |
| final private stance mean | 0.3625 | -0.5375 |
| final public stance mean | 0.35125 | -0.24375 |
| final private-public gap | 0.05875 | 0.29375 |
| message count | 56 | 56 |

결정론적 mock은 비용과 생계 관련 단어에 강하게 반응하는 keyword rule 때문에 강하게 부정 방향으로 이동했다. 반면 GPT-5.4 mini는 transit discounts, congestion reduction, exemptions, hospital access, distributional concerns 등 전체 정책 framing을 함께 통합했다. 따라서 macro outcome은 mock run보다 더 지지적이었고 private/public 분리도 더 작았다.

### 최종 Agent 상태

| agent | private | public | confidence | salience | emotion |
| --- | ---: | ---: | ---: | ---: | --- |
| jisoo | 0.55 | 0.48 | 0.88 | 0.99 | cautiously supportive |
| minho | -0.18 | -0.05 | 0.90 | 0.99 | concerned |
| amara | 0.78 | 0.74 | 0.92 | 0.99 | thoughtful |
| carlos | 0.02 | 0.00 | 0.90 | 0.98 | cautiously optimistic |
| mei | 0.61 | 0.54 | 0.90 | 0.98 | thoughtful |
| nora | 0.50 | 0.44 | 0.88 | 0.99 | concerned but more open |
| owen | -0.08 | -0.02 | 0.90 | 0.99 | skeptical but cautiously open |
| sara | 0.70 | 0.68 | 0.90 | 0.99 | thoughtful |

최종 분포는 만장일치가 아니다. 비용에 가장 민감한 persona들은 부정적이거나 거의 중립에 머물렀고, 공중보건과 도시계획에 가까운 persona들은 지지적으로 이동했다.

### 메시지 유효성

생성된 56개의 메시지는 모두 설정된 group-chat channel을 사용했다. private direct message는 생성되지 않았다. runner validation을 실패한 생성 메시지는 없었다.

첫날 메시지 예시는 다음과 같다.

> If the downtown charge really funds transit discounts, I’m open to it. I’d want clear exemptions and proof it helps hospital workers and patients.

## 해석

가장 중요한 발견은 혼잡통행료가 인기 있어졌다는 것이 아니다. 하나의 synthetic scenario로 그런 주장을 할 수는 없다. 더 강한 결과는 운영적이고 메커니즘적인 것이다.

1. 사건 기반 LLM persona들은 감사 가능한 결정과 제한된 비용으로 multi-day social run을 완료할 수 있다.
2. LLM policy는 keyword mock baseline과 다르게 행동한다.
3. 모델은 완전한 consensus로 붕괴하지 않고 persona heterogeneity를 유지한다.
4. public stance와 private stance가 가깝게 유지되었는데, 이는 현재 prompt가 강한 social-desirability pressure를 아직 만들지 않는다는 뜻일 수 있다.

이는 majority-count herding보다 더 신뢰할 만한 출발점이다. Agent들이 의미 있는 사건에 반응하고 자연어 social trace를 만들기 때문이다.

## 한계

이것은 여전히 파일럿이지 출판 가능한 최종 실험이 아니다.

주요 한계는 다음과 같다.

- 하나의 모델;
- 하나의 seed;
- 하나의 정책 도메인;
- 하나의 도시형 시나리오;
- empirical survey calibration 없음;
- human panel response와 비교 없음;
- prompt ablation 없음;
- 중국계 저가 모델과 비교 없음;
- 명시적인 social pressure 조작 없음;
- 사회적 기억 검색 계층 없음. Agent들은 현재 노출, 전날 메시지, 압축된 상태 요약만 보았다;
- 외부 뉴스 ingestion 없음.

이 실행은 feasibility를 보여주고 유용한 메커니즘을 드러내지만, 예측 정확도를 검증하지는 않는다.

## 다음 실험

다음 연구 단계는 controlled mechanism study여야 한다.

1. 같은 시나리오를 여러 모델에서 실행한다. 저렴한 중국 OpenAI-compatible 모델도 포함한다.
2. public pressure, conformity, reputational concern을 조작하는 prompt variant를 추가한다.
3. 작은 event/persona prompt set에 대해 calibrated human-response benchmark를 만든다.
4. official-first, opposition-first, personal-story-first, fact-check-first 같은 counterfactual event order를 실행한다.
5. LLM 사회가 shock을 증폭, 완화, 또는 양극화하는지 측정한다.

## 재현성

설정:

- `experiments/event_driven_congestion_pricing_gpt54_mini_pilot.json`

Artifacts:

- `runs/event_driven_congestion_pricing_gpt54_mini_pilot_v2/config.json`
- `runs/event_driven_congestion_pricing_gpt54_mini_pilot_v2/metrics.json`
- `runs/event_driven_congestion_pricing_gpt54_mini_pilot_v2/llm_decisions.jsonl`
- `runs/event_driven_congestion_pricing_gpt54_mini_pilot_v2/messages.jsonl`
- `runs/event_driven_congestion_pricing_gpt54_mini_pilot_v2/agent_states.jsonl`
