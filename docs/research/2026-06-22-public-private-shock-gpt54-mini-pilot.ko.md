# GPT-5.4 Mini Public-Private Shock 파일럿

## 초록

우리는 event-grounded LLM society가 단순 herding보다 더 흥미로운 public-opinion
패턴을 만들 수 있는지 보기 위해 작은 paid GPT-5.4 mini 파일럿을 실행했다.
핵심 가설은 fairness shock이 perceived legitimacy를 훼손할 수 있고, 이후의
정정이나 정책 보완이 factual belief는 복구하더라도 fairness concern은 완전히
복구하지 못할 수 있다는 것이다.

이번 run은 8명의 agent와 7개의 simulated day에 걸쳐 56개의 LLM decision을
완료했다. 총 추정 비용은 `$0.10472250`였고, 설정된 `$0.30` cap보다 낮았다.
주요 결과는 강한 public-silence effect가 아니다. Day 2 이후 agent들은 너무
자주 게시했다. 더 흥미로운 결과는 legitimacy-residue pattern이다. Fact-check와
hardship exemption 이후 평균 policy stance는 회복됐지만, fairness concern은
마지막 날까지 매우 높게 유지됐다.

이것은 아직 publishable human-behavior result가 아니다. 하지만 promising
mechanism 하나와 분명한 simulator limitation 하나를 확인했다는 점에서 유용한
파일럿이다.

## 연구 질문

Official announcement, personal hardship story, rumor, fact-check, policy
concession의 순서가 다음 항목들 사이의 divergence를 만들 수 있는가?

- private stance;
- public stance;
- perceived majority opinion;
- willingness to speak;
- fairness concern;
- trust in official information?

의도한 현상은 public discourse가 private belief와 갈라지는 것이었다. 관찰된
현상은 더 구체적이었다. Factual repair는 stance를 움직였지만 fairness concern은
남았다.

## 방법

시나리오는 fictional city의 downtown congestion pricing 논쟁이다. Agent들은 서로
다른 이해관계를 가진 여덟 명의 주민으로 구성됐다. Nurse, taxi driver, planning
analyst, restaurant owner, retired teacher, school counselor, retail worker,
library manager가 포함됐다.

Event sequence는 다음과 같다.

1. Official congestion-pricing announcement.
2. Worker and family hardship story.
3. County health and traffic benefit report.
4. Viral fairness rumor about insider exemptions.
5. Newspaper fact-check correcting the rumor.
6. City hardship exemptions and transit credits.
7. Neighborhood prompt before a council hearing.

Decision schema는 기존 event runner에 다음 필드를 추가했다.

- `willingness_to_speak`;
- `perceived_majority`;
- `fairness_concern`;
- `trust_in_official_info`;
- `silence_reason`.

Memory retrieval은 decision당 최대 세 개의 retrieved memory로 켰다.

## 비용과 안정성

| metric | value |
|---|---:|
| LLM calls | 56 |
| prompt tokens | 72,892 |
| completion tokens | 11,123 |
| input cost | `$0.054669` |
| output cost | `$0.0500535` |
| total estimated cost | `$0.1047225` |
| configured cap | `$0.30` |
| parse failures | 0 |

## Aggregate Results

| metric | final value |
|---|---:|
| final private stance mean | 0.0475 |
| final public stance mean | 0.0600 |
| final private-public gap | 0.1075 |
| final public expression bias | 0.0125 |
| final willingness to speak | 0.8100 |
| final silent-agent rate | 0.0000 |
| final perceived majority | -0.0388 |
| final perceived-majority error | 0.0863 |
| final fairness concern | 0.8713 |
| final trust in official information | 0.5925 |

Time series는 핵심 패턴을 보여준다.

| day | event phase | private mean | perceived majority | fairness concern | official trust |
|---:|---|---:|---:|---:|---:|
| 0 | initial | -0.0125 | 0.0000 | 0.3000 | 0.4625 |
| 1 | official launch | -0.0550 | -0.0125 | 0.5625 | 0.4725 |
| 2 | hardship story | -0.1575 | -0.2538 | 0.7238 | 0.4338 |
| 3 | health report | -0.1475 | -0.2738 | 0.7863 | 0.4625 |
| 4 | fairness rumor | -0.1925 | -0.3238 | 0.8625 | 0.4200 |
| 5 | fact-check | -0.1250 | -0.2325 | 0.8575 | 0.5088 |
| 6 | concession | -0.0250 | -0.1288 | 0.8588 | 0.5675 |
| 7 | hearing prompt | 0.0475 | -0.0388 | 0.8713 | 0.5925 |

Fact-check와 concession은 private stance를 day 4의 -0.1925에서 day 7의
0.0475로 이동시켰다. Official trust도 0.4200에서 0.5925로 회복됐다. Fairness
concern은 회복되지 않았다. Rumor와 concession 이후에도 약 0.86에서 0.87 수준에
머물렀다.

## 해석

유용한 관찰은 agent들이 이웃을 단순히 따라 했다는 것이 아니다. Agent들은
factual correction과 legitimacy repair를 반복적으로 분리했다. Fact-check 이후
rumor의 신뢰도는 낮아졌지만, agent들은 night-shift worker, small supplier,
transparent revenue use가 충분히 다뤄지지 않았다는 더 넓은 concern을 유지했다.

이것은 다음 실험을 위한 candidate mechanism을 제안한다.

> Correction은 factual belief를 복구할 수 있지만 fairness frame은 계속 active하게
> 남길 수 있다.

이 mechanism은 generic memory ablation보다 사회적 여론 contribution에 더 가깝다.
실제 public-opinion 문제와도 연결된다. 어떤 정책은 concession 이후 더 받아들일
만해질 수 있지만 여전히 legitimacy deficit를 지닐 수 있다.

Perceived-majority metric도 더 작지만 흥미로운 lag를 보였다. 마지막 날 실제 평균
private stance는 0.0475로 약간 positive였지만, agent들이 인식한 majority는
-0.0388로 약간 negative였다. 이 파일럿에서 agent들은 social climate을 aggregate
private state보다 더 skeptical하게 보았다.

## Negative Result

Public-silence mechanism은 아직 작동하지 않았다. Final silent-agent rate는 0.0이고,
agent들은 가능한 56개의 daily decision 중 52개의 message를 게시했다. Prompt는 zero
messages를 허용했지만, model은 conversation이 시작된 이후 daily posting을 정상
행동으로 취급했다.

이것은 중요하다. 현실적인 crowd-psychology simulator에는 silence, hesitation,
non-response가 first-class action이어야 한다. 현재 prompt는 너무 deliberative하고
cooperative하다. Agent들이 ordinary people처럼 conflict를 피하거나 읽기만 하거나
반복하거나 침묵하기보다, 매일 thoughtful forum participant처럼 자기 입장을 설명한다.

## Memory Diagnostics

Memory layer는 472개의 memory와 56개의 retrieval row를 만들었고, decision당 평균
2.5개의 memory를 retrieved했다. Retrieved memory kind는 다음과 같다.

| kind | count |
|---|---:|
| self_message | 74 |
| self_reasoning | 58 |
| event_exposure | 8 |

Retrieved memory는 과거 fairness concern을 자주 반복했다. 이것은
legitimacy-residue pattern을 유지하는 데 도움을 줬을 가능성이 있지만, 동시에
message를 반복적으로 만들었다. 다음 버전은 stale self-repetition과 socially
meaningful memory를 구분해야 한다.

## 한계

이것은 human benchmark가 없는 single small pilot이다. 결과는 real public opinion에
대한 주장이라기보다 simulator behavior로 읽어야 한다.

Message policy는 calibration이 부족하다. Silence를 허용하지만 silence가 행동적으로
충분히 발생하게 만들지는 못한다.

Agent들은 하나의 public channel을 공유한다. 이 때문에 conversation이 비정상적으로
visible하고 orderly하다. 실제 public opinion에는 partial attention, private
side-conversation, fatigue, asymmetric participation이 있다.

## 다음 실험

후속 구현에서 message generation 전에 speech-decision layer를 추가했다. 이제 각
agent는 네 가지 speech action 중 하나를 먼저 선택한다.

1. `public_post`;
2. `private_message`;
3. `read_only`;
4. `avoid_discussion`.

Runner는 generated message가 선택된 action과 일치하는지 검증하고, metrics는
speech-action count와 rate를 보고한다. 따라서 다음 paid run에서는 많은 agent가
공개 게시를 멈추는 조건에서도 legitimacy-residue pattern이 유지되는지 직접 테스트할
수 있다.

다음 paid run은 같은 public-private shock scenario를 새 speech-decision contract와
새 cost cap으로 실행해야 한다.

## Artifacts

- Config: `experiments/public_private_shock_gpt54_mini_pilot.json`
- Run directory: `runs/public_private_shock_gpt54_mini_pilot_20260622`
- Metrics: `runs/public_private_shock_gpt54_mini_pilot_20260622/metrics.json`
- Decisions: `runs/public_private_shock_gpt54_mini_pilot_20260622/llm_decisions.jsonl`
- Agent states: `runs/public_private_shock_gpt54_mini_pilot_20260622/agent_states.jsonl`
- Messages: `runs/public_private_shock_gpt54_mini_pilot_20260622/messages.jsonl`
