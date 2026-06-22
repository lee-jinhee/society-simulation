# GPT-5.4 Mini Speech-Decision Public-Private Shock Pilot

## 초록

event-driven LLM society에서 speech-decision layer가 agent behavior를 어떻게
바꾸는지 보기 위해 paid GPT-5.4 mini pilot을 실행했다. 이전 public-private shock
pilot은 agent가 message를 생략할 수 있게 했지만, model은 daily posting을 기본 행동처럼
다루었다. 이번 pilot에서는 speech choice를 명시적으로 만들었다. message를 생성하기
전에 각 agent는 `public_post`, `private_message`, `read_only`, `avoid_discussion`
중 하나를 먼저 선택해야 했다.

run은 8 agents와 7 simulated days에 걸쳐 56 LLM decisions를 완료했다. estimated total
cost는 `$0.07967550`였고, configured `$0.30` cap보다 낮았다. 주요 결과는 private-message
pattern이 아니라 강한 public-abstention pattern이다. agent들은 salience와 fairness
concern이 상승했음에도 day 5까지 모든 decision에서 `read_only`로 남았다. public posting은
day 6 policy concession과 day 7 public-hearing prompt 이후에만 나타났다. final day에는
3명이 public poster였고 5명은 read-only agent였다.

이것은 simple herding보다 의미 있는 pilot이지만, 아직 human-validated behavioral result는
아니다. explicit speech actions가 non-participation을 visible하게 만들 수 있음을 보여주며,
다음 두 가지 engineering problem을 드러낸다. private side conversation은 아직 자연스럽게
trigger되지 않고, `silence_reason`에는 더 엄격한 validation이 필요하다.

## 연구 질문

explicit speech-decision contract가 public-opinion shock sequence 안에서 public
abstention, delayed participation, channel choice를 LLM social simulation에 표현하게 만들
수 있는가?

관심 현상은 agent들이 majority를 copy하는지 여부가 아니었다. 더 중요한 질문은 agent들이
private opinion을 유지하고, event로부터 update하면서도, social situation이 speech를 useful
하거나 legitimate하게 만들기 전까지 publicly participate하지 않을 수 있는가였다.

## 방법

scenario는 fictional city의 downtown congestion pricing debate였다. nurse, taxi driver,
planning analyst, restaurant owner, retired teacher, school counselor, retail worker,
library manager라는 여덟 resident persona를 사용했다.

event sequence는 다음과 같다.

1. official congestion-pricing announcement;
2. worker and family hardship story;
3. health and traffic benefit report;
4. insider exemptions에 대한 viral fairness rumor;
5. rumor를 correcting하는 newspaper fact-check;
6. city hardship exemptions and transit credits;
7. council hearing 전 neighborhood prompt.

Memory retrieval은 decision당 최대 세 개의 retrieved memories로 enabled했다. runner는
speech contract를 강제했다.

| speech action | required message behavior |
|---|---|
| `public_post` | exactly one public message |
| `private_message` | exactly one private message |
| `read_only` | no message |
| `avoid_discussion` | no message |

이 run은 이전 public-private shock scenario와 같은 population, events, network를 사용했지만,
artifacts는 fresh output directory에 기록했다.

## 비용과 신뢰성

| metric | value |
|---|---:|
| LLM calls | 56 |
| prompt tokens | 50,272 |
| completion tokens | 9,327 |
| input cost | `$0.037704` |
| output cost | `$0.0419715` |
| total estimated cost | `$0.0796755` |
| configured cap | `$0.30` |
| parse failures | 0 |

run은 cap보다 훨씬 낮게 끝났다. observed cost는 이전 public-private pilot보다도 낮았는데,
이번 run이 더 적은 messages와 더 짧은 completions를 생성했기 때문이다.

## 집계 결과

| metric | final value |
|---|---:|
| final private stance mean | 0.0438 |
| final public stance mean | 0.0225 |
| final private-public gap | 0.1313 |
| final public expression bias | -0.0213 |
| final willingness to speak | 0.4375 |
| final silent-agent rate | 0.6250 |
| final perceived-majority error | 0.0163 |
| final fairness concern | 0.6588 |
| final trust in official information | 0.5425 |
| final speech-action counts | `{'public_post': 3, 'read_only': 5}` |

time series는 delayed public activation을 보여준다.

| day | event phase | speech actions | private mean | public mean | fairness concern | official trust | willingness |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | initial | 8 read-only | -0.0125 | 0.0187 | 0.3000 | 0.4625 | 0.5000 |
| 1 | official launch | 8 read-only | -0.0150 | -0.0025 | 0.4775 | 0.4600 | 0.4437 |
| 2 | hardship story | 8 read-only | -0.1100 | -0.0725 | 0.6550 | 0.4313 | 0.3562 |
| 3 | health report | 8 read-only | -0.0575 | -0.0462 | 0.6400 | 0.4700 | 0.3638 |
| 4 | fairness rumor | 8 read-only | -0.1800 | -0.1663 | 0.8300 | 0.3675 | 0.2362 |
| 5 | fact-check | 8 read-only | -0.0975 | -0.1000 | 0.7475 | 0.4838 | 0.3075 |
| 6 | concession | 2 public, 6 read-only | -0.0050 | -0.0275 | 0.6675 | 0.5450 | 0.3912 |
| 7 | hearing prompt | 3 public, 5 read-only | 0.0438 | 0.0225 | 0.6588 | 0.5425 | 0.4375 |

## Message-Level Observations

전체 run에서 public messages는 다섯 개뿐이었다. day 6에 두 개, day 7에 세 개가 생성되었다.
private messages는 생성되지 않았다.

public posts는 simple majority copying이 아니었다. 그것들은 conditional acceptance frame
주변에 모였다. hardship exemptions와 transit credits는 도움이 되었지만, fairness와
verification은 여전히 중요하다는 frame이다. speaker는 day 6의 Mei와 Owen, day 7의 Minho,
Mei, Owen이었다. posting한 agent들은 가장 일관되게 pro-policy인 agent들이 아니었다. 그들은
unresolved fairness 또는 implementation concern을 가진 상태에서 qualified way로 말할 수
있게 된 agent들이었다.

이것이 central qualitative finding이다. concession은 단순히 모두를 persuade하지 않았다.
그것은 partial support와 continued skepticism을 공개적으로 말할 수 있는 vocabulary를 만들었다.

## 해석

speech-decision layer는 simulator의 behavioral surface를 바꾸었다. 이전 public-private pilot에서
agent들은 거의 매일 post했고, public channel은 비현실적으로 바빴다. 이번 run에서 agent들은
대부분 읽고 privately update했다. public discussion은 policy system이 concrete concessions를
제공하고 group이 누가 말할 것인지 explicit하게 물은 뒤에야 시작되었다.

이 pattern은 simple herding보다 crowd-psychology question에 더 가깝다.

> 사람들은 publicly expressive해지기 전에 highly attentive해질 수 있다.

run은 더 약한 legitimacy-residue pattern도 유지했다. private stance는 rumor day의 -0.1800에서
final day의 0.0438로 회복했고, trust도 0.3675에서 0.5425로 회복했다. fairness concern은 rumor
peak인 0.8300에서 내려왔지만 final day에도 0.6588로 높게 남았다. 이는 initial 0.3000의 두 배를
넘는 값이다.

즉 factual repair와 policy concessions는 stance를 움직였지만, fairness concern은 salient
residue로 남았다. 발전시킬 가치가 있는 부분은 여기다. simulator는 crowd가 policy를 support하는지
뿐 아니라 apparent persuasion 이후에도 어떤 frame이 discussion을 계속 organize하는지 추적해야 한다.

## Negative Results

private messaging은 나타나지 않았다. model은 silence에는 `read_only`를, late participation에는
`public_post`를 사용했지만, `private_message`나 `avoid_discussion`은 선택하지 않았다. 이는 현재
prompt와 config의 channel affordance가 약하기 때문일 가능성이 높다. agent들은 relationship을 갖고
있지만, scenario는 private contact가 natural next move가 되는 상황을 아직 만들지 못한다.

`silence_reason` field도 under-specified되어 있다. 몇몇 `read_only` states가 silence reason으로
여전히 `not_silent`를 반환했다. runner는 message behavior를 올바르게 강제했기 때문에 invalid
messages는 기록되지 않았지만, future runs에는 state-level consistency check가 필요하다. silent
speech actions에는 actual reason을 요구하고, public/private speech에는 `not_silent`를 사용해야 한다.

## 한계

이것은 single small pilot이며 human benchmark, seed sweep, external validation이 없다. 결과는 real
public opinion에 대한 claim이 아니라 simulator behavior로 읽어야 한다.

prompt는 여전히 model에게 structured decision을 내리고 있다고 말한다. instrumentation에는 유용하지만,
social context에서 speech behavior가 먼저 emerge하고 이후 parsed되는 conversation-first simulation보다
덜 자연스럽다.

run에는 public group channel이 하나뿐이다. 실제 social life에는 fragmented attention, private
backchannels, weak ties, direct messages, avoidance, offline constraints가 있다. 그것들은 아직 충분히
강하게 표현되지 않았다.

## 다음 실험

다음 실험은 private speech를 schema상 available하게만 두지 말고 socially available하게 만들어야 한다.
유용한 design은 explicit private-channel affordances와 one-to-one contact가 자연스러운 event trigger를
추가하는 것이다.

1. 한 agent의 occupation 또는 neighborhood를 직접 implicate하는 rumor;
2. issue를 privately clarify할 수 있는 trusted tie;
3. public posting을 socially risky하게 만드는 public conflict cost;
4. coherent `silence_reason` values를 요구하는 validation.

target phenomenon은 public statements, private coordination, quiet observation의 three-way split이어야 한다.
그것은 이번 pilot의 public-abstention pattern보다 강한 crowd-psychology result가 될 것이다.

## Artifacts

- Config: `experiments/public_private_shock_speech_decision_gpt54_mini_pilot.json`
- Run directory: `runs/public_private_shock_speech_decision_gpt54_mini_pilot_20260622`
- Metrics: `runs/public_private_shock_speech_decision_gpt54_mini_pilot_20260622/metrics.json`
- Decisions: `runs/public_private_shock_speech_decision_gpt54_mini_pilot_20260622/llm_decisions.jsonl`
- Agent states: `runs/public_private_shock_speech_decision_gpt54_mini_pilot_20260622/agent_states.jsonl`
- Messages: `runs/public_private_shock_speech_decision_gpt54_mini_pilot_20260622/messages.jsonl`
