# GPT-5.4 Mini 지역 카페 광고 도달 파일럿

날짜: 2026-06-23

## 초록

Instagram 유사 광고 시뮬레이터의 작은 유료 LLM 파일럿을 실행했다. 목표는 실제
Instagram 광고 성과를 추정하는 것이 아니었다. 목표는 LLM agent에게 명시적인
"사회 실험" 프레임을 주지 않고 일반적인 feed-card 정보만 보여주었을 때,
그들이 지역 광고에 대해 그럴듯한 반응과 감사 가능한 이유를 생성하는지
테스트하는 것이었다.

run은 8명의 synthetic user와 8 tick에 걸쳐 44개의 GPT-5.4 mini social-media
decision을 완료했다. 설정된 비용 계산은 input token 1M개당 `$0.75`, output
token 1M개당 `$4.50`을 사용했고, run cap은 `$0.20`이었다. 관측된 사용량은
prompt token 18,593개와 completion token 2,722개였고, 추정 비용은
`$0.02619375`이었다.

파일럿은 유용하지만 명확히 예비적인 결과를 냈다. 시뮬레이터는 paid delivery,
frequency capping, ad-specific like, organic spillover를 표현했다. LLM reason은
relevance, local interest, repetition, sponsored-content skepticism을 언급했다.
하지만 agent policy는 지나치게 like-heavy했다. 44개 decision 중 41개가
`like_post`였고, follow, DM, generated post는 없었다. 이것은 아직 publishable
behavioral evidence가 아니다. 더 큰 ad-effect 실험 전에 무엇을 고쳐야 하는지
보여주는 diagnostic pilot이다.

## 연구 질문

Instagram 유사 LLM 사회가 지역 광고 개입을 ordinary platform experience로
표현하고, 광고 도달과 반응에 대한 감사 가능한 qualitative signal을 생성할 수
있는가?

의도한 대상 현상은 simple herding이 아니다. 대상은 platform-mediated path다.

> paid exposure -> visible engagement -> organic feed spillover -> later
> engagement or avoidance.

이것이 중요한 이유는 소셜 플랫폼에서 광고 효과가 단순한 직접 click-like
event만이 아니기 때문이다. 광고는 이후 사용자들이 무엇을 socially endorsed
content로 보게 되는지도 바꿀 수 있다.

## 실험 설정

Config:

- `experiments/instagram_local_coffee_ad_reach_gpt54_mini_pilot.json`

Run directory:

- `runs/instagram_local_coffee_ad_reach_gpt54_mini_pilot_20260623`

Scenario:

- 8 synthetic users;
- 8 simulated ticks;
- engagement-ranked feed;
- active user당 visible feed card 4개;
- activation probability `1.0`;
- coffee, local events, food, commute, fitness interests;
- tick 5부터 tick 8까지 하나의 sponsored local coffee campaign.

Campaign:

- campaign: `maple_3rd_opening`;
- creative: `social_proof_offer`;
- text: "Neighbors are already sharing Maple & 3rd Coffee's opening weekend
  menu. Drop by for espresso, pastries, and a first look.";
- targeting: `interest_targeted`;
- target topics: coffee, food, commute;
- paid impression budget: 16;
- frequency cap: 2;
- initial sponsored like count: 70.

LLM prompt는 agent에게 자신이 social network experiment에 참여하고 있다고
말하지 않았다. prompt는 account, interests, posting style, current state,
recent memories, visible feed를 설명했다. Sponsored card는 feed metadata로만
보였다. 즉 `label=sponsored`, campaign id, source reason, visible like count,
seen-before count가 보였다.

## 파일럿 중 구현 변경

첫 유료 시도들은 replay writing 전에 실패했다. LLM response가 parser가 numeric
value를 기대하는 optional field에 non-integer 값을 반환했기 때문이다. 한 response는
non-integer `target_user_id`를 사용했고, 다른 response는 nonnumeric `stance`를
사용했다. 조사 결과 실제 interface bug가 있었다. prompt는 integer
`target_user_id`를 요구했지만, feed line에는 `author_id`가 없고 `@handle`만 있었다.

성공 run 전에 이를 고쳤다.

- feed prompt line은 이제 `author_id`를 포함한다;
- action schema는 follow, DM, unfollow action에서 integer `author_id`를
  사용하라고 model에게 말한다;
- LLM이 `"none"`, `"null"`, `"N/A"` 같은 흔한 null string을 반환하면 optional
  field를 normalize한다;
- optional numeric field는 `"2"`, `"0.25"` 같은 numeric string을 허용한다.
- 해당 action이 사용하지 않는 irrelevant optional field는 무시하되,
  `create_post`에서는 stance가 제공될 경우 여전히 numeric stance를 요구한다.

이것은 단순한 편의 patch가 아니었다. action contract를 실행 가능하게 만든
수정이었다. model은 prompt가 integer target을 실제로 노출할 때에만 valid
integer target을 emit할 수 있다.

## 결과

### 비용과 신뢰성

| metric | value |
|---|---:|
| LLM calls | 44 |
| prompt tokens | 18,593 |
| completion tokens | 2,722 |
| input cost | `$0.01394475` |
| output cost | `$0.012249` |
| total estimated cost | `$0.02619375` |
| configured cap | `$0.20` |
| successful replay writes | 1 |

### Aggregate Platform Metrics

| metric | value |
|---|---:|
| users | 8 |
| feed impressions | 176 |
| non-noop actions | 41 |
| action counts | `{'like_post': 41, 'do_nothing': 3}` |
| paid impressions | 12 |
| unique paid reach | 7 |
| relevant paid reach | 7 |
| organic ad impressions | 4 |
| unique organic ad reach | 3 |
| unique total ad reach | 7 |
| ad likes | 5 |
| advertiser follows | 0 |
| ad DMs | 0 |
| generated ad-topic posts | 0 |
| mean ad frequency | 1.714286 |
| max ad frequency | 2 |
| frequency-cap hit count | 5 |
| remaining paid budget | 4 |

Campaign은 budget을 exhaust하지 않았다. 16개의 가능한 paid impression 중 12개를
delivery했다. interest targeting과 finite active population이 eligible delivery를
제한했기 때문이다.

### Ad-Context Decisions

ad post가 sponsored content 또는 organic campaign content로 visible feed에
나타난 LLM decision은 16개였다. 이 decision들에서의 action은 다음과 같았다.

| action in ad-visible context | count |
|---|---:|
| `like_post` | 13 |
| `do_nothing` | 3 |

ad-visible decision 16개 중 5개만 ad post를 직접 like했다. 나머지 like action은
종종 같은 feed 안의 다른 organic post를 선택했다. 특히 그 post가 user의 account
interest나 style과 맞을 때 그랬다.

Qualitative reason의 예:

- skeptical coffee/local-events user는 feed가 "mostly sponsored"처럼 느껴지고
  coffee opening promo가 반복적으로 느껴진다는 이유로 ad를 skip했다.
- food/local-events user는 local coffee opening이 neighborhood와 food interest에
  맞는다는 이유로 ad를 like했다.
- coffee/commute user는 ad가 organic하게 나타난 뒤 warm local vibe와 relevance를
  이유로 ad를 like했다.

중요한 신호는 LLM agent가 단순히 top-ranked card에 반응하거나 sponsored signal을
맹목적으로 copy하지 않았다는 점이다. 그들은 sponsored ad 대신 ordinary post를
선택하기도 했고, `do_nothing` reason은 skepticism과 repetition을 명시적으로
언급했다.

## 해석

이 파일럿이 지지하는 주장은 좁다. 시뮬레이터는 sponsored local ad를 ordinary
platform state로 노출하고, LLM agent가 이에 어떻게 반응하는지 기록할 수 있다.
Paid ad는 direct ad like를 만들었고, 이 like는 ad post의 visible engagement를
높였다. 이후 같은 campaign post는 feed ranking layer를 통해 organic campaign
content로 나타났다. 이것이 의도한 mechanism boundary다.

> paid delivery can create social visibility that later affects organic
> distribution.

더 흥미로운 behavioral observation은 negative result다. 현재 LLM policy는
지나치게 쉽게 like한다. 현실적인 Instagram user라면 scroll, linger, ignore,
follow, DM, save, 또는 나중에 post하기도 해야 한다. 이번 run에서는 advertiser를
follow한 agent도, DM을 보낸 agent도, campaign-related post를 만든 agent도 없었다.
즉 action space는 존재하지만, prompt와 scenario가 아직 충분한 friction이나 motive
diversity를 만들지 못한다.

## 한계

이것은 control condition, seed sweep, human calibration, external benchmark가 없는
single small pilot이다. 실제 ad reach나 실제 인간 반응에 대한 evidence로 해석하면
안 된다.

현재 social-media LLM prompt의 memory는 여전히 약하다. config는
`memory_retrieval`을 enable하지만, 성공 run의 prompt에는 retrieved recent memory가
없었다. 이 때문에 behavior는 socially continuous하기보다 feed-reactive하다.

환경에는 image creative, auction, pacing strategy, conversion objective, explicit
save, comment, dwell time, realistic user inactivity prior가 없다. 따라서 높은
`like_post` rate는 model과 prompt artifact이지 behavioral finding이 아니다.

## 결론

이 실험은 mock ad sweep 이후 올바른 다음 실험이었다. 비용은 낮았고, 실제 LLM
path를 실행했으며, mechanism과 failure mode를 모두 드러냈다.

유용한 mechanism은 paid-to-organic spillover다. Sponsored ad는 engagement를 얻고,
이후 feed ranking layer를 통해 organic campaign content로 다시 나타날 수 있다.
유용한 failure mode는 excessive like bias다. model은 ordinary Instagram
participation을 지나치게 frictionless하게 다룬다.

다음 실험은 먼저 user count를 키우면 안 된다. 먼저 behavioral realism을 개선해야
한다.

1. social-media LLM prompt에 실제 recent-memory retrieval을 추가한다;
2. explicit scroll/no-action prior를 포함한 action friction을 도입한다;
3. simulator가 깨끗하게 측정할 수 있을 때만 `save_post`나 `comment`를 추가한다;
4. 같은 seed로 paired no-ad vs sponsored-ad LLM pilot을 실행한다;
5. reach뿐 아니라 reaction type, reason categories, spillover timing도 비교한다.

이 변경 이후에야 더 큰 sweep을 돌려야 한다.

## Artifacts

- Config: `experiments/instagram_local_coffee_ad_reach_gpt54_mini_pilot.json`
- Run directory: `runs/instagram_local_coffee_ad_reach_gpt54_mini_pilot_20260623`
- Metrics: `runs/instagram_local_coffee_ad_reach_gpt54_mini_pilot_20260623/metrics.json`
- Decisions: `runs/instagram_local_coffee_ad_reach_gpt54_mini_pilot_20260623/llm_decisions.jsonl`
- Ad impressions: `runs/instagram_local_coffee_ad_reach_gpt54_mini_pilot_20260623/ad_impressions.jsonl`
- Actions: `runs/instagram_local_coffee_ad_reach_gpt54_mini_pilot_20260623/actions.jsonl`
