# 합성 Instagram 유사 사회에서 지역 카페 광고 도달 실험

날짜: 2026-06-23

## 초록

Instagram 유사 소셜 시뮬레이터에 감사 가능한 광고 개입 계층을 구현하고
테스트했다. 48-run mock sweep에서 지역 카페 sponsored campaign은 no-ad 및
organic-post control보다 훨씬 높은 total ad reach와 engagement를 만들었다.
이 결과는 실제 Instagram 예측 타당성의 증거가 아니다. 이 결과는
시뮬레이터가 paid delivery, organic spillover, frequency cap, creative
variant, treatment/control 비교를 replay 가능한 platform state로 표현할 수
있다는 증거다.

## 주장 범위

이 실험은 synthetic, uncalibrated mock-policy 실험이다. Meta reach, CTR,
conversion, ROAS 또는 실제 인간 행동을 예측하지 않는다. 유효한 주장은 더
좁다. 구현이 controlled counterfactual ad intervention을 실행하고, 상대적
mechanism screening을 위한 auditable metric을 생성할 수 있다는 것이다.

## 설정

시나리오: `maple_3rd_opening`, 지역 카페 opening-weekend campaign.

규모:

- 40 synthetic users;
- 16 ticks;
- tick 9에서 광고 시작;
- 2 seeds;
- 48 mock-policy runs;
- API 비용 0 USD.

요인:

- `ad_condition`: `no_ad`, `organic_post`, `sponsored_ad`;
- `creative_id`: `discount_offer`, `social_proof_offer`;
- `targeting`: `broad`, `interest_targeted`;
- `feed_policy`: `chronological_following`, `engagement_ranked`;
- `seed`: `20260623`, `20260624`.

산출물:

- sweep summary: `runs/sweeps/instagram_local_coffee_ad_reach_sweep/summary.csv`;
- analysis report: `runs/sweeps/instagram_local_coffee_ad_reach_sweep/analysis/report.md`;
- 각 run replay에는 `ad_impressions.jsonl` 및 campaign-aware
  `feed_impressions.jsonl`이 포함된다.

## 기계적 타당성 확인

sweep은 48/48 runs를 실패 없이 완료했다.

- no-ad는 paid impression을 0개 생성했다;
- organic-post control은 paid impression을 0개 생성했지만 nonzero organic
  ad exposure를 만들었다;
- sponsored-ad treatment는 sponsored run마다 정확히 45개의 paid impression을
  생성했고 campaign budget에 의해 제한되었다;
- replay artifact는 paid impression과 organic feed exposure를 분리해서
  기록한다;
- analyzer output은 paid reach, total ad reach, relevant reach,
  ad-specific engagement를 분리한다.

## 결과

Condition-level means:

| condition | paid impressions | unique paid reach | organic ad impressions | total ad reach | relevant total reach | ad likes | advertiser follows | ad DMs | generated posts | spillover rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_ad | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| organic_post | 0.00 | 0.00 | 76.38 | 17.25 | 16.38 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| sponsored_ad | 45.00 | 33.50 | 83.06 | 36.25 | 35.00 | 13.63 | 6.50 | 3.25 | 2.06 | 1.85 |

Sponsored-only contrasts:

| factor | value | total ad reach | relevant reach | organic ad impressions | ad likes | follows | DMs | generated posts | spillover rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| creative | discount_offer | 35.75 | 34.50 | 66.13 | 6.50 | 7.75 | 3.50 | 1.88 | 1.47 |
| creative | social_proof_offer | 36.75 | 35.50 | 100.00 | 20.75 | 5.25 | 3.00 | 2.25 | 2.22 |
| targeting | broad | 36.50 | 35.00 | 82.50 | 13.50 | 6.38 | 3.25 | 2.00 | 1.83 |
| targeting | interest_targeted | 36.00 | 35.00 | 83.63 | 13.75 | 6.63 | 3.25 | 2.13 | 1.86 |
| feed policy | chronological_following | 34.50 | 33.50 | 43.25 | 17.13 | 6.63 | 3.25 | 3.88 | 0.96 |
| feed policy | engagement_ranked | 38.00 | 36.50 | 122.88 | 10.13 | 6.38 | 3.25 | 0.25 | 2.73 |

## 해석

sponsored condition은 기계적으로 의미가 있다. paid delivery를 추가하고,
더 많은 unique user에게 도달하며, organic-post control보다 더 많은 downstream
campaign engagement를 만든다. organic control도 유용하다. 많은 feed
impression을 만들 수 있지만, 그 impression은 더 낮은 unique reach와 약한
campaign engagement로 집중된다.

가장 강한 creative signal은 social proof였다. sponsored-only run에서
social-proof creative는 direct discount creative보다 더 많은 organic ad
impression, total reach, like를 만들었다. 이는 mock policy에서 high visible
endorsement와 neighbor-sharing language가 conformity-sensitive public
engagement를 높이도록 설계되어 있기 때문이다.

feed-policy contrast는 단순히 "ranking이 더 좋다"는 이야기보다 흥미롭다.
Engagement-ranked feed는 organic spillover와 total reach를 크게 늘렸지만,
sponsored-only subset에서 chronological feed는 더 많은 direct ad like와 더
많은 generated post를 만들었다. 이는 현재 mock policy가 여전히 양식화되어
있지만, 시뮬레이터가 distribution과 action type 사이의 tradeoff를 드러낼 수
있음을 시사한다.

Targeting은 큰 차이를 만들지 못했다. Broad와 interest-targeted sponsored
run은 거의 같았다. 이것은 광고에 대한 주장이 아니라 진단 결과다. 현재
synthetic population과 targeting topics가 너무 넓어서 sharp targeting
contrast를 만들지 못한다. 다음 버전은 더 뚜렷한 user segment, 더 희소한
budget, 덜 겹치는 interests가 필요하다.

## 한계

이 결과는 publishable behavioral evidence가 아니다. 정책은 deterministic
mock behavior이며 인간 또는 LLM 행동이 아니다. population은 synthetic이고
uncalibrated이다. delivery model에는 auction, pacing, conversion objective,
quality model, image creative, real platform feedback loop가 없다. creative
effect는 일부 mock policy에 내장되어 있으므로 empirical discovery가 아니라
mechanism test다.

## 결론

이 milestone은 이전 toy herding experiment를 넘어섰다는 점에서 유용하다.
광고는 이제 platform state로 표현되고, paid delivery는 audit 가능하며,
organic spillover는 paid impression과 분리되고, treatment/control 비교는
자동으로 생성될 수 있다.

다음 연구 단계는 또 다른 큰 mock sweep이 아니어야 한다. 이 ad-card
representation을 사용한 더 작은 LLM pilot이어야 하며, agent가 relevance,
social proof, repetition, distrust, privacy, local interest 같은 평범한
인간다운 이유를 제시하는지 확인하는 데 집중해야 한다.
