# Visible Endorsement Cascade Mock Pilot

## 요약

이 mock pilot은 Instagram-like simulator가 최소한의 endorsement cascade를
표현할 수 있는지 테스트한다. 같은 작성자, 같은 글, 같은 텍스트를 유지하고
visible like count만 바꿨을 때 downstream engagement가 달라지는지 보는
실험이다.

결과는 의도적으로 작지만 유용하다. deterministic mock policy에서 visible
endorsement는 likes에 명확한 단조 반응을 만든다.

| visible likes 조건 | mean action count | mean like count |
| --- | ---: | ---: |
| low, 초기 0 likes | 0.0000 | 0.0000 |
| moderate, 초기 40 likes | 9.0000 | 9.0000 |
| high, 초기 80 likes | 32.0000 | 32.0000 |

이것은 실제 인간 행동에 대한 증거가 아니다. 이것은 system-validity result다.
simulator가 현실적인 social signal을 agent에게 노출하고, agent policy가 그
signal에 반응하며, sweep analyzer가 그 반응을 aggregate할 수 있게 되었다.

## 실험

실험 정의:

- `experiments/instagram_visible_endorsement_sweep.json`

총 여섯 개의 mock simulation을 실행한다.

- 2 seeds: `20260622`, `20260623`
- 3 visible endorsement levels: `low`, `moderate`, `high`

각 run은 Instagram-like world의 초기 상태에 하나의 configured seed post를
주입한다.

> The new bus lane finally makes downtown feel easier to reach.

바뀌는 것은 seed post의 초기 `like_count`뿐이다.

- low: `0`
- moderate: `40`
- high: `80`

update policy는 deterministic `mock_social`이고
`response_style=endorsement_sensitive`다. policy는 실험 label을 받지 않는다.
agent는 ordinary feed-card metadata, 즉 visible likes, topic, text, author
handle을 본다.

## 구현 변경

이 pilot을 위해 simulator에 네 가지 변경을 했다.

1. `InstagramSocialDynamicsConfig`가 optional `seed_posts`를 지원한다.
2. seed generator가 configured seed posts를 initial world에 주입한다.
3. `FeedItem`이 visible card metadata를 가진다: like count, topic, text,
   author handle.
4. `MockSocialMediaPolicy`가 visible endorsement를 social signal로 사용한다.
   `endorsement_sensitive` mode에서는 algorithmic feed score만으로 like하지
   않고, visible endorsement가 persona-dependent threshold를 넘어야 한다.

또한 social-media metrics의 `action_count`를 `do_nothing` 제외로 바꿨다. 그래서
report는 모든 activation decision이 아니라 실제 platform action을 보여준다.

## 결과

Analyzer output:

- `runs/sweeps/instagram_visible_endorsement_sweep/analysis/report.md`

핵심 aggregate pattern:

- low visible likes는 downstream likes를 만들지 않았다.
- moderate visible likes는 작은 cascade를 만들었다.
- high visible likes는 큰 cascade를 만들었다.
- exposure diversity는 조금 변했지만 주요 조작 outcome은 아니다.
- graph rewiring은 일어나지 않았다. 현재 mock policy가 endorsement에 반응해서
  follow/unfollow를 하지는 않기 때문이다.

## 해석

이 실험은 이전 toy herding setup보다 낫다. agent에게 social-science label이나
neighbor vote table을 주지 않는다. agent는 평범한 platform affordance, 즉
visible likes가 있는 post card를 본다. 관찰된 aggregate behavior는 platform
state와 agent policy의 상호작용에서 나온다.

하지만 현재 결과는 인간 행동 finding으로 publishable하지 않다. policy는
deterministic이고 hand-authored다. 이 실험의 가치는 paid LLM pilot 전에 필요한
experimental plumbing을 검증했다는 데 있다.

- controlled social stimulus
- ordinary social-media presentation
- auditable action traces
- sweep-level aggregate comparison
- LLM call 전 no-cost debugging

## 다음 단계

다음 paid 또는 semi-paid pilot은 같은 visible-endorsement sweep을 small LLM
policy와 strict cost cap으로 실행해야 한다. 연구 질문은 다음이다.

> post content, author, topic, feed policy를 고정했을 때 LLM persona가 visible
> endorsement에 graded response를 보이는가?

pilot에서는 raw action reasons도 함께 봐야 한다. LLM agent가 visible likes,
source credibility, social proof, skepticism을 명시적으로 언급한다면, 이것은
platform-mediated crowd behavior를 연구하기 위한 더 그럴듯한 기반이 된다.
