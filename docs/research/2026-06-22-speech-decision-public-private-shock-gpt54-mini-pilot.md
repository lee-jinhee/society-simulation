# Speech-Decision Public-Private Shock Pilot with GPT-5.4 Mini

## Abstract

We ran a paid GPT-5.4 mini pilot to test whether a speech-decision layer changes
the behavior of an event-driven LLM society. The previous public-private shock
pilot allowed agents to omit messages, but the model treated daily posting as
the normal behavior. This pilot made speech choice explicit: before generating a
message, each agent had to select `public_post`, `private_message`, `read_only`,
or `avoid_discussion`.

The run completed 56 LLM decisions across 8 agents and 7 simulated days. The
estimated total cost was `$0.07967550`, below the configured `$0.30` cap. The
main result is a strong public-abstention pattern rather than a private-message
pattern. Agents remained `read_only` for every decision through day 5 despite
rising salience and fairness concern. Public posting appeared only after a
policy concession on day 6 and a public-hearing prompt on day 7. The final day
had 3 public posters and 5 read-only agents.

This is a more meaningful pilot than simple herding, but it is still not a
human-validated behavioral result. It shows that explicit speech actions can
make non-participation visible, and it exposes two next engineering problems:
private side conversations are not yet naturally triggered, and
`silence_reason` needs stricter validation.

## Research Question

Can an explicit speech-decision contract make an LLM social simulation represent
public abstention, delayed participation, and channel choice under a sequence of
public-opinion shocks?

The phenomenon of interest was not whether agents copy a majority. The more
important question was whether agents can hold private opinions, update them
from events, and still choose not to participate publicly until the social
situation makes speech feel useful or legitimate.

## Method

The scenario was a fictional city debate over downtown congestion pricing. It
used eight resident personas: a nurse, taxi driver, planning analyst, restaurant
owner, retired teacher, school counselor, retail worker, and library manager.

The event sequence was:

1. official congestion-pricing announcement;
2. worker and family hardship story;
3. health and traffic benefit report;
4. viral fairness rumor about insider exemptions;
5. newspaper fact-check correcting the rumor;
6. city hardship exemptions and transit credits;
7. neighborhood prompt before a council hearing.

Memory retrieval was enabled with a limit of three retrieved memories per
decision. The runner enforced the speech contract:

| speech action | required message behavior |
|---|---|
| `public_post` | exactly one public message |
| `private_message` | exactly one private message |
| `read_only` | no message |
| `avoid_discussion` | no message |

The run used the same population, events, and network as the earlier
public-private shock scenario, but wrote artifacts to a fresh output directory.

## Cost and Reliability

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

The run stayed far below the cap. The observed cost was also lower than the
previous public-private pilot because this run produced fewer messages and
shorter completions.

## Aggregate Results

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

The time series shows delayed public activation:

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

Only five public messages were produced in the whole run: two on day 6 and
three on day 7. No private messages were produced.

The public posts were not simple majority copying. They clustered around a
conditional acceptance frame: hardship exemptions and transit credits helped,
but fairness and verification still mattered. The speakers were Mei and Owen on
day 6, followed by Minho, Mei, and Owen on day 7. The agents who posted were not
the most uniformly pro-policy agents. They were agents with unresolved fairness
or implementation concerns who could now speak in a qualified way.

This is the central qualitative finding: the concession did not simply persuade
everyone. It created a public vocabulary for partial support and continued
skepticism.

## Interpretation

The speech-decision layer changed the simulator's behavioral surface. In the
previous public-private pilot, agents posted almost every day, which made the
public channel unrealistically busy. In this run, agents mostly read and updated
privately. Public discussion did not begin until the policy system offered
concrete concessions and the group explicitly asked who would speak.

That pattern is closer to a crowd-psychology question than to simple herding:

> People may become highly attentive before they become publicly expressive.

The run also preserved a weaker legitimacy-residue pattern. Private stance
recovered from -0.1800 on the rumor day to 0.0438 on the final day, and trust
recovered from 0.3675 to 0.5425. Fairness concern fell from its rumor peak of
0.8300 but remained high at 0.6588, more than double the initial 0.3000.

In other words, factual repair and policy concessions moved stance, but fairness
concern remained a salient residue. This is the part worth developing: a
simulator should track not just whether a crowd supports a policy, but what
frame continues to organize discussion after apparent persuasion.

## Negative Results

Private messaging did not emerge. The model used `read_only` for silence and
`public_post` for late participation, but never selected `private_message` or
`avoid_discussion`. This probably reflects weak channel affordances in the
current prompt and config. The agents have relationships, but the scenario does
not yet create situations where a private contact is the natural next move.

The `silence_reason` field is also under-specified. Several `read_only` states
still returned `not_silent` as the silence reason. The runner correctly enforced
message behavior, so no invalid messages were written, but future runs need a
state-level consistency check: silent speech actions should require an actual
reason, while public and private speech should use `not_silent`.

## Limitations

This is a single small pilot with no human benchmark, no seed sweep, and no
external validation. The results should be read as simulator behavior, not as a
claim about real public opinion.

The prompt still tells the model that it is making a structured decision. That
is useful for instrumentation but less natural than a conversation-first
simulation where speech behavior emerges from social context and then gets
parsed afterward.

The run has only one public group channel. Real social life has fragmented
attention, private backchannels, weak ties, direct messages, avoidance, and
offline constraints. Those are not yet represented strongly enough.

## Next Experiment

The next experiment should make private speech socially available rather than
merely schema-available. A useful design would add explicit private-channel
affordances and event triggers that make one-to-one contact natural:

1. a rumor that directly implicates one agent's occupation or neighborhood;
2. a trusted tie who can clarify the issue privately;
3. a public conflict cost that makes public posting socially risky;
4. validation requiring coherent `silence_reason` values.

The target phenomenon should be a three-way split: public statements, private
coordination, and quiet observation. That would be a stronger crowd-psychology
result than this pilot's public-abstention pattern.

## Artifacts

- Config: `experiments/public_private_shock_speech_decision_gpt54_mini_pilot.json`
- Run directory: `runs/public_private_shock_speech_decision_gpt54_mini_pilot_20260622`
- Metrics: `runs/public_private_shock_speech_decision_gpt54_mini_pilot_20260622/metrics.json`
- Decisions: `runs/public_private_shock_speech_decision_gpt54_mini_pilot_20260622/llm_decisions.jsonl`
- Agent states: `runs/public_private_shock_speech_decision_gpt54_mini_pilot_20260622/agent_states.jsonl`
- Messages: `runs/public_private_shock_speech_decision_gpt54_mini_pilot_20260622/messages.jsonl`
