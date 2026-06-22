# Public-Private Shock Pilot with GPT-5.4 Mini

## Abstract

We ran a small paid GPT-5.4 mini pilot to test whether an event-grounded LLM
society can produce a public-opinion pattern more interesting than simple
herding: a fairness shock may damage perceived legitimacy, and later corrections
or policy concessions may repair factual beliefs without fully repairing the
fairness concern.

The run completed 56 LLM decisions across 8 agents and 7 simulated days. The
estimated total cost was `$0.10472250`, below the configured `$0.30` cap. The
main result is not a strong public-silence effect. Agents posted too often after
day 2. The more interesting result is a legitimacy-residue pattern: average
policy stance recovered after a fact-check and hardship exemptions, but fairness
concern stayed very high through the final day.

This is not yet a publishable human-behavior result. It is a useful pilot because
it identifies one promising mechanism and one clear simulator limitation.

## Research Question

Can a sequence of official announcements, personal hardship stories, rumors,
fact-checks, and policy concessions produce a divergence between:

- private stance;
- public stance;
- perceived majority opinion;
- willingness to speak;
- fairness concern;
- trust in official information?

The intended phenomenon was public discourse diverging from private belief. The
observed phenomenon was more specific: factual repair moved stance, but fairness
concern persisted.

## Method

The scenario was a fictional city debate over downtown congestion pricing.
Agents represented eight residents with different interests: a nurse, taxi
driver, planning analyst, restaurant owner, retired teacher, school counselor,
retail worker, and library manager.

The event sequence was:

1. Official congestion-pricing announcement.
2. Worker and family hardship story.
3. County health and traffic benefit report.
4. Viral fairness rumor about insider exemptions.
5. Newspaper fact-check correcting the rumor.
6. City hardship exemptions and transit credits.
7. Neighborhood prompt before a council hearing.

The decision schema extended the earlier event runner with:

- `willingness_to_speak`;
- `perceived_majority`;
- `fairness_concern`;
- `trust_in_official_info`;
- `silence_reason`.

Memory retrieval was enabled with a limit of three retrieved memories per
decision.

## Cost and Reliability

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

The time series shows the main pattern:

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

The fact-check and concession moved private stance from -0.1925 on day 4 to
0.0475 on day 7. Official trust also recovered from 0.4200 to 0.5925. Fairness
concern did not recover; it stayed around 0.86 to 0.87 after the rumor and
concession.

## Interpretation

The useful observation is not that agents copied neighbors. They did not merely
follow a majority. Instead, they repeatedly separated factual correction from
legitimacy repair. The rumor became less credible after the fact-check, but the
agents retained the broader concern that the policy had not fully handled
night-shift workers, small suppliers, and transparent revenue use.

This suggests a candidate mechanism for future experiments:

> Corrections can repair factual beliefs while leaving the fairness frame active.

That mechanism is closer to a social-opinion contribution than a generic memory
ablation. It connects to real public-opinion problems: a policy can become more
acceptable after concessions while still carrying a legitimacy deficit.

The perceived-majority metric also showed a smaller but interesting lag. By the
final day, the actual mean private stance was slightly positive at 0.0475, while
agents perceived the majority as slightly negative at -0.0388. In this pilot,
agents saw the social climate as more skeptical than the aggregate private state.

## Negative Result

The public-silence mechanism did not work yet. The final silent-agent rate was
0.0, and agents posted 52 messages out of a possible 56 daily decisions. The
prompt allowed zero messages, but the model still treated daily posting as the
normal action once conversation began.

That matters. A realistic crowd-psychology simulator needs silence, hesitation,
and non-response as first-class actions. The current prompt is too deliberative
and too cooperative: agents explain themselves every day like thoughtful forum
participants, not like ordinary people who often read, withhold, repeat, or avoid
conflict.

## Memory Diagnostics

The memory layer produced 472 memories and 56 retrieval rows, with an average of
2.5 retrieved memories per decision. Retrieved memory kinds were:

| kind | count |
|---|---:|
| self_message | 74 |
| self_reasoning | 58 |
| event_exposure | 8 |

Retrieved memories often repeated earlier fairness concerns. This probably
helped preserve the legitimacy-residue pattern, but it also made messages
repetitive. The next version should distinguish stale self-repetition from
socially meaningful memory.

## Limitations

This is a single small pilot with no human benchmark. The result should be read
as simulator behavior, not a claim about real public opinion.

The message policy is under-calibrated. It allows silence but does not make
silence behaviorally likely enough.

The agents share one public channel, which makes the conversation unusually
visible and orderly. Real public opinion has partial attention, private
side-conversations, fatigue, and asymmetric participation.

## Next Experiment

Follow-up implementation added a speech-decision layer before message
generation. Each agent now chooses one of four speech actions:

1. `public_post`;
2. `private_message`;
3. `read_only`;
4. `avoid_discussion`.

The runner validates that generated messages match the selected action, and the
metrics now report speech-action counts and rates. This means the next paid run
can directly test whether the legitimacy-residue pattern persists when many
agents stop publicly posting.

The next paid run should use the same public-private shock scenario with the new
speech-decision contract and a fresh cost cap.

## Artifacts

- Config: `experiments/public_private_shock_gpt54_mini_pilot.json`
- Run directory: `runs/public_private_shock_gpt54_mini_pilot_20260622`
- Metrics: `runs/public_private_shock_gpt54_mini_pilot_20260622/metrics.json`
- Decisions: `runs/public_private_shock_gpt54_mini_pilot_20260622/llm_decisions.jsonl`
- Agent states: `runs/public_private_shock_gpt54_mini_pilot_20260622/agent_states.jsonl`
- Messages: `runs/public_private_shock_gpt54_mini_pilot_20260622/messages.jsonl`
