# Speech Decision Layer Design

## Research Motivation

The public-private shock pilot produced a useful legitimacy-residue pattern, but
it failed to produce realistic silence. Agents posted in almost every decision
after the conversation started. This makes the social feed too orderly and too
deliberative.

The next implementation should make speaking a first-class action rather than a
side effect of every daily decision.

## Design

Add `speech_action` to `EventAgentState`. Allowed values:

- `public_post`: the agent posts one public group/channel message.
- `private_message`: the agent sends one private message to another agent.
- `read_only`: the agent follows the discussion but does not speak.
- `avoid_discussion`: the agent avoids or disengages from the conversation.

The LLM decision contract must include `speech_action`. The prompt should tell
the agent to choose the action before writing a message. The prompt must not
describe the run as an experiment.

## Consistency Rules

The runner should validate generated messages against `speech_action`:

- `public_post`: exactly one message, `recipient_agent_id` is `None`.
- `private_message`: exactly one message, `recipient_agent_id` is an agent id.
- `read_only`: zero messages.
- `avoid_discussion`: zero messages.

Existing message validation still checks sender, channel, recipient existence,
and day.

## Metrics

Add speech-action metrics:

- per-day `speech_action_counts`;
- final `final_speech_action_counts`;
- final `final_public_post_rate`;
- final `final_private_message_rate`;
- final `final_read_only_rate`;
- final `final_avoid_discussion_rate`.

Existing `silent_agent_rate` remains based on absence of public posts, so private
messages still count as public silence.

## Scope

This implementation should update schema, parser, prompt, mock policy, runner
validation, metrics, replay summaries, and tests. It should not run a new paid
LLM experiment yet. The next paid run should happen only after the prompt and
validation are in place and the user approves the cost.
