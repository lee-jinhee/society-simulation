# Prior-Art Report: Network Herding and LLM Society Simulation

Date: 2026-06-16

## Purpose

This report summarizes how adjacent research projects build LLM-based social simulations, especially for opinion dynamics, echo chambers, rumor spread, information cascades, and network herding. The current project direction is no longer "individual behavioral-economics benchmark first." The working target is:

> Build a modular AI society simulator for studying how individual signals, social networks, information exposure, and interaction rules produce aggregate crowd behavior such as herding, cascades, polarization, rumor spread, and eventually public-opinion movement.

The important design decision is structural. Agent count, model vendor, and run size should remain configuration. The hard-to-change choices are:

- whether graph topology is explicit or improvised by LLMs;
- whether belief/action state is structured or only natural-language memory;
- whether interaction scheduling is controlled or free-form;
- whether validation compares against known social-learning/opinion-dynamics baselines;
- whether every run is replayable with prompts, model calls, graph state, and metrics.

## Sources Reviewed

Primary user-provided sources:

- Stanford CS222: AI Agents and Simulations
  - https://joonspk-research.github.io/cs222-fall24/index.html
- Joon Sung Park site and dissertation listing
  - https://www.joonsungpark.com/
  - https://purl.stanford.edu/jm164ch6237
- Park et al. 2023, "Generative Agents: Interactive Simulacra of Human Behavior"
  - https://arxiv.org/abs/2304.03442
- Park et al. 2024/2026, "LLM Agents Grounded in Self-Reports Enable General-Purpose Simulation of Individuals"
  - https://arxiv.org/abs/2411.10109
- Piao et al. 2025, "AgentSociety"
  - https://arxiv.org/abs/2502.08691
  - https://arxiv.org/html/2502.08691v1
  - https://github.com/tsinghua-fib-lab/agentsociety
- Artificial Societies, YC company profile
  - https://www.ycombinator.com/companies/artificial-societies

Additional network/opinion-dynamics sources:

- Chuang et al. 2024, "Simulating Opinion Dynamics with Networks of LLM-based Agents"
  - https://aclanthology.org/2024.findings-naacl.211/
  - https://github.com/yunshiuan/llm-agent-opinion-dynamics
- Chang et al. 2025, "LLMs generate structurally realistic social networks but overestimate political homophily"
  - https://arxiv.org/abs/2408.16629
  - https://github.com/snap-stanford/llm-social-network
- He and Jiang 2025, "Steering the Herd: A Framework for LLM-based Control of Social Learning"
  - https://arxiv.org/html/2504.02648v4
- Wang et al. 2025, "Decoding Echo Chambers: LLM-Powered Simulations Revealing Polarization in Social Networks"
  - https://arxiv.org/abs/2409.19338
  - https://aclanthology.org/2025.coling-main.264/
- Hu et al. 2025, "Simulating Rumor Spreading in Social Networks using LLM Agents"
  - https://arxiv.org/abs/2502.01450
  - https://github.com/UT-SysML/rumors-in-multi-agent

Repository snapshots checked on 2026-06-16:

- `tsinghua-fib-lab/agentsociety`: `8cb830819da839028c1b2085e1475ebe26f13ee2`
- `yunshiuan/llm-agent-opinion-dynamics`: `edfd7c91c366307ee06344477dcc8082eb720215`
- `snap-stanford/llm-social-network`: `a841a29c40e0488418f8384c4f6f48c6088bd6c8`

## Executive Takeaways

1. The serious projects do not let LLMs freely invent society. They make the network, intervention, observation scope, and metrics explicit.
2. LLMs are usually used for language generation, role-played judgment, belief explanation, or structured decision output, not as the whole simulator.
3. Graph topology is first-class. Random, small-world, scale-free, homophilic, Facebook-like, public/private, and local/LLM-generated networks are all experimental variables.
4. Belief state is usually discretized or otherwise structured. Natural-language memory is valuable, but aggregate science needs numeric or categorical state that can be plotted and tested.
5. Every credible setup has controls: no-interaction baselines, classical ABM baselines, intervention/no-intervention comparisons, or human-data validation.
6. For our project, the right v0 is a network-first herding simulator with pluggable update policies. It should support non-LLM, LLM, and hybrid agents behind the same experiment interface.

## CS222 and Park: Believability Is Not Accuracy

CS222 frames society simulation as a way to ask counterfactual "what if" questions about complex social systems. The course explicitly separates individuals, groups, populations, cognitive architectures, interactive worlds, and validation. This matters because a believable agent world is not automatically a valid forecasting tool.

Park's 2023 Generative Agents paper gives the classic architecture: memory stream, reflection, planning, and natural-language interaction in a small town of 25 agents. The result is rich and socially coherent. But the validation target is believability: can agents remember, plan, react, reflect, and produce plausible emergent social behavior?

Park's later self-report grounded agent work moves toward accuracy. It builds 1,052 person-specific agents from interviews, surveys, or both, then compares held-out simulated responses against real participants' responses. The key lesson is that agent grounding matters. Demographic-only agents are a baseline, not the end state.

Implication for us:

- Use Park's memory/reflection/planning as inspiration for agent internals, not as the first system boundary.
- For crowd psychology, the first-class object should be the interaction system: graph, exposure, signals, and update rules.
- If we later target public-opinion prediction, we will need grounded agents or real survey calibration. Synthetic personas alone are not enough for strong prediction claims.

## AgentSociety: Large Simulator, But Controlled Experiments

AgentSociety looks like a full LLM society platform, but the experiments are more structured than the headline suggests.

### How They Structure Agents

The paper separates profile, status, mental processes, and social behavior:

- profile: static identity and demographic attributes;
- status: dynamic needs, emotions, finances, relationships, and other changing variables;
- mental process: emotion, needs, cognition;
- behavior: mobility, social interaction, employment, consumption.

The repo's AgentSociety 2 README describes a modern LLM-native platform with modular environment tools, multiple reasoning routers, and SQLite-based experiment replay. In the v1 examples, experiments commonly configure Qwen as the LLM provider, enable SQLite storage, create a fixed number of citizen agents, run for a set number of days, and save selected contexts before and after the run.

### How Their Polarization Experiment Works

The `examples/polarization/control.py` experiment:

- creates 100 citizen agents from `profiles/profiles.json`;
- saves initial `attitude`;
- runs the simulation for 3 days;
- saves final `attitude`;
- saves `chat_histories`.

The treatment scripts add explicit persuasive agents:

- `AgreeAgent`: always supports stronger gun control;
- `DisagreeAgent`: always opposes stronger gun control.

These agents generate persuasive messages with LLM calls and send them to their friends. Message propagation is not unlimited: messages carry a `propagation_count`, and the custom chat handler stops responding after the count exceeds 5. This is a very important design pattern: LLMs generate persuasive text, but diffusion control is explicit code.

### How Their Rumor/Inflammatory Message Setup Works

The `examples/rumor_spreader/network_generator.py` file generates public and private networks using Barabasi-Albert graphs with separate seeds. The `utils.py` initializer:

- assigns private friends and public friends from graph neighbors;
- assigns relationship types: family, colleague, friend;
- samples relationship strengths by relationship type;
- initializes per-friend chat histories and interaction logs.

The inflammatory-message examples seed selected agents' chat histories with a message and then run the simulation. The paper compares control, inflammatory-message treatment, node intervention, and edge intervention. Again, the intervention is not left to emergent chat. The researcher decides where the information is injected and how moderation/removal is applied.

### What We Should Copy

- Separate static profile from dynamic state.
- Represent networks explicitly, including tie type and tie strength.
- Treat interventions as first-class experiment objects.
- Save initial state, final state, chat histories, memories, and run metadata.
- Support OpenAI-compatible providers, including cheaper non-OpenAI models.

### What We Should Not Copy Yet

- Do not adopt the full urban/mobility/economy stack for v0.
- Do not make unconstrained multi-day life simulation the default.
- Do not start with persuasive free-form chat as the only belief update mechanism.

AgentSociety's most useful lesson for us is not "build a city." It is: even a city-scale LLM simulator should keep graph, state, interventions, and replay explicit.

## Chuang et al.: LLM Opinion Dynamics as a Controlled Twitter-Like Lab

The Chuang et al. paper and repo are very relevant to our current direction because they simulate opinion dynamics over agent interactions.

Their core experiment uses:

- agents with personas and initial opinion values;
- a discrete opinion space, e.g. `[-2, -1, 0, 1, 2]`;
- a random pair scheduler;
- one agent producing a tweet;
- another agent reading the tweet and reporting an updated belief;
- per-agent memory;
- logs for belief history, tweet history, response history, and conversation memory.

They also implement a control condition where agents do not interact. Each agent repeatedly reports opinions without seeing others' tweets. This isolates drift caused by repeated prompting from social influence caused by interaction.

They test prompt variants such as default, confirmation bias, and strong confirmation bias. The paper reports a key limitation: raw LLM agents tend toward accurate/scientific consensus, which can erase real-world resistance to consensus. Prompting confirmation bias creates more fragmentation and brings behavior closer to classic opinion-dynamics expectations.

Implication for us:

- Always include a no-interaction control.
- Track belief over time at agent level, not only final aggregates.
- Treat confirmation bias, conformity, skepticism, and susceptibility as explicit agent/state variables or prompt conditions.
- Do not assume off-the-shelf LLM agents naturally herd like humans.

## Chang et al.: LLMs Can Generate Networks, But Bias the Social Graph

Chang et al. evaluate LLM-generated social networks against real networks. Their finding is nuanced:

- local prompting methods, where the model constructs one persona's relations at a time, produce more realistic networks than global methods;
- generated networks can match density, clustering, connectivity, and degree distribution reasonably well;
- LLMs overemphasize political homophily and significantly overestimate political homophily.

The repo reflects this methodologically:

- personas are generated with demographic attributes;
- networks are generated as adjacency lists;
- methods include global, local, sequential, and iterative;
- token/cost statistics are saved;
- generated networks are compared with real-network statistics.

Implication for us:

- LLM-generated networks can be an experiment condition, not the default truth.
- For herding, graph topology must be controlled: Barabasi-Albert, Watts-Strogatz, Erdos-Renyi, stochastic block model, and empirical graph imports should all be possible.
- If we use LLM-generated ties, we should measure homophily, assortativity, clustering, degree distribution, and connected components before using the graph in outcome experiments.

## Echo Chamber and Rumor Work: Network Structure Drives Outcomes

The echo-chamber and rumor-spread papers converge on the same pattern:

- construct several network structures;
- assign personas to nodes;
- vary recommendation/selection/spreading policy;
- run message exposure and belief/update steps;
- compare against classical models or intervention baselines;
- measure polarization, reach, or echo-chamber indices.

Wang et al. construct typical social network structures, simulate interaction under recommendation algorithms, compare with Bounded Confidence Model and Friedkin-Johnsen baselines, and test active/passive nudges.

Hu et al. simulate rumor spreading across different network structures and agent behaviors. Their repo supports random, scale-free, small-world, and Facebook-data-based networks; selection policies such as random or more-friends-first; patient-zero policies; fact-checker variants; friend filtering; and post deletion probability.

Implication for us:

- "Herding" is not just a property of agents. It is often a property of the graph plus exposure rule.
- The simulator must make recommendation/exposure policy explicit.
- Fact-checking, counter-speech, edge removal, node suspension, and seed-node selection should be modular interventions.

## Steering the Herd: Information Cascades Need Public/Private Signals

"Steering the Herd" is especially useful for our first milestone because it deals directly with social learning and information cascades.

Their LLM simulation uses three roles:

- Agent: observes previous actions plus a private signal, forms belief, and chooses an action.
- Planner: observes action history and chooses private-signal precision.
- Oracle: generates a private signal with the desired precision.

This setup keeps the information structure explicit. Agents do not just chat; they face a controlled public/private signal problem.

They report that LLM agents deviate from Bayesian updating:

- they underreact to private signals aligned with prior beliefs;
- they overreact to private signals against prior beliefs;
- they require stronger public belief to enter an information cascade.

Implication for us:

- Sequential information cascade should be our cleanest first scientific benchmark.
- We need a Bayesian/rule-based baseline next to LLM agents.
- The simulator should separate private signal, observed public actions, belief update, and final action.
- For network herding, the same idea generalizes: each agent sees private evidence plus neighbor actions/messages, not the whole world.

## Artificial Societies: Product Signal, Not Peer-Reviewed Method

Artificial Societies is useful as a market signal. Their YC profile describes networks of AI personas that simulate stakeholder opinions, response to surveys, and reaction to information. They claim millions of personas grounded in human behavior, enterprise-scale response generation, and high accuracy against human self-replication.

We should not treat these claims as peer-reviewed evidence without methodological details. But the product shape reinforces the same strategic direction:

- the output is not a chat transcript, but an audience-level opinion distribution;
- personas are connected to influence/stakeholder networks;
- information exposure is a first-class input;
- customers care about counterfactual decisions: messaging, crisis communications, positioning, and stakeholder reaction.

Implication for us:

- Long-term value is likely in calibrated population response and scenario comparison, not just agent role-play.
- We need explicit uncertainty, validation, and calibration before making prediction claims.
- The product direction aligns with public-opinion simulation, but the research architecture should remain more transparent than a black-box market-research engine.

## Architecture Patterns in the Field

### Pattern 1: Full Generative Society

Agents have rich memory, goals, daily plans, conversations, and environment actions. AgentSociety and Generative Agents sit here.

Strengths:

- rich qualitative behavior;
- useful for scenario exploration;
- easier to inspect as narratives.

Weaknesses:

- high cost;
- weaker causal control;
- harder reproducibility;
- easy to mistake believability for predictive validity.

This is attractive as a long-term vision, not as our first implementation target.

### Pattern 2: Classical ABM Core With LLM Message Layer

The graph, state variables, interaction schedule, and update equations are explicit. LLMs generate messages, explanations, or structured decisions only where language matters.

Strengths:

- cheap;
- reproducible;
- comparable to known social-science models;
- good for ablations and causal experiments.

Weaknesses:

- less expressive than a full agent society;
- may miss qualitative reasoning and persuasive content effects.

This should be our v0 foundation.

### Pattern 3: Hybrid Structured LLM Agents

Agents receive controlled observations and must output structured JSON such as belief, confidence, action, rationale, and outgoing message. The environment owns graph, state, scheduling, and validation.

Strengths:

- keeps causal control while allowing richer cognition;
- works with OpenAI-compatible cheap models;
- logs can be audited.

Weaknesses:

- more expensive than rule-based models;
- LLM numeric consistency may be noisy;
- prompts and parsers become part of the experimental method.

This should be our v1 target after v0 baselines.

### Pattern 4: LLM-Generated Social Network

An LLM generates edges between personas. Chang et al. show this is possible but biased.

Strengths:

- flexible zero-shot network generation;
- useful when empirical network data is unavailable.

Weaknesses:

- political homophily bias;
- difficult to distinguish social reality from model stereotype;
- graph generation errors can dominate downstream herding results.

This should be an optional graph generator, not the default.

## Recommended Direction for This Project

Build a network-first society simulator. The first version should be a scientific experiment runner, not a city sandbox.

Core modules:

1. `AgentProfile`
   - stable attributes: demographic tags, ideology/topic prior, susceptibility, skepticism, conformity, attention, trust tendency.
2. `AgentState`
   - dynamic attributes: belief, confidence, action, memory summary, exposure count, emotion if needed later.
3. `Graph`
   - explicit topology: Erdos-Renyi, Watts-Strogatz, Barabasi-Albert, stochastic block model, empirical import, optional LLM-generated graph.
4. `Signal`
   - true state, private signal, public/news signal, precision, source credibility.
5. `ObservationPolicy`
   - what each agent can see: neighbor actions, neighbor messages, global trend, platform recommendation feed.
6. `UpdatePolicy`
   - pluggable update engines: Bayesian cascade, DeGroot, Friedkin-Johnsen, threshold model, voter model, LLM structured decision, hybrid.
7. `InteractionScheduler`
   - synchronous rounds, random pair interaction, network-neighbor exposure, feed-based exposure, sequential cascade.
8. `MessagePolicy`
   - none, templated messages, LLM-generated messages, persuasive agents, rumor seeds.
9. `Intervention`
   - seed selection, fact-checking, counter-message, edge removal, node suspension, recommendation diversification.
10. `Metrics`
   - cascade size, wrong-cascade rate, belief variance, polarization, modularity/assortativity, convergence time, recovery time, reach, cost.
11. `ReplayLog`
   - graph snapshot, run config, model provider, prompts, raw outputs, parsed outputs, metrics, random seeds, token/cost usage.

## First Experimental Ladder

### Milestone 0: Non-LLM Baselines

Implement no LLM calls first:

- sequential Bayesian information cascade;
- simple threshold/herding model on a graph;
- DeGroot or Friedkin-Johnsen opinion update;
- voter-model style neighbor imitation.

Purpose: verify the simulator, metrics, and replay without model cost.

### Milestone 1: Sequential Information Cascade With LLM Agents

Each agent sees:

- private signal;
- previous agents' actions;
- optional profile and bias parameters.

Each agent outputs:

- belief;
- confidence;
- action;
- short rationale.

Compare against:

- Bayesian rational baseline;
- heuristic baseline;
- no-social-information baseline.

Metrics:

- correct cascade rate;
- wrong cascade rate;
- point at which private signals are ignored;
- action/belief divergence;
- model cost per run.

### Milestone 2: Network Herding

Move from sequential public history to graph-local observation:

- agents have neighbors;
- agents see neighbor actions/messages from the previous round;
- topology and homophily are varied;
- seed nodes or news shocks are injected.

Metrics:

- cascade reach;
- cluster lock-in;
- polarization;
- hub influence;
- recovery after correction;
- intervention effectiveness.

### Milestone 3: Rumor/Echo Chamber

Add message content and platform exposure:

- rumor seed message;
- LLM or templated messages;
- recommendation policy;
- fact-check/counter-message interventions;
- optional edge/node interventions.

### Milestone 4: Public Opinion / Polling

Only after the mechanism is stable:

- introduce real news streams;
- ground agents in survey/persona data;
- compare against polling or historical opinion movement;
- include uncertainty and calibration, not just point predictions.

## Model Provider Implications

Using cheap Chinese or OpenAI-compatible models is compatible with this architecture because model calls live behind `UpdatePolicy` and `MessagePolicy`.

The simulator should support:

- rule-based runs with zero model calls;
- cheap LLM runs for broad sweeps;
- stronger OpenAI model runs for smaller validation runs;
- token and cost logging at every call;
- dry-run cost estimates from the run config.

The provider choice is not the main architecture decision. The architecture decision is to keep LLM calls replaceable and measurable.

## Decision Recommendation

Adopt Pattern 2 as the foundation and Pattern 3 as the near-term target:

> Explicit graph/state/experiment runner first; LLM agents as structured, replaceable cognition/message modules.

This gives us the best path toward the user's long-term goal: a society simulator that can study crowd psychology and eventually public opinion. It also avoids the common failure mode of building a vivid but scientifically slippery sandbox.

The first design spec should be for a "network herding experiment runner," not a general artificial society. The simulator should feel small at first, but its abstractions should be the ones that later scale into a larger society.

## Concrete Do / Do Not

Do:

- make networks explicit and inspectable;
- make observation scope explicit;
- log every prompt, parsed output, state transition, and random seed;
- run no-interaction and non-LLM controls;
- make model providers interchangeable;
- keep interventions as first-class objects;
- treat validation as central from day one.

Do not:

- let LLMs freely generate the social graph and call it society;
- use agent count as the main design axis;
- begin with a city simulation;
- rely on believable conversations as evidence;
- claim polling prediction before grounding and calibration exist.
