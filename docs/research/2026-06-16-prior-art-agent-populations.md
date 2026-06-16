# Prior-Art Report: Agent Populations for Behavioral Simulation

Date: 2026-06-16

Status note: this report reflects the earlier behavioral-economics framing. After subsequent scoping, the project direction moved toward crowd psychology, information cascades, and network herding. For the current architecture recommendation, see `docs/research/2026-06-16-network-herding-prior-art.md`.

## Purpose

This report summarizes how the cited projects construct agents, populations, interactions, and validation loops. The goal is to inform the first milestone of this repository:

- Domain: behavioral economics.
- First benchmark: a mini benchmark with one framing-effect task and one loss-aversion task.
- Validation target: aggregate results from classic literature and replication-oriented sources.
- Current population decision: start with a synthetic population with explicit demographic and psychographic attributes.

The key question is not "can agents talk to each other?" It is: can a population of agents reproduce known aggregate behavioral effects in a way that is inspectable, repeatable, and calibrated against human evidence?

## Sources Reviewed

- Stanford CS222: AI Agents and Simulations
  - https://joonspk-research.github.io/cs222-fall24/index.html
- Joon Sung Park site and dissertation listing
  - https://www.joonsungpark.com/
  - https://www.joonsungpark.com/papers/Joon_Sung_Park_Dissertation/
  - Stanford Digital Repository record: https://purl.stanford.edu/jm164ch6237
- Park et al. 2023, "Generative Agents: Interactive Simulacra of Human Behavior"
  - https://arxiv.org/abs/2304.03442
  - https://ar5iv.labs.arxiv.org/html/2304.03442
  - https://github.com/joonspk-research/generative_agents
- Park et al. 2024/2026, "LLM Agents Grounded in Self-Reports Enable General-Purpose Simulation of Individuals"
  - https://arxiv.org/abs/2411.10109
  - https://github.com/StanfordHCI/genagents
- Piao et al. 2025/2026, "AgentSociety"
  - https://arxiv.org/abs/2502.08691
  - https://arxiv.org/html/2502.08691v1
  - https://github.com/tsinghua-fib-lab/agentsociety/
- Artificial Societies, YC company page
  - https://www.ycombinator.com/companies/artificial-societies
- Classic behavioral economics targets for the first benchmark
  - Tversky and Kahneman 1981, "The Framing of Decisions and the Psychology of Choice"
    - https://sites.stat.columbia.edu/gelman/surveys.course/TverskyKahneman1981.pdf
  - Kahneman, Knetsch, and Thaler 1991, "The Endowment Effect, Loss Aversion, and Status Quo Bias"
    - https://www.aeaweb.org/articles?id=10.1257%2Fjep.5.1.193
  - Tversky and Kahneman 1991, "Loss Aversion in Riskless Choice"
    - https://ideas.repec.org/a/oup/qjecon/v106y1991i4p1039-1061..html

## Field Frame From CS222

CS222 frames human behavioral simulation as a way to ask counterfactual "what if" questions about complex social systems. The course sequence matters for us: it separates individuals, groups, and populations; then cognitive architectures; then generative agents; then interactive worlds; then believability versus accuracy; then individual models and agent banks.

The important lesson is that "believable" and "accurate" are separate axes. A small town of agents that feels alive is not automatically a valid prediction engine. For this project, the first milestone should sit closer to the accuracy side: reproduce known aggregate effects under controlled stimuli before building a large social world.

Implication for us:

- Start with a benchmark runner, not a sandbox world.
- Treat agent population design as an experimental variable.
- Keep the first environment minimal: stimuli in, structured responses out.
- Add interactions later, after individual-response validity is measurable.

## Park's Research Trajectory

Park's site describes the throughline clearly: generative agents with memory, reflection, and planning; then methods to ground agents in real-world data and validate simulated attitudes and behaviors against ground truth; then population-scale simulations with agents representative of real-world populations.

The dissertation, "Generative Agent Simulations of Human Behavior," consolidates this line. The Stanford Digital Repository abstract frames generative-agent simulations as scientific instruments for probing assumptions embedded in institutions, revealing emergent dynamics, and exploring counterfactual designs before deployment. It names three architectural capabilities--memory, reflection, and planning--then adds methods for constructing agents from rich individual-level data and a measurement framework for evaluating alignment against ground truth on people's attitudes and behaviors. It also describes a population simulation of 1,000 agents representing a cross-section of U.S. adults.

For our immediate decision, the dissertation is best treated as the umbrella argument: useful social simulation requires moving from believable agent behavior toward grounded, validated agent populations.

Implication for us:

- We should borrow the "agent as inspectable cognitive unit" idea.
- We should not stop at Park 2023 style believability.
- The first project claim should be modest: synthetic agents can be used to prototype a behavioral-economics benchmark and compare population-level effects against literature.

## Generative Agents 2023

### How They Build Agents

Generative Agents uses 25 unique agents in a Smallville sandbox. Each agent begins with a human-authored natural-language seed description containing identity, occupation, relationships, preferences, and local social context. These descriptions are split into initial memories.

The architecture has three core modules:

- Memory stream: stores the agent's experiences in natural language.
- Reflection: synthesizes higher-level inferences from memories.
- Planning: creates high-level and detailed behavior plans from memory, reflection, and environment.

Agents communicate in natural language and act in an environment. The system emphasizes long-term coherence across unfolding interactions rather than one-shot survey response.

### How They Validate

The validation is mostly believability-oriented. They interview agents to test whether they stay in character, remember, plan, react, and reflect. They also run end-to-end social scenarios, such as a Valentine party spreading through the community, and ablate memory/reflection/planning.

### What We Should Copy

- Use explicit agent state rather than one-shot prompting.
- Keep memory and reflections inspectable.
- Save prompts, outputs, and intermediate reasoning artifacts for audit.
- Probe agents with interviews/surveys in natural language.

### What We Should Not Copy Yet

- Do not start with a full simulated town.
- Do not validate primarily by "it looks believable."
- Do not hand-author rich backstories for a small number of agents if the first goal is aggregate behavioral economics.

For our first milestone, this paper is an architecture inspiration, not the population model.

## Self-Report Grounded Agents 2024/2026

### How They Build Agents

This is the closest methodological fit for serious behavioral prediction. The paper builds person-specific generative agents from a diverse national sample of 1,052 Americans. Agents are grounded in one of three data sources:

- Two-hour semi-structured interviews using the American Voices Project schedule.
- Structured surveys, including General Social Survey items and Big Five personality inventory.
- A combination of interviews and surveys.

The current public repo, `StanfordHCI/genagents`, exposes tools for creating and interacting with agents, querying them with categorical, numerical, and open-ended questions, and using memory/reflection. It also includes a demographic agent bank of over 3,000 agents based on GSS demographic information as a starting point. The real interview-grounded 1,000-person agent bank is not fully public because of privacy constraints.

### How They Validate

They compare simulated responses against the real participants' held-out responses. The key metric is not ordinary accuracy in the abstract; it is accuracy relative to participants' own two-week test-retest consistency. Reported results:

- Interview-only agents: 83% of test-retest consistency.
- Survey-only agents: 82%.
- Interview + survey agents: 86%.
- Demographics-only baseline: 74%.

They also evaluate personality traits and experimental behaviors, and report reduced disparities across racial and ideological groups relative to demographics-only baselines.

### What We Should Copy

- Use structured survey-style tasks as the interface.
- Include a demographics-only baseline.
- Include a richer profile baseline, even if synthetic at first.
- Score against human reliability or aggregate literature targets rather than subjective believability.
- Treat richer self-report data as the eventual path to higher validity.

### What We Should Be Careful About

Our selected first version is synthetic population A. That is fine for v0, but this literature says synthetic demographics are a lower-validity baseline, not the end state. If we claim "predicting people," we eventually need real self-report or survey-grounded agents. Until then, we should say "benchmarking whether an agent population can reproduce known aggregate effects under controlled conditions."

## AgentSociety 2025/2026

### How They Build Agents

AgentSociety is a large-scale social simulator. Its agent design separates:

- Profile: static attributes such as name, age, gender, education, personality.
- Status: dynamic state such as needs, satisfaction, financial status, emotions, and relationships.
- Mental processes: emotion, needs, and cognition.
- Social behaviors: mobility, online social interaction, employment, and consumption.

The paper explicitly ties these parts to psychology, economics, and behavioral science. It is less "one prompt per agent" and more a modular social actor whose behavior is mediated by profile, state, need, cognition, and environment.

### How They Build Populations

For large scenarios, AgentSociety uses real-world data sources where possible. For example, the hurricane case uses SafeGraph POI/mobility data and Census Block Group demographic profiles, sampling agents with attributes such as gender, age, race, income, and home CBG.

### How They Model Interactions

AgentSociety models relationship types such as family, friendship, and colleagues, each with relationship strength. Agents select interaction partners based on relationship type, strength, needs, topic relevance, and profile characteristics. Conversations update emotional state, relationships, and cognition. The platform also supports structured survey delivery to agents and extensive logging.

### How They Validate

Validation is case-specific and social-outcome-oriented:

- Polarization: compare opinion movement under control, homophilic exposure, and heterogeneous exposure.
- Inflammatory messages: compare diffusion and emotional intensity under different interventions.
- UBI: compare economic and mental-health metrics against observed effects in a Texas UBI-related setting.
- Hurricane shock: compare simulated mobility patterns against real mobility trends.

### What We Should Copy

- Separate static profile from dynamic state.
- Include psychographic variables relevant to the task, not just demographics.
- Keep survey outcomes, state changes, prompts, and logs as first-class data.
- Use interventions as experimental treatments.

### What We Should Not Copy Yet

- Do not adopt a full urban environment for the first benchmark.
- Do not add social interactions before we know the individual choice module works.
- Do not model mobility/economy unless the target effect requires them.

For our first behavioral-economics benchmark, AgentSociety supports our A decision: synthetic agents with explicit demographics and psychographics are a reasonable starting point if the state schema is transparent and the validation is honest.

## Artificial Societies YC

Artificial Societies is a product signal rather than a peer-reviewed methodology reference. Their public page says they build networks of AI personas that simulate stakeholder opinions, simulate how large groups respond to surveys and information, and have built over 2.5 million personas grounded in real human behavior. They also claim high distribution accuracy against human self-replication.

What matters for us is the product shape:

- Audience/persona network.
- Survey and information exposure as inputs.
- Aggregate opinion distribution as output.
- Validation framed around distribution accuracy and self-replication.

We should treat their accuracy numbers as claims, not scientific evidence, unless independent details are available. But the product direction reinforces the same strategic point: the valuable artifact is not chatty agents. It is a calibrated audience simulator with measurable response distributions.

## Behavioral Economics Targets

### Framing Effect

The classic Asian Disease Problem is suitable for v0 because it gives us a clean manipulation: equivalent outcomes are framed as gains or losses, and the aggregate preference shifts. This is exactly the kind of effect an agent simulator should be able to reproduce if it captures non-rational framing sensitivity.

For v0, use one gain-frame condition and one loss-frame condition. The main measure is the change in risky-option choice rate between conditions.

### Loss Aversion

For loss aversion, the cleanest v0 task is a mixed gamble or selling/buying valuation task. The key target is asymmetry: losses should weigh more than equivalent gains. The main measure can be the acceptance threshold for a 50/50 gain-loss gamble, or the WTA/WTP gap in an endowment-style task.

The endowment-effect route is more complex because it requires ownership framing and valuation. A mixed-gamble acceptance task is cleaner for a first benchmark.

## Cross-Source Pattern

The reviewed work suggests a ladder of agent population validity:

1. Prompt-only repeated samples.
   - Fast, but mostly measures model priors.
2. Synthetic demographics.
   - Better control, still weak grounding.
3. Synthetic demographics plus psychographics.
   - Good for mechanistic prototyping and ablations.
4. Real survey-grounded agents.
   - Stronger empirical grounding.
5. Interview/self-report grounded agents.
   - Best current path for individual-level simulation.
6. Real behavior and network grounded agents.
   - Needed for high-stakes population forecasting.

Our selected path is level 3 for v0. That is the right pragmatic starting point, as long as we explicitly define it as a benchmark-development baseline rather than a finished predictor of society.

## Recommendation for Our First Benchmark

Build a synthetic population benchmark with three baselines:

1. No-persona LLM baseline.
   - Same prompt, repeated many times.
   - Purpose: measure raw model prior.
2. Demographics-only synthetic agents.
   - Age, gender, education, income, occupation.
   - Purpose: compare against the common weaker baseline used in self-report grounded work.
3. Demographics plus psychographics synthetic agents.
   - Add risk tolerance, loss sensitivity, numeracy, need for cognition, impulsivity, and uncertainty aversion.
   - Purpose: test whether explicit behavioral variables improve aggregate effect reproduction.

The v0 benchmark should include:

- Task 1: Asian Disease-style framing.
- Task 2: mixed-gamble loss-aversion task.
- Agent count: start with 300 to 1,000 synthetic agents.
- Randomization: assign each agent to one condition per task; avoid showing both frames to the same agent in the same run.
- Output format: forced choice plus short rationale.
- Metrics:
  - Directional correctness: does the effect go the same way as human literature?
  - Effect-size error: how far is the simulated aggregate shift from the target?
  - Calibration error: difference between simulated choice proportions and literature proportions.
  - Baseline delta: psychographic agents should beat no-persona and demographics-only baselines.
- Audit logs:
  - agent profile
  - task condition
  - prompt
  - raw model response
  - parsed choice
  - rationale
  - run metadata

## Design Implications

The first implementation should be a benchmark lab, not a society engine.

Recommended initial modules:

- `AgentProfile`
  - Static demographics and psychographics.
- `PopulationSampler`
  - Creates synthetic agents from configurable distributions.
- `Stimulus`
  - Defines behavioral-economics tasks and conditions.
- `ExperimentRunner`
  - Assigns agents to conditions and collects responses.
- `ResponseParser`
  - Converts model outputs into structured choices.
- `Evaluator`
  - Compares aggregate results to literature targets.
- `RunLog`
  - Saves everything needed to reproduce and audit a run.

Do not implement memory, reflection, social networks, or multi-day interaction in v0 unless the benchmark fails specifically because those are missing. For framing and loss aversion, the immediate question is individual decision response under controlled framing.

## Decision Record

Current decisions:

- First research domain: behavioral economics.
- First benchmark: framing effect plus loss aversion.
- First validation target: aggregate human literature.
- First population model: synthetic agents with explicit demographics and psychographics.

Decision still needed:

- Which model/API provider to use for v0.
- Target agent count for the first run.
- Exact literature target proportions for the two tasks.
- Whether v0 should be CLI-only or include a small report-generation notebook/script.

## Bottom Line

The cited research points in one direction: serious social simulation must move from "agents that feel alive" to "agents whose aggregate and individual responses can be validated." For this project, the right first move is a small, instrumented behavioral-economics benchmark. Synthetic demographics plus psychographics are a strong v0 choice because they are cheap, transparent, and easy to ablate. But the report should treat them as an entry point, not the final scientific standard.
