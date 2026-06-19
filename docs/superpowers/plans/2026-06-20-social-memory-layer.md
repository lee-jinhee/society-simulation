# Social Memory Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an auditable social memory layer to `event_driven_opinion_dynamics` so agents can store event/message/decision memories, retrieve relevant memories before each decision, and write replay traces for later ablation studies.

**Architecture:** Create a focused `event_memory.py` module for memory dataclasses, deterministic memory creation, and retrieval scoring. Thread memory retrieval through the existing event runner and policy prompt without changing the decision output schema. Persist memories and retrieval traces through the existing replay artifact pattern and expose operational memory metrics.

**Tech Stack:** Python 3.11 dataclasses, standard-library JSON/path utilities, existing pytest suite, existing `event_*` modules, no external embedding or vector database dependency.

---

## Scope Check

This plan implements the first deterministic social memory layer. It does not run a new paid LLM experiment and does not implement reflection generation yet. It leaves a clean `reflection` memory kind for the next phase.

## File Structure

Create:

- `src/society_simulation/event_memory.py`: social memory models, memory creation helpers, retrieval scoring, retrieval trace serialization.
- `tests/test_event_memory.py`: validation and retrieval unit tests.

Modify:

- `src/society_simulation/event_config.py`: parse and validate optional `memory_retrieval` config.
- `tests/test_event_config.py`: cover memory config defaults and validation errors.
- `src/society_simulation/event_policy.py`: accept optional retrieved memories and include them in the prompt.
- `tests/test_event_policy.py`: verify prompt includes memories only when passed.
- `src/society_simulation/event_runner.py`: create memories, retrieve memories before decisions, pass retrieved memories to policy, collect traces.
- `tests/test_event_runner.py`: verify enabled/disabled runner behavior and replay artifacts.
- `src/society_simulation/event_replay.py`: write `memories.jsonl` and `retrievals.jsonl`.
- `tests/test_event_replay.py`: verify artifact writing.
- `src/society_simulation/event_metrics.py`: add memory summary metrics.
- `tests/test_event_metrics.py`: verify metric fields.
- `experiments/event_driven_congestion_pricing.json`: keep retrieval disabled by default for backward-compatible mock behavior.
- `experiments/event_driven_congestion_pricing_gpt54_mini_pilot.json`: keep retrieval disabled to preserve the already reported paid pilot.

---

### Task 1: Social Memory Models and Retrieval

**Files:**
- Create: `src/society_simulation/event_memory.py`
- Create: `tests/test_event_memory.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_event_memory.py`:

```python
import pytest

from society_simulation.event_memory import (
    MemoryQuery,
    SocialMemory,
    build_memory_query,
    retrieve_memories,
)
from society_simulation.event_models import EventExposure


def memory(
    memory_id: str,
    *,
    day: int,
    text: str,
    importance: float = 0.5,
    source_trust: float = 0.5,
    emotional_intensity: float = 0.5,
    identity_relevance: float = 0.5,
) -> SocialMemory:
    return SocialMemory(
        memory_id=memory_id,
        agent_id="jisoo",
        day=day,
        kind="event_exposure",
        text=text,
        source_id=memory_id,
        source_type="event",
        channel="news_feed",
        related_agent_ids=(),
        related_event_ids=(memory_id,),
        stance_signal=0.2,
        emotional_intensity=emotional_intensity,
        source_trust=source_trust,
        identity_relevance=identity_relevance,
        importance=importance,
        private=False,
    )


def test_social_memory_validates_probability_fields() -> None:
    with pytest.raises(ValueError, match="importance must be a number between 0 and 1"):
        memory("bad", day=1, text="bad", importance=1.2)


def test_social_memory_validates_stance_signal() -> None:
    with pytest.raises(ValueError, match="stance_signal must be a number between -1 and 1"):
        SocialMemory(
            memory_id="bad",
            agent_id="jisoo",
            day=1,
            kind="event_exposure",
            text="bad",
            source_id="event",
            source_type="event",
            channel="news_feed",
            related_agent_ids=(),
            related_event_ids=(),
            stance_signal=1.2,
            emotional_intensity=0.2,
            source_trust=0.5,
            identity_relevance=0.5,
            importance=0.5,
            private=False,
        )


def test_retrieve_memories_returns_top_scored_memories_with_components() -> None:
    query = MemoryQuery(
        agent_id="jisoo",
        day=4,
        text="hospital commute asthma traffic",
        related_agent_ids=(),
        related_event_ids=("health_report",),
        stance_hint=0.4,
        affected_interests=("hospital access", "public health"),
    )
    memories = (
        memory(
            "health_report",
            day=3,
            text="public health report links traffic to asthma visits",
            importance=0.8,
            source_trust=0.9,
            emotional_intensity=0.4,
            identity_relevance=0.9,
        ),
        memory(
            "old_parking",
            day=1,
            text="downtown parking was expensive",
            importance=0.2,
            source_trust=0.4,
            emotional_intensity=0.2,
            identity_relevance=0.1,
        ),
    )

    retrieved = retrieve_memories(memories, query, limit=1)

    assert len(retrieved) == 1
    assert retrieved[0].memory.memory_id == "health_report"
    assert retrieved[0].score > 0.0
    assert retrieved[0].relevance_score > retrieved[0].recency_score / 2
    assert retrieved[0].to_dict()["memory"]["memory_id"] == "health_report"


def test_retrieve_memories_filters_to_agent_and_excludes_future_memories() -> None:
    query = MemoryQuery(
        agent_id="jisoo",
        day=2,
        text="traffic",
        related_agent_ids=(),
        related_event_ids=(),
        stance_hint=0.0,
        affected_interests=(),
    )
    other_agent = SocialMemory(
        memory_id="other",
        agent_id="minho",
        day=1,
        kind="event_exposure",
        text="traffic",
        source_id="event",
        source_type="event",
        channel="news_feed",
        related_agent_ids=(),
        related_event_ids=(),
        stance_signal=0.0,
        emotional_intensity=0.5,
        source_trust=0.5,
        identity_relevance=0.5,
        importance=0.5,
        private=False,
    )
    future = memory("future", day=3, text="traffic")

    assert retrieve_memories((other_agent, future), query, limit=5) == ()


def test_build_memory_query_combines_exposure_text_and_metadata() -> None:
    exposure = EventExposure(
        day=2,
        agent_id="jisoo",
        source_type="event",
        source_id="public_health_report",
        channel="news_feed",
        content="Report links traffic to asthma near schools.",
    )

    query = build_memory_query(
        agent_id="jisoo",
        day=2,
        exposures=(exposure,),
        affected_interests=("public health",),
    )

    assert query.text == "Report links traffic to asthma near schools."
    assert query.related_event_ids == ("public_health_report",)
    assert query.affected_interests == ("public health",)
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
/home/jhlee/repo/society-simulation/.venv/bin/python -m pytest tests/test_event_memory.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'society_simulation.event_memory'`.

- [ ] **Step 3: Implement `event_memory.py`**

Create `src/society_simulation/event_memory.py` with:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp
import re

from society_simulation.event_models import EventExposure, validate_probability, validate_stance

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_VALID_MEMORY_KINDS = {
    "event_exposure",
    "social_message",
    "self_reasoning",
    "self_message",
    "reflection",
}


def _require_non_empty_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _str_tuple(value: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{field_name} must be a tuple")
    return tuple(_require_non_empty_str(item, f"{field_name}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True)
class SocialMemory:
    memory_id: str
    agent_id: str
    day: int
    kind: str
    text: str
    source_id: str
    source_type: str
    channel: str
    related_agent_ids: tuple[str, ...]
    related_event_ids: tuple[str, ...]
    stance_signal: float
    emotional_intensity: float
    source_trust: float
    identity_relevance: float
    importance: float
    private: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory_id", _require_non_empty_str(self.memory_id, "memory_id"))
        object.__setattr__(self, "agent_id", _require_non_empty_str(self.agent_id, "agent_id"))
        object.__setattr__(self, "day", _require_non_negative_int(self.day, "day"))
        object.__setattr__(self, "kind", _require_non_empty_str(self.kind, "kind"))
        if self.kind not in _VALID_MEMORY_KINDS:
            raise ValueError("kind must be a supported social memory kind")
        object.__setattr__(self, "text", _require_non_empty_str(self.text, "text"))
        object.__setattr__(self, "source_id", _require_non_empty_str(self.source_id, "source_id"))
        object.__setattr__(self, "source_type", _require_non_empty_str(self.source_type, "source_type"))
        object.__setattr__(self, "channel", _require_non_empty_str(self.channel, "channel"))
        object.__setattr__(self, "related_agent_ids", _str_tuple(self.related_agent_ids, "related_agent_ids"))
        object.__setattr__(self, "related_event_ids", _str_tuple(self.related_event_ids, "related_event_ids"))
        object.__setattr__(self, "stance_signal", validate_stance(self.stance_signal, "stance_signal"))
        object.__setattr__(self, "emotional_intensity", validate_probability(self.emotional_intensity, "emotional_intensity"))
        object.__setattr__(self, "source_trust", validate_probability(self.source_trust, "source_trust"))
        object.__setattr__(self, "identity_relevance", validate_probability(self.identity_relevance, "identity_relevance"))
        object.__setattr__(self, "importance", validate_probability(self.importance, "importance"))
        if not isinstance(self.private, bool):
            raise ValueError("private must be a boolean")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryQuery:
    agent_id: str
    day: int
    text: str
    related_agent_ids: tuple[str, ...]
    related_event_ids: tuple[str, ...]
    stance_hint: float
    affected_interests: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _require_non_empty_str(self.agent_id, "agent_id"))
        object.__setattr__(self, "day", _require_non_negative_int(self.day, "day"))
        object.__setattr__(self, "text", _require_non_empty_str(self.text, "text"))
        object.__setattr__(self, "related_agent_ids", _str_tuple(self.related_agent_ids, "related_agent_ids"))
        object.__setattr__(self, "related_event_ids", _str_tuple(self.related_event_ids, "related_event_ids"))
        object.__setattr__(self, "stance_hint", validate_stance(self.stance_hint, "stance_hint"))
        object.__setattr__(self, "affected_interests", _str_tuple(self.affected_interests, "affected_interests"))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievedMemory:
    memory: SocialMemory
    score: float
    recency_score: float
    relevance_score: float
    importance_score: float
    trust_score: float
    emotion_score: float
    identity_score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "memory": self.memory.to_dict(),
            "score": self.score,
            "recency_score": self.recency_score,
            "relevance_score": self.relevance_score,
            "importance_score": self.importance_score,
            "trust_score": self.trust_score,
            "emotion_score": self.emotion_score,
            "identity_score": self.identity_score,
        }


def build_memory_query(
    *,
    agent_id: str,
    day: int,
    exposures: tuple[EventExposure, ...],
    affected_interests: tuple[str, ...],
) -> MemoryQuery:
    text = " ".join(exposure.content for exposure in exposures).strip() or "No new information."
    return MemoryQuery(
        agent_id=agent_id,
        day=day,
        text=text,
        related_agent_ids=tuple(
            sorted(
                {
                    exposure.source_id.split(":", 1)[0]
                    for exposure in exposures
                    if exposure.source_type == "message" and ":" in exposure.source_id
                }
            )
        ),
        related_event_ids=tuple(
            sorted(
                exposure.source_id
                for exposure in exposures
                if exposure.source_type == "event"
            )
        ),
        stance_hint=0.0,
        affected_interests=affected_interests,
    )


def retrieve_memories(
    memories: tuple[SocialMemory, ...],
    query: MemoryQuery,
    *,
    limit: int,
) -> tuple[RetrievedMemory, ...]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    scored = [
        _score_memory(memory, query)
        for memory in memories
        if memory.agent_id == query.agent_id and memory.day <= query.day
    ]
    return tuple(sorted(scored, key=lambda item: (-item.score, item.memory.memory_id))[:limit])


def _score_memory(memory: SocialMemory, query: MemoryQuery) -> RetrievedMemory:
    recency_score = exp(-0.35 * max(0, query.day - memory.day))
    relevance_score = _relevance_score(memory, query)
    importance_score = memory.importance
    trust_score = memory.source_trust
    emotion_score = memory.emotional_intensity
    identity_score = memory.identity_relevance
    score = (
        0.20 * recency_score
        + 0.25 * relevance_score
        + 0.20 * importance_score
        + 0.10 * trust_score
        + 0.10 * emotion_score
        + 0.15 * identity_score
    )
    return RetrievedMemory(
        memory=memory,
        score=score,
        recency_score=recency_score,
        relevance_score=relevance_score,
        importance_score=importance_score,
        trust_score=trust_score,
        emotion_score=emotion_score,
        identity_score=identity_score,
    )


def _relevance_score(memory: SocialMemory, query: MemoryQuery) -> float:
    query_tokens = _tokens(query.text) | _tokens(" ".join(query.affected_interests))
    memory_tokens = _tokens(memory.text)
    token_score = (
        len(query_tokens & memory_tokens) / len(query_tokens | memory_tokens)
        if query_tokens and memory_tokens
        else 0.0
    )
    agent_score = 0.25 if set(memory.related_agent_ids) & set(query.related_agent_ids) else 0.0
    event_score = 0.35 if set(memory.related_event_ids) & set(query.related_event_ids) else 0.0
    return min(1.0, token_score + agent_score + event_score)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))
```

- [ ] **Step 4: Run focused tests and verify pass**

Run:

```bash
/home/jhlee/repo/society-simulation/.venv/bin/python -m pytest tests/test_event_memory.py -q
```

Expected: `5 passed`.

- [ ] **Step 5: Run full suite**

Run:

```bash
/home/jhlee/repo/society-simulation/.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/society_simulation/event_memory.py tests/test_event_memory.py
git commit -m "feat: add social memory retrieval models"
```

---

### Task 2: Memory Retrieval Config

**Files:**
- Modify: `src/society_simulation/event_config.py`
- Modify: `tests/test_event_config.py`

- [ ] **Step 1: Add failing config tests**

Append tests to `tests/test_event_config.py`:

```python
def test_event_config_defaults_memory_retrieval_disabled(tmp_path: Path) -> None:
    config = EventDrivenOpinionConfig.from_dict(valid_event_config(tmp_path))

    assert config.memory_retrieval == {
        "enabled": False,
        "limit": 5,
    }


def test_event_config_accepts_memory_retrieval_settings(tmp_path: Path) -> None:
    data = valid_event_config(tmp_path)
    data["memory_retrieval"] = {"enabled": True, "limit": 3}

    config = EventDrivenOpinionConfig.from_dict(data)

    assert config.memory_retrieval == {"enabled": True, "limit": 3}
    assert config.to_dict()["memory_retrieval"] == {"enabled": True, "limit": 3}


@pytest.mark.parametrize(
    ("memory_retrieval", "message"),
    [
        ({"enabled": "yes", "limit": 3}, "memory_retrieval.enabled must be a boolean"),
        ({"enabled": True, "limit": 0}, "memory_retrieval.limit must be positive"),
        ({"enabled": True, "limit": True}, "memory_retrieval.limit must be an integer"),
        ({"enabled": True, "unknown": 1}, "unsupported memory_retrieval key: unknown"),
    ],
)
def test_event_config_rejects_invalid_memory_retrieval(
    tmp_path: Path,
    memory_retrieval: dict[str, object],
    message: str,
) -> None:
    data = valid_event_config(tmp_path)
    data["memory_retrieval"] = memory_retrieval

    with pytest.raises(ValueError, match=message):
        EventDrivenOpinionConfig.from_dict(data)
```

- [ ] **Step 2: Run focused config tests and verify failure**

Run:

```bash
/home/jhlee/repo/society-simulation/.venv/bin/python -m pytest tests/test_event_config.py -q
```

Expected: fail because `EventDrivenOpinionConfig` has no `memory_retrieval` field.

- [ ] **Step 3: Implement config parsing**

In `src/society_simulation/event_config.py`:

1. Add helper:

```python
def _normalize_memory_retrieval(value: object | None) -> Mapping[str, object]:
    if value is None:
        return MappingProxyType({"enabled": False, "limit": 5})
    data = dict(_require_free_form_mapping(value, "memory_retrieval"))
    for key in data:
        if not isinstance(key, str):
            raise ValueError("object keys must be strings")
        if key not in {"enabled", "limit"}:
            raise ValueError(f"unsupported memory_retrieval key: {key}")
    enabled = data.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("memory_retrieval.enabled must be a boolean")
    limit = data.get("limit", 5)
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("memory_retrieval.limit must be an integer")
    if limit <= 0:
        raise ValueError("memory_retrieval.limit must be positive")
    return MappingProxyType({"enabled": enabled, "limit": limit})
```

2. Add dataclass field:

```python
memory_retrieval: Mapping[str, object]
```

3. In `__post_init__`, normalize it:

```python
object.__setattr__(
    self,
    "memory_retrieval",
    _normalize_memory_retrieval(self.memory_retrieval),
)
```

4. In `from_dict`, pass:

```python
memory_retrieval=_normalize_memory_retrieval(data.get("memory_retrieval")),
```

5. In `to_dict`, include:

```python
"memory_retrieval": _to_json_ready_mapping(self.memory_retrieval),
```

- [ ] **Step 4: Run tests**

Run:

```bash
/home/jhlee/repo/society-simulation/.venv/bin/python -m pytest tests/test_event_config.py -q
/home/jhlee/repo/society-simulation/.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/society_simulation/event_config.py tests/test_event_config.py
git commit -m "feat: configure event memory retrieval"
```

---

### Task 3: Replay and Metrics for Memory Traces

**Files:**
- Modify: `src/society_simulation/event_replay.py`
- Modify: `src/society_simulation/event_metrics.py`
- Modify: `tests/test_event_replay.py`
- Modify: `tests/test_event_metrics.py`

- [ ] **Step 1: Add failing replay test**

Add to `tests/test_event_replay.py`:

```python
from society_simulation.event_memory import SocialMemory, RetrievedMemory


def test_event_replay_writes_memory_artifacts(tmp_path: Path) -> None:
    config = EventDrivenOpinionConfig.from_dict(valid_event_config(tmp_path))
    writer = EventReplayWriter(config)
    state = config.agents[0].initial_state(day=0)
    social_memory = SocialMemory(
        memory_id="jisoo:1:event:city",
        agent_id="jisoo",
        day=1,
        kind="event_exposure",
        text="City announcement",
        source_id="city",
        source_type="event",
        channel="news_feed",
        related_agent_ids=(),
        related_event_ids=("city",),
        stance_signal=0.2,
        emotional_intensity=0.3,
        source_trust=0.8,
        identity_relevance=0.6,
        importance=0.7,
        private=False,
    )

    output_dir = writer.write(
        states_by_day=((state,),),
        exposures=(),
        messages=(),
        metrics={"agent_count": 1},
        llm_decisions=(),
        memories=(social_memory,),
        retrievals=(
            {
                "agent_id": "jisoo",
                "day": 1,
                "query": {"agent_id": "jisoo"},
                "retrieved": [RetrievedMemory(social_memory, 1, 1, 1, 1, 1, 1, 1).to_dict()],
            },
        ),
    )

    assert (output_dir / "memories.jsonl").exists()
    assert (output_dir / "retrievals.jsonl").exists()
```

- [ ] **Step 2: Add failing metrics test**

Add to `tests/test_event_metrics.py`:

```python
from society_simulation.event_memory import SocialMemory


def test_compute_event_metrics_includes_memory_summary() -> None:
    states_by_day = ((state("jisoo", day=0, private=-0.1, public=0.0),),)
    memories = (
        SocialMemory(
            memory_id="m1",
            agent_id="jisoo",
            day=0,
            kind="self_reasoning",
            text="private concern",
            source_id="decision",
            source_type="self",
            channel="internal",
            related_agent_ids=(),
            related_event_ids=(),
            stance_signal=-0.1,
            emotional_intensity=0.4,
            source_trust=1.0,
            identity_relevance=0.8,
            importance=0.7,
            private=True,
        ),
    )

    metrics = compute_event_metrics(states_by_day, (), memories=memories, retrievals=())

    assert metrics["memory_count"] == 1
    assert metrics["private_memory_count"] == 1
    assert metrics["public_memory_count"] == 0
    assert metrics["retrieval_count"] == 0
```

- [ ] **Step 3: Run focused tests and verify failure**

Run:

```bash
/home/jhlee/repo/society-simulation/.venv/bin/python -m pytest tests/test_event_replay.py tests/test_event_metrics.py -q
```

Expected: fail because replay and metrics do not accept memory arguments yet.

- [ ] **Step 4: Implement replay and metrics changes**

In `EventReplayWriter.write`, add keyword-only parameters with defaults:

```python
memories: tuple[object, ...] = (),
retrievals: tuple[dict[str, Any], ...] = (),
```

Then write:

```python
self._write_jsonl(
    output_dir / "memories.jsonl",
    tuple(memory.to_dict() for memory in memories if hasattr(memory, "to_dict")),
)
self._write_jsonl(output_dir / "retrievals.jsonl", retrievals)
```

In `compute_event_metrics`, update signature:

```python
def compute_event_metrics(
    states_by_day: tuple[tuple[EventAgentState, ...], ...],
    messages: tuple[EventMessage, ...],
    *,
    memories: tuple[object, ...] = (),
    retrievals: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
```

Add fields to returned dict:

```python
"memory_count": len(memories),
"retrieval_count": len(retrievals),
"private_memory_count": sum(1 for memory in memories if getattr(memory, "private", False)),
"public_memory_count": sum(1 for memory in memories if not getattr(memory, "private", False)),
"mean_retrieved_memories_per_decision": (
    mean(len(row.get("retrieved", ())) for row in retrievals) if retrievals else 0.0
),
"mean_retrieval_score": (
    mean(
        item["score"]
        for row in retrievals
        for item in row.get("retrieved", ())
        if isinstance(item, dict) and isinstance(item.get("score"), (int, float))
    )
    if retrievals and any(row.get("retrieved") for row in retrievals)
    else 0.0
),
```

- [ ] **Step 5: Run tests**

Run:

```bash
/home/jhlee/repo/society-simulation/.venv/bin/python -m pytest tests/test_event_replay.py tests/test_event_metrics.py -q
/home/jhlee/repo/society-simulation/.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/society_simulation/event_replay.py src/society_simulation/event_metrics.py tests/test_event_replay.py tests/test_event_metrics.py
git commit -m "feat: write event memory replay metrics"
```

---

### Task 4: Policy Prompt Integration

**Files:**
- Modify: `src/society_simulation/event_policy.py`
- Modify: `tests/test_event_policy.py`

- [ ] **Step 1: Add failing prompt tests**

Add to `tests/test_event_policy.py`:

```python
from society_simulation.event_memory import SocialMemory


def test_openai_compatible_persona_policy_includes_retrieved_memories() -> None:
    captured: dict[str, object] = {}

    def transport(url, headers, payload, timeout_seconds):  # type: ignore[no-untyped-def]
        del url, headers, timeout_seconds
        captured["payload"] = payload
        return {
            "choices": [{"message": {"content": llm_decision_content()}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 40},
        }

    remembered = SocialMemory(
        memory_id="m1",
        agent_id="jisoo",
        day=1,
        kind="event_exposure",
        text="You remember a public-health report about asthma near schools.",
        source_id="public_health_report",
        source_type="event",
        channel="news_feed",
        related_agent_ids=(),
        related_event_ids=("public_health_report",),
        stance_signal=0.5,
        emotional_intensity=0.4,
        source_trust=0.8,
        identity_relevance=0.9,
        importance=0.8,
        private=False,
    )
    policy = OpenAICompatiblePersonaPolicy(
        model="cheap-chat",
        api_key="secret-key",
        base_url="https://example.test/v1",
        transport=transport,
    )

    policy.decide(profile(), state(), (exposure(),), day=2, retrieved_memories=(remembered,))

    prompt_text = json.dumps(captured["payload"])
    assert "Things you currently remember" in prompt_text
    assert "public-health report about asthma" in prompt_text
    assert "retrieval" not in prompt_text.lower()
```

Also add a mock policy test:

```python
def test_mock_persona_policy_accepts_retrieved_memories() -> None:
    policy = MockPersonaPolicy(response_style="silent")

    decision = policy.decide(profile(), state(), (exposure(),), day=1, retrieved_memories=())

    assert decision.messages == ()
```

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
/home/jhlee/repo/society-simulation/.venv/bin/python -m pytest tests/test_event_policy.py -q
```

Expected: fail because `decide()` does not accept `retrieved_memories`.

- [ ] **Step 3: Update policy signatures and prompt**

In `MockPersonaPolicy.decide` and `OpenAICompatiblePersonaPolicy.decide`, add:

```python
retrieved_memories: Sequence[object] = (),
```

Pass to `_event_user_message`.

Update `_event_prompt` and `_event_user_message` signatures with:

```python
retrieved_memories: Sequence[object] = (),
```

Inside `_event_user_message`, add:

```python
memory_lines = _memory_context_lines(retrieved_memories)
memory_context = (
    "Things you currently remember:\n"
    + "\n".join(f"- {line}" for line in memory_lines)
    + "\n"
    if memory_lines
    else ""
)
```

Place `memory_context` before `profile=...`.

Add helper:

```python
def _memory_context_lines(retrieved_memories: Sequence[object]) -> tuple[str, ...]:
    lines: list[str] = []
    for item in retrieved_memories:
        memory = getattr(item, "memory", item)
        text = getattr(memory, "text", None)
        if isinstance(text, str) and text.strip():
            lines.append(text.strip())
    return tuple(lines)
```

- [ ] **Step 4: Run tests**

Run:

```bash
/home/jhlee/repo/society-simulation/.venv/bin/python -m pytest tests/test_event_policy.py -q
/home/jhlee/repo/society-simulation/.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/society_simulation/event_policy.py tests/test_event_policy.py
git commit -m "feat: include social memories in event prompts"
```

---

### Task 5: Runner Memory Integration

**Files:**
- Modify: `src/society_simulation/event_runner.py`
- Modify: `tests/test_event_runner.py`

- [ ] **Step 1: Add failing runner tests**

Add to `tests/test_event_runner.py`:

```python
def test_event_runner_writes_empty_memory_artifacts_when_disabled(tmp_path: Path) -> None:
    data = valid_event_config(tmp_path)
    data["memory_retrieval"] = {"enabled": False, "limit": 3}
    data["output_dir"] = str(tmp_path / "memory-disabled")

    result = run_experiment(EventDrivenOpinionConfig.from_dict(data))

    assert result.metrics["memory_count"] == 0
    assert (result.output_dir / "memories.jsonl").read_text(encoding="utf-8") == ""
    assert (result.output_dir / "retrievals.jsonl").read_text(encoding="utf-8") == ""


def test_event_runner_records_memories_and_retrievals_when_enabled(tmp_path: Path) -> None:
    data = valid_event_config(tmp_path)
    data["memory_retrieval"] = {"enabled": True, "limit": 2}
    data["output_dir"] = str(tmp_path / "memory-enabled")

    result = run_experiment(EventDrivenOpinionConfig.from_dict(data))

    assert result.metrics["memory_count"] > 0
    assert result.metrics["retrieval_count"] > 0
    memory_rows = [
        json.loads(line)
        for line in (result.output_dir / "memories.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    retrieval_rows = [
        json.loads(line)
        for line in (result.output_dir / "retrievals.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert memory_rows
    assert retrieval_rows
    assert {"agent_id", "day", "query", "retrieved"} <= retrieval_rows[0].keys()
```

- [ ] **Step 2: Run focused runner tests and verify failure**

Run:

```bash
/home/jhlee/repo/society-simulation/.venv/bin/python -m pytest tests/test_event_runner.py -q
```

Expected: fail because memory artifacts are not produced.

- [ ] **Step 3: Implement runner integration**

In `event_runner.py`, import:

```python
from society_simulation.event_memory import (
    SocialMemory,
    build_memory_query,
    retrieve_memories,
)
```

Extend `EventRunResult`:

```python
memories: tuple[SocialMemory, ...] = ()
retrievals: tuple[dict[str, Any], ...] = ()
```

Inside `run_event_driven_opinion_dynamics`, create:

```python
memory_enabled = bool(config.memory_retrieval["enabled"])
memory_limit = int(config.memory_retrieval["limit"])
all_memories: list[SocialMemory] = []
all_retrievals: list[dict[str, Any]] = []
```

Before `policy.decide`, compute:

```python
agent_exposures = tuple(exposures_by_agent.get(profile.agent_id, []))
retrieved = ()
if memory_enabled:
    query = build_memory_query(
        agent_id=profile.agent_id,
        day=day,
        exposures=agent_exposures,
        affected_interests=profile.material_interests + profile.core_values,
    )
    retrieved = retrieve_memories(tuple(all_memories), query, limit=memory_limit)
    all_retrievals.append(
        {
            "agent_id": profile.agent_id,
            "day": day,
            "query": query.to_dict(),
            "retrieved": [item.to_dict() for item in retrieved],
        }
    )
```

Pass:

```python
retrieved_memories=retrieved,
```

to `policy.decide`.

After exposures are known, add event/message memories using helper functions. Implement local helpers:

```python
def _memory_from_exposure(
    *,
    exposure: EventExposure,
    agent_profile: EventAgentProfile,
    sequence: int,
) -> SocialMemory:
    related_event_ids = (exposure.source_id,) if exposure.source_type == "event" else ()
    related_agent_ids = (
        (exposure.source_id.split(":", 1)[0],)
        if exposure.source_type == "message" and ":" in exposure.source_id
        else ()
    )
    return SocialMemory(
        memory_id=f"{exposure.agent_id}:{exposure.day}:exposure:{sequence}",
        agent_id=exposure.agent_id,
        day=exposure.day,
        kind="social_message" if exposure.source_type == "message" else "event_exposure",
        text=exposure.content,
        source_id=exposure.source_id,
        source_type=exposure.source_type,
        channel=exposure.channel,
        related_agent_ids=related_agent_ids,
        related_event_ids=related_event_ids,
        stance_signal=0.0,
        emotional_intensity=_interest_overlap_score(exposure.content, agent_profile.core_values),
        source_trust=0.5,
        identity_relevance=_interest_overlap_score(
            exposure.content,
            agent_profile.material_interests + agent_profile.core_values,
        ),
        importance=max(
            0.3,
            _interest_overlap_score(
                exposure.content,
                agent_profile.material_interests + agent_profile.core_values,
            ),
        ),
        private=False,
    )
```

After `decision` validation, append private reasoning memory:

```python
all_memories.append(
    SocialMemory(
        memory_id=f"{profile.agent_id}:{day}:self_reasoning",
        agent_id=profile.agent_id,
        day=day,
        kind="self_reasoning",
        text=decision.state.last_private_reasoning,
        source_id=f"{profile.agent_id}:{day}:decision",
        source_type="self",
        channel="internal",
        related_agent_ids=(),
        related_event_ids=(),
        stance_signal=decision.state.private_stance,
        emotional_intensity=decision.state.salience,
        source_trust=1.0,
        identity_relevance=decision.state.salience,
        importance=decision.state.confidence,
        private=True,
    )
)
```

Only create memories when `memory_enabled` is true, so disabled mode preserves old behavior and empty memory artifacts.

When writing metrics and replay, pass:

```python
memories=tuple(all_memories),
retrievals=tuple(all_retrievals),
```

- [ ] **Step 4: Run tests**

Run:

```bash
/home/jhlee/repo/society-simulation/.venv/bin/python -m pytest tests/test_event_runner.py -q
/home/jhlee/repo/society-simulation/.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/society_simulation/event_runner.py tests/test_event_runner.py
git commit -m "feat: integrate social memory into event runner"
```

---

### Task 6: Experiment Configs and Final Verification

**Files:**
- Modify: `experiments/event_driven_congestion_pricing.json`
- Modify: `experiments/event_driven_congestion_pricing_gpt54_mini_pilot.json`
- Modify: `docs/research/2026-06-19-event-driven-congestion-pricing-gpt54-mini-pilot.md`
- Modify: `docs/research/2026-06-19-event-driven-congestion-pricing-gpt54-mini-pilot.ko.md`

- [ ] **Step 1: Add explicit disabled memory config to existing experiments**

Add near `update_policy` in both experiment JSON files:

```json
"memory_retrieval": {
  "enabled": false,
  "limit": 5
},
```

This preserves reported pilot results and makes the ablation status explicit.

- [ ] **Step 2: Add report note**

In both paid pilot reports, add one sentence to limitations:

English:

```markdown
- no social memory retrieval layer; agents saw only current exposures, previous-day messages, and compact state summaries.
```

Korean:

```markdown
- 사회적 기억 검색 계층이 없었고, 에이전트는 현재 노출, 전날 메시지, 압축된 상태 요약만 보았다.
```

- [ ] **Step 3: Run full verification**

Run:

```bash
/home/jhlee/repo/society-simulation/.venv/bin/python -m pytest -q
git diff --check
```

Expected:

- `454+ passed`;
- `git diff --check` exits `0`.

- [ ] **Step 4: Commit**

Run:

```bash
git add experiments/event_driven_congestion_pricing.json experiments/event_driven_congestion_pricing_gpt54_mini_pilot.json docs/research/2026-06-19-event-driven-congestion-pricing-gpt54-mini-pilot.md docs/research/2026-06-19-event-driven-congestion-pricing-gpt54-mini-pilot.ko.md
git commit -m "docs: mark event pilot memory ablation status"
```

---

## Self-Review Checklist

- Spec coverage:
  - memory model: Task 1;
  - retrieval scoring: Task 1;
  - config flag: Task 2;
  - replay artifacts: Task 3;
  - metrics: Task 3;
  - prompt integration: Task 4;
  - runner integration: Task 5;
  - experiment/report ablation status: Task 6.

- Placeholder scan:
  - no unresolved markers or unspecified test steps.

- Type consistency:
  - `SocialMemory`, `MemoryQuery`, and `RetrievedMemory` are defined before use;
  - `retrieved_memories` is passed to policy as a sequence of `RetrievedMemory`;
  - replay writes memory objects through `to_dict()`.

## Execution Handoff

Recommended execution mode: **Subagent-Driven**. The tasks are mostly independent after Task 1 and small enough for one subagent per task with review between tasks.
