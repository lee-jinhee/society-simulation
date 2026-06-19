from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

from society_simulation.event_models import EventAgentProfile, EventRelationship, OpinionEvent


def _require_mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _require_free_form_mapping(value: object, field: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _require_field(data: dict[str, object], field: str, path: str) -> object:
    if field not in data:
        raise ValueError(f"{path} is required")
    return data[field]


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _require_non_empty_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_sequence(value: object, field: str) -> list[object] | tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list")
    return value


def _unsupported_json_value_error() -> ValueError:
    return ValueError("free-form config values must contain only JSON-compatible values")


def _freeze_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        copied: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("object keys must be strings")
            copied[key] = _freeze_json_value(item)
        return MappingProxyType(copied)
    if isinstance(value, tuple):
        return tuple(_freeze_json_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise _unsupported_json_value_error()
        return value
    raise _unsupported_json_value_error()


def _freeze_json_mapping(value: object, field: str) -> Mapping[str, object]:
    frozen = _freeze_json_value(_require_free_form_mapping(value, field))
    if not isinstance(frozen, Mapping):
        raise ValueError(f"{field} must be an object")
    return frozen


def _to_json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        data: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("object keys must be strings")
            data[key] = _to_json_ready(item)
        return data
    if isinstance(value, tuple):
        return [_to_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_to_json_ready(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise _unsupported_json_value_error()
        return value
    raise _unsupported_json_value_error()


def _to_json_ready_mapping(value: Mapping[str, object]) -> dict[str, object]:
    copied = _to_json_ready(value)
    if not isinstance(copied, dict):
        raise ValueError("object must be an object")
    return copied


def _parse_agents(value: object) -> tuple[EventAgentProfile, ...]:
    return tuple(
        EventAgentProfile.from_dict(_require_mapping(item, f"agents[{index}]"))
        for index, item in enumerate(_require_sequence(value, "agents"))
    )


def _parse_relationships(value: object) -> tuple[EventRelationship, ...]:
    return tuple(
        EventRelationship.from_dict(_require_mapping(item, f"relationships[{index}]"))
        for index, item in enumerate(_require_sequence(value, "relationships"))
    )


def _parse_events(value: object) -> tuple[OpinionEvent, ...]:
    return tuple(
        OpinionEvent.from_dict(_require_mapping(item, f"events[{index}]"))
        for index, item in enumerate(_require_sequence(value, "events"))
    )


def _parse_channels(value: object) -> tuple[Mapping[str, object], ...]:
    return tuple(
        _freeze_json_mapping(_require_mapping(item, f"channels[{index}]"), f"channels[{index}]")
        for index, item in enumerate(_require_sequence(value, "channels"))
    )


def _parse_update_policy(value: object) -> Mapping[str, object]:
    return _normalize_update_policy(_require_mapping(value, "update_policy"))


def _normalize_update_policy(value: object) -> Mapping[str, object]:
    policy = dict(_require_free_form_mapping(value, "update_policy"))
    policy_type = _require_non_empty_str(
        _require_field(policy, "type", "update_policy.type"),
        "update_policy.type",
    )
    if policy_type == "mock_persona" and "response_style" not in policy:
        policy["response_style"] = "balanced"
    return _freeze_json_mapping(policy, "update_policy")


@dataclass(frozen=True)
class EventDrivenOpinionConfig:
    experiment_name: str
    seed: int
    scenario_name: str
    days: int
    agents: tuple[EventAgentProfile, ...]
    relationships: tuple[EventRelationship, ...]
    events: tuple[OpinionEvent, ...]
    channels: tuple[Mapping[str, object], ...]
    update_policy: Mapping[str, object]
    output_dir: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "channels",
            tuple(_freeze_json_mapping(channel, "channel") for channel in self.channels),
        )
        object.__setattr__(self, "update_policy", _normalize_update_policy(self.update_policy))

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> EventDrivenOpinionConfig:
        data = _require_mapping(data, "event config")
        return cls(
            experiment_name=_require_non_empty_str(
                _require_field(data, "experiment_name", "experiment_name"),
                "experiment_name",
            ),
            seed=_require_int(_require_field(data, "seed", "seed"), "seed"),
            scenario_name=_require_non_empty_str(
                _require_field(data, "scenario_name", "scenario_name"),
                "scenario_name",
            ),
            days=_require_int(_require_field(data, "days", "days"), "days"),
            agents=_parse_agents(_require_field(data, "agents", "agents")),
            relationships=_parse_relationships(
                _require_field(data, "relationships", "relationships")
            ),
            events=_parse_events(_require_field(data, "events", "events")),
            channels=_parse_channels(_require_field(data, "channels", "channels")),
            update_policy=_parse_update_policy(
                _require_field(data, "update_policy", "update_policy")
            ),
            output_dir=_require_non_empty_str(
                _require_field(data, "output_dir", "output_dir"),
                "output_dir",
            ),
        )

    def validate(self) -> None:
        if self.experiment_name != "event_driven_opinion_dynamics":
            raise ValueError("unsupported experiment_name")
        if self.days <= 0:
            raise ValueError("days must be positive")
        if not self.agents:
            raise ValueError("agents must not be empty")

        agent_ids = {agent.agent_id for agent in self.agents}
        if len(agent_ids) != len(self.agents):
            raise ValueError("agent ids must be unique")

        channel_ids = self._validate_channel_ids()
        for relationship in self.relationships:
            if relationship.source_agent_id not in agent_ids:
                raise ValueError("relationship source_agent_id is not in agents")
            if relationship.target_agent_id not in agent_ids:
                raise ValueError("relationship target_agent_id is not in agents")
            for channel in relationship.channels:
                if channel not in channel_ids:
                    raise ValueError("relationship channel is not in channels")

        event_ids = {event.event_id for event in self.events}
        if len(event_ids) != len(self.events):
            raise ValueError("event ids must be unique")
        for event in self.events:
            if not 0 <= event.day <= self.days:
                raise ValueError("event day must be between 0 and days")

        self._validate_update_policy()

    def _validate_channel_ids(self) -> set[str]:
        channel_ids: set[str] = set()
        for channel in self.channels:
            channel_id = channel.get("channel_id")
            if not isinstance(channel_id, str) or not channel_id.strip():
                raise ValueError("channel_id must be a non-empty string")
            if channel_id in channel_ids:
                raise ValueError("channel ids must be unique")
            channel_ids.add(channel_id)
        return channel_ids

    def _validate_update_policy(self) -> None:
        policy_type = self.update_policy.get("type")
        if policy_type == "mock_persona":
            response_style = self.update_policy.get("response_style", "balanced")
            if response_style not in ("balanced", "silent", "reactive"):
                raise ValueError("unsupported mock_persona response_style")
            return

        if policy_type == "llm":
            model = self.update_policy.get("model")
            if not isinstance(model, str) or not model.strip():
                raise ValueError("llm.model must be a non-empty string")
            return

        raise ValueError("unsupported event update_policy type")

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "scenario_name": self.scenario_name,
            "days": self.days,
            "agents": [agent.to_dict() for agent in self.agents],
            "relationships": [relationship.to_dict() for relationship in self.relationships],
            "events": [event.to_dict() for event in self.events],
            "channels": [_to_json_ready_mapping(channel) for channel in self.channels],
            "update_policy": _to_json_ready_mapping(self.update_policy),
            "output_dir": self.output_dir,
        }
