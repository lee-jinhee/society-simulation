from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from society_simulation.social_media_ads import AdImpression
from society_simulation.social_media_config import InstagramSocialDynamicsConfig
from society_simulation.social_media_models import (
    DirectMessage,
    FeedItem,
    FollowEdge,
    PlatformAction,
    SocialMediaUserState,
    SocialMediaWorld,
)


class SocialMediaReplayWriter:
    def __init__(self, config: InstagramSocialDynamicsConfig) -> None:
        self.config = config

    def write(
        self,
        *,
        final_world: SocialMediaWorld,
        initial_edges: tuple[FollowEdge, ...],
        feed_items: tuple[FeedItem, ...],
        actions: tuple[PlatformAction, ...],
        dm_messages: tuple[DirectMessage, ...],
        states_by_tick: tuple[tuple[SocialMediaUserState, ...], ...],
        metrics: dict[str, Any],
        llm_decisions: tuple[dict[str, Any], ...],
        ad_impressions: tuple[AdImpression, ...] = (),
    ) -> Path:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(output_dir / "config.json", self.config.to_dict())
        self._write_jsonl(
            output_dir / "users.jsonl",
            tuple(profile.to_dict() for profile in final_world.profiles),
        )
        self._write_jsonl(
            output_dir / "posts.jsonl",
            tuple(post.to_dict() for post in final_world.posts),
        )
        self._write_jsonl(
            output_dir / "follow_edges_initial.jsonl",
            tuple(edge.to_dict() for edge in initial_edges),
        )
        self._write_jsonl(
            output_dir / "follow_edges_final.jsonl",
            tuple(edge.to_dict() for edge in final_world.follow_edges),
        )
        self._write_jsonl(
            output_dir / "feed_impressions.jsonl",
            tuple(item.to_dict() for item in feed_items),
        )
        self._write_jsonl(
            output_dir / "ad_impressions.jsonl",
            tuple(impression.to_dict() for impression in ad_impressions),
        )
        self._write_jsonl(
            output_dir / "actions.jsonl",
            tuple(action.to_dict() for action in actions),
        )
        self._write_jsonl(
            output_dir / "dm_messages.jsonl",
            tuple(message.to_dict() for message in dm_messages),
        )
        self._write_jsonl(
            output_dir / "user_states.jsonl",
            tuple(state.to_dict() for tick_states in states_by_tick for state in tick_states),
        )
        self._write_json(output_dir / "metrics.json", metrics)
        self._write_jsonl(output_dir / "llm_decisions.jsonl", llm_decisions)
        self._write_summary(output_dir / "summary.md", metrics)
        return output_dir

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_jsonl(self, path: Path, rows: tuple[dict[str, Any], ...]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")

    def _write_summary(self, path: Path, metrics: dict[str, Any]) -> None:
        lines = [
            f"# {self.config.scenario_name}",
            "",
            f"- experiment_name: `{self.config.experiment_name}`",
            f"- user_count: `{metrics.get('user_count')}`",
            f"- action_count: `{metrics.get('action_count')}`",
            f"- feed_impression_count: `{metrics.get('feed_impression_count')}`",
            f"- action_counts: `{metrics.get('action_counts')}`",
            f"- final_follow_edge_count: `{metrics.get('final_follow_edge_count')}`",
            f"- final_stance_mean: `{metrics.get('final_stance_mean')}`",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
