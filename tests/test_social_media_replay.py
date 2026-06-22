import json

from society_simulation.social_media_config import InstagramSocialDynamicsConfig
from society_simulation.social_media_ads import AdImpression
from society_simulation.social_media_replay import SocialMediaReplayWriter
from society_simulation.social_media_seed import build_initial_world
from tests.test_social_media_config import valid_social_media_config


def test_replay_writer_writes_core_artifacts(tmp_path) -> None:
    data = valid_social_media_config()
    data["seed"] = 1
    data["scenario_name"] = "replay_test"
    data["ticks"] = 1
    data["num_users"] = 3
    data["historical_posts_per_user"] = 1
    data["feed_size"] = 2
    data["topics"] = ["transit"]
    data["seed_generator"] = dict(data["seed_generator"], mean_following=1)  # type: ignore[arg-type]
    data["memory_retrieval"] = {"enabled": False, "limit": 5}
    data["output_dir"] = str(tmp_path / "run")
    config = InstagramSocialDynamicsConfig.from_dict(data)
    world = build_initial_world(config)

    output_dir = SocialMediaReplayWriter(config).write(
        final_world=world,
        initial_edges=world.follow_edges,
        feed_items=(),
        actions=(),
        dm_messages=(),
        states_by_tick=(world.states,),
        metrics={"action_count": 0},
        llm_decisions=(),
    )

    assert (output_dir / "config.json").exists()
    assert (output_dir / "users.jsonl").exists()
    assert json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))[
        "action_count"
    ] == 0


def test_replay_writer_writes_ad_impressions(tmp_path) -> None:
    data = valid_social_media_config()
    data["seed"] = 1
    data["scenario_name"] = "replay_ad_test"
    data["ticks"] = 1
    data["num_users"] = 3
    data["historical_posts_per_user"] = 1
    data["feed_size"] = 2
    data["topics"] = ["transit"]
    data["seed_generator"] = dict(data["seed_generator"], mean_following=1)  # type: ignore[arg-type]
    data["memory_retrieval"] = {"enabled": False, "limit": 5}
    data["output_dir"] = str(tmp_path / "run")
    config = InstagramSocialDynamicsConfig.from_dict(data)
    world = build_initial_world(config)

    output_dir = SocialMediaReplayWriter(config).write(
        final_world=world,
        initial_edges=world.follow_edges,
        feed_items=(),
        actions=(),
        dm_messages=(),
        states_by_tick=(world.states,),
        metrics={"action_count": 0},
        llm_decisions=(),
        ad_impressions=(
            AdImpression(
                tick=1,
                viewer_id=1,
                campaign_id="maple_3rd_opening",
                advertiser_id=0,
                post_id="ad-maple_3rd_opening",
                targeting="broad",
                source_reason="sponsored_broad",
                seen_count=1,
                frequency_cap=2,
                topic="transit",
            ),
        ),
    )

    rows = [
        json.loads(line)
        for line in (output_dir / "ad_impressions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [
        {
            "advertiser_id": 0,
            "campaign_id": "maple_3rd_opening",
            "frequency_cap": 2,
            "post_id": "ad-maple_3rd_opening",
            "seen_count": 1,
            "source_reason": "sponsored_broad",
            "targeting": "broad",
            "tick": 1,
            "topic": "transit",
            "viewer_id": 1,
        }
    ]
