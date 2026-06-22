from society_simulation.social_media_config import InstagramSocialDynamicsConfig
from society_simulation.social_media_seed import build_initial_world
from tests.test_social_media_config import valid_social_media_config


def _config(seed: int = 11) -> InstagramSocialDynamicsConfig:
    data = valid_social_media_config()
    data["seed"] = seed
    data["scenario_name"] = "seed_test"
    data["num_users"] = 6
    data["historical_posts_per_user"] = 2
    data["output_dir"] = "runs/seed_test"
    data["seed_generator"] = dict(data["seed_generator"], mean_following=2)  # type: ignore[arg-type]
    return InstagramSocialDynamicsConfig.from_dict(data)


def test_seed_generator_is_deterministic() -> None:
    first = build_initial_world(_config(seed=11))
    second = build_initial_world(_config(seed=11))

    assert [profile.to_dict() for profile in first.profiles] == [
        profile.to_dict() for profile in second.profiles
    ]
    assert [edge.to_dict() for edge in first.follow_edges] == [
        edge.to_dict() for edge in second.follow_edges
    ]
    assert [post.to_dict() for post in first.posts] == [
        post.to_dict() for post in second.posts
    ]


def test_seed_generator_creates_historical_posts_for_each_user() -> None:
    world = build_initial_world(_config())

    assert len(world.profiles) == 6
    assert len(world.states) == 6
    assert len(world.posts) == 12
    assert {post.author_id for post in world.posts} == {0, 1, 2, 3, 4, 5}


def test_seed_generator_has_no_self_follows() -> None:
    world = build_initial_world(_config())

    assert all(edge.follower_id != edge.followed_id for edge in world.follow_edges)
