import json
from pathlib import Path

import pytest

from society_simulation.config import load_config
from society_simulation.runner import run_experiment
from society_simulation.social_media_runner import run_instagram_social_dynamics
from tests.test_social_media_config import valid_social_media_config


def _config(tmp_path: Path) -> dict[str, object]:
    data = valid_social_media_config()
    data["seed"] = 3
    data["scenario_name"] = "runner_test"
    data["ticks"] = 2
    data["num_users"] = 5
    data["historical_posts_per_user"] = 1
    data["feed_size"] = 2
    data["update_policy"] = {"type": "mock_social", "response_style": "balanced"}
    data["output_dir"] = str(tmp_path / "run")
    data["seed_generator"] = dict(data["seed_generator"], mean_following=2)  # type: ignore[arg-type]
    return data


def test_run_instagram_social_dynamics_writes_actions(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config(tmp_path)), encoding="utf-8")
    config = load_config(config_path)

    result = run_instagram_social_dynamics(config)

    assert result.metrics["action_count"] > 0
    assert result.output_dir == tmp_path / "run"


def test_runner_dispatches_from_generic_run_experiment(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config(tmp_path)), encoding="utf-8")
    config = load_config(config_path)

    result = run_experiment(config)

    assert "like_post" in result.metrics["action_counts"]


def test_llm_policy_requires_configured_api_key_env(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    data = _config(tmp_path)
    data["update_policy"] = {
        "type": "llm",
        "provider": "openai_compatible",
        "model": "cheap-chat",
        "api_key_env": "MISSING_SOCIAL_MEDIA_KEY",
        "max_estimated_cost_usd": 1.0,
    }
    config_path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.delenv("MISSING_SOCIAL_MEDIA_KEY", raising=False)
    config = load_config(config_path)

    with pytest.raises(ValueError, match="MISSING_SOCIAL_MEDIA_KEY environment variable is required"):
        run_instagram_social_dynamics(config)
