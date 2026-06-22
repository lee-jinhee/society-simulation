import csv
import json
from pathlib import Path

from society_simulation.sweep_artifacts import SweepRunRecord, write_sweep_artifacts
from society_simulation.sweep_config import expand_sweep, parse_sweep_config
from tests.test_sweep_config import valid_sweep_dict


def test_write_sweep_artifacts_writes_manifest_csv_and_summary_json(
    tmp_path: Path,
) -> None:
    sweep = parse_sweep_config(valid_sweep_dict(tmp_path))
    planned_runs = expand_sweep(sweep)
    records = (
        SweepRunRecord(
            run_id=planned_runs[0].run_id,
            labels=planned_runs[0].labels,
            experiment_name="network_herding",
            output_dir=planned_runs[0].config["output_dir"],
            status="completed",
            error=None,
            metrics={
                "final_action_counts": {"A": 4, "B": 2},
                "final_a_fraction": 0.6666666667,
                "consensus_reached": False,
                "consensus_action": None,
                "time_to_consensus": None,
                "polarization_index": 0.2,
                "opinion_variance": 0.1,
                "mean_belief": 0.62,
                "edge_disagreement_rate": 0.25,
                "component_count": 1,
            },
        ),
        SweepRunRecord(
            run_id=planned_runs[1].run_id,
            labels=planned_runs[1].labels,
            experiment_name="network_herding",
            output_dir=planned_runs[1].config["output_dir"],
            status="failed",
            error="boom",
            metrics={},
        ),
    )

    paths = write_sweep_artifacts(sweep, planned_runs, records)

    assert paths.output_dir == Path(sweep.output_dir)
    assert paths.manifest_path.exists()
    assert paths.summary_csv_path.exists()
    assert paths.summary_json_path.exists()
    assert paths.sweep_config_path.exists()

    manifest_lines = paths.manifest_path.read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == len(planned_runs)
    assert json.loads(manifest_lines[0])["status"] == "completed"
    assert json.loads(manifest_lines[1])["error"] == "boom"
    assert json.loads(manifest_lines[2])["status"] == "pending"

    with paths.summary_csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == [
        "run_id",
        "seed",
        "initial_a",
        "topology",
        "experiment_name",
        "output_dir",
        "status",
        "error",
        "final_action_counts_A",
        "final_action_counts_B",
        "final_a_fraction",
        "consensus_reached",
        "consensus_action",
        "time_to_consensus",
        "polarization_index",
        "opinion_variance",
        "mean_belief",
        "edge_disagreement_rate",
        "component_count",
        "final_private_stance_mean",
        "final_public_stance_mean",
        "final_private_public_gap",
        "final_private_stance_variance",
        "final_public_stance_variance",
        "final_mean_confidence",
        "final_mean_salience",
        "message_count",
        "agent_count",
        "day_count",
        "experiment_family",
        "user_count",
        "post_count",
        "feed_impression_count",
        "action_count",
        "like_count",
        "dm_count",
        "follow_count",
        "unfollow_count",
        "initial_follow_edge_count",
        "final_follow_edge_count",
        "follow_edge_delta",
        "new_follow_edge_count",
        "removed_follow_edge_count",
        "final_stance_mean",
        "final_stance_variance",
        "exposure_diversity",
        "states_recorded",
        "paid_impression_count",
        "unique_paid_reach",
        "organic_ad_impression_count",
        "unique_organic_ad_reach",
        "unique_total_ad_reach",
        "relevant_paid_reach",
        "relevant_total_reach",
        "mean_ad_frequency",
        "max_ad_frequency",
        "frequency_cap_hit_count",
        "ad_like_count",
        "advertiser_follow_count",
        "ad_dm_count",
        "ad_generated_post_count",
        "ad_negative_action_count",
        "paid_to_organic_spillover_rate",
        "ad_delivery_exhausted_budget",
        "ad_delivery_remaining_budget",
        "burn_in_action_mean",
        "burn_in_follow_churn",
        "burn_in_exposure_diversity",
    ]
    assert rows[0]["run_id"] == planned_runs[0].run_id
    assert rows[0]["status"] == "completed"
    assert rows[0]["final_action_counts_A"] == "4"
    assert rows[1]["status"] == "failed"
    assert rows[1]["error"] == "boom"
    assert rows[1]["final_a_fraction"] == ""

    summary = json.loads(paths.summary_json_path.read_text(encoding="utf-8"))
    assert summary["sweep_name"] == "network_topology_sweep"
    assert summary["runs"] == len(planned_runs)
    assert summary["completed"] == 1
    assert summary["failed"] == 1
    assert summary["metric_means"]["final_a_fraction"] == 0.6666666667
    assert summary["groups"]["seed"]["1"]["completed"] == 1


def test_write_sweep_artifacts_includes_all_planned_runs_when_records_missing(
    tmp_path: Path,
) -> None:
    sweep = parse_sweep_config(valid_sweep_dict(tmp_path))
    planned_runs = expand_sweep(sweep)

    paths = write_sweep_artifacts(sweep, planned_runs, records=())

    with paths.summary_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(planned_runs)
    assert rows[0]["status"] == "pending"


def test_write_sweep_artifacts_writes_sweep_config_json(tmp_path: Path) -> None:
    sweep = parse_sweep_config(valid_sweep_dict(tmp_path))
    planned_runs = expand_sweep(sweep)

    paths = write_sweep_artifacts(sweep, planned_runs, records=())

    text = paths.sweep_config_path.read_text(encoding="utf-8")
    assert json.loads(text) == sweep.to_dict()
    assert text.splitlines()[1] == '  "base_config": {'


def test_write_sweep_artifacts_serializes_record_output_dir_paths(
    tmp_path: Path,
) -> None:
    sweep = parse_sweep_config(valid_sweep_dict(tmp_path))
    planned_runs = expand_sweep(sweep)
    output_dir = Path(planned_runs[0].config["output_dir"])

    paths = write_sweep_artifacts(
        sweep,
        planned_runs,
        records=(
            SweepRunRecord(
                run_id=planned_runs[0].run_id,
                labels=planned_runs[0].labels,
                experiment_name="network_herding",
                output_dir=output_dir,
                status="completed",
                error=None,
                metrics={},
            ),
        ),
    )

    manifest_row = json.loads(paths.manifest_path.read_text(encoding="utf-8").splitlines()[0])
    assert manifest_row["output_dir"] == str(output_dir)
    assert isinstance(manifest_row["output_dir"], str)
    with paths.summary_csv_path.open(newline="", encoding="utf-8") as handle:
        csv_row = next(csv.DictReader(handle))
    assert csv_row["output_dir"] == str(output_dir)


def test_summary_metric_means_use_completed_rows_and_continuous_fields_only(
    tmp_path: Path,
) -> None:
    sweep = parse_sweep_config(valid_sweep_dict(tmp_path))
    planned_runs = expand_sweep(sweep)
    records = (
        SweepRunRecord(
            run_id=planned_runs[0].run_id,
            labels=planned_runs[0].labels,
            experiment_name="network_herding",
            output_dir=planned_runs[0].config["output_dir"],
            status="completed",
            error=None,
            metrics={
                "final_action_counts": {"A": 4, "B": 2},
                "final_a_fraction": 0.75,
                "consensus_reached": True,
                "component_count": 1,
            },
        ),
        SweepRunRecord(
            run_id=planned_runs[1].run_id,
            labels=planned_runs[1].labels,
            experiment_name="network_herding",
            output_dir=planned_runs[1].config["output_dir"],
            status="failed",
            error="boom",
            metrics={
                "final_action_counts": {"A": 100, "B": 100},
                "final_a_fraction": 0.0,
                "consensus_reached": False,
                "component_count": 9,
            },
        ),
    )

    paths = write_sweep_artifacts(sweep, planned_runs, records)

    summary = json.loads(paths.summary_json_path.read_text(encoding="utf-8"))
    assert summary["metric_means"]["final_a_fraction"] == 0.75
    assert summary["metric_means"]["component_count"] == 1.0
    assert "consensus_reached" not in summary["metric_means"]
    assert "final_action_counts_A" not in summary["metric_means"]
    assert summary["groups"]["seed"]["1"]["metric_means"]["final_a_fraction"] == 0.75
    assert summary["groups"]["seed"]["1"]["metric_means"]["component_count"] == 1.0


def test_write_sweep_artifacts_accepts_action_counts_metric_fallback(
    tmp_path: Path,
) -> None:
    sweep = parse_sweep_config(valid_sweep_dict(tmp_path))
    planned_runs = expand_sweep(sweep)
    records = (
        SweepRunRecord(
            run_id=planned_runs[0].run_id,
            labels=planned_runs[0].labels,
            experiment_name="network_herding",
            output_dir=planned_runs[0].config["output_dir"],
            status="completed",
            error=None,
            metrics={"action_counts": {"A": 1, "B": 2}},
        ),
    )

    paths = write_sweep_artifacts(sweep, planned_runs, records)

    with paths.summary_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["final_action_counts_A"] == "1"
    assert rows[0]["final_action_counts_B"] == "2"


def test_write_sweep_artifacts_includes_event_driven_opinion_metrics(
    tmp_path: Path,
) -> None:
    sweep = parse_sweep_config(valid_sweep_dict(tmp_path))
    planned_runs = expand_sweep(sweep)
    records = (
        SweepRunRecord(
            run_id=planned_runs[0].run_id,
            labels=planned_runs[0].labels,
            experiment_name="event_driven_opinion_dynamics",
            output_dir=planned_runs[0].config["output_dir"],
            status="completed",
            error=None,
            metrics={
                "final_private_stance_mean": 0.2,
                "final_public_stance_mean": 0.1,
                "final_private_public_gap": 0.1,
                "final_private_stance_variance": 0.04,
                "final_public_stance_variance": 0.03,
                "final_mean_confidence": 0.7,
                "final_mean_salience": 0.8,
                "message_count": 5,
                "agent_count": 8,
                "day_count": 7,
            },
        ),
    )

    paths = write_sweep_artifacts(sweep, planned_runs, records)

    manifest_row = json.loads(paths.manifest_path.read_text(encoding="utf-8").splitlines()[0])
    assert manifest_row["final_private_stance_mean"] == 0.2
    assert manifest_row["final_public_stance_mean"] == 0.1
    assert manifest_row["final_private_public_gap"] == 0.1
    assert manifest_row["final_private_stance_variance"] == 0.04
    assert manifest_row["final_public_stance_variance"] == 0.03
    assert manifest_row["final_mean_confidence"] == 0.7
    assert manifest_row["final_mean_salience"] == 0.8
    assert manifest_row["message_count"] == 5
    assert manifest_row["agent_count"] == 8
    assert manifest_row["day_count"] == 7

    with paths.summary_csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    for field in (
        "final_private_stance_mean",
        "final_public_stance_mean",
        "final_private_public_gap",
        "final_private_stance_variance",
        "final_public_stance_variance",
        "final_mean_confidence",
        "final_mean_salience",
        "message_count",
        "agent_count",
        "day_count",
    ):
        assert field in (reader.fieldnames or [])
    assert rows[0]["final_private_stance_mean"] == "0.2"
    assert rows[0]["final_private_stance_variance"] == "0.04"
    assert rows[0]["final_public_stance_variance"] == "0.03"
    assert rows[0]["final_mean_salience"] == "0.8"
    assert rows[0]["message_count"] == "5"
    assert rows[0]["agent_count"] == "8"
    assert rows[0]["day_count"] == "7"

    summary = json.loads(paths.summary_json_path.read_text(encoding="utf-8"))
    assert summary["metric_means"]["final_private_stance_mean"] == 0.2
    assert summary["metric_means"]["final_public_stance_mean"] == 0.1
    assert summary["metric_means"]["final_private_public_gap"] == 0.1
    assert summary["metric_means"]["final_private_stance_variance"] == 0.04
    assert summary["metric_means"]["final_public_stance_variance"] == 0.03
    assert summary["metric_means"]["final_mean_confidence"] == 0.7
    assert summary["metric_means"]["final_mean_salience"] == 0.8
    assert summary["metric_means"]["message_count"] == 5.0
    assert summary["metric_means"]["agent_count"] == 8.0
    assert summary["metric_means"]["day_count"] == 7.0


def test_write_sweep_artifacts_includes_instagram_social_metrics(
    tmp_path: Path,
) -> None:
    sweep = parse_sweep_config(valid_sweep_dict(tmp_path))
    planned_runs = expand_sweep(sweep)
    records = (
        SweepRunRecord(
            run_id=planned_runs[0].run_id,
            labels=planned_runs[0].labels,
            experiment_name="instagram_social_dynamics",
            output_dir=planned_runs[0].config["output_dir"],
            status="completed",
            error=None,
            metrics={
                "experiment_family": "instagram_social_dynamics",
                "user_count": 12,
                "post_count": 24,
                "feed_impression_count": 192,
                "action_count": 48,
                "like_count": 20,
                "dm_count": 7,
                "follow_count": 5,
                "unfollow_count": 2,
                "initial_follow_edge_count": 40,
                "final_follow_edge_count": 43,
                "follow_edge_delta": 3,
                "new_follow_edge_count": 5,
                "removed_follow_edge_count": 2,
                "final_stance_mean": 0.18,
                "final_stance_variance": 0.09,
                "exposure_diversity": 2.75,
                "states_recorded": 60,
            },
        ),
    )

    paths = write_sweep_artifacts(sweep, planned_runs, records)

    manifest_row = json.loads(paths.manifest_path.read_text(encoding="utf-8").splitlines()[0])
    assert manifest_row["feed_impression_count"] == 192
    assert manifest_row["follow_edge_delta"] == 3
    assert manifest_row["exposure_diversity"] == 2.75

    with paths.summary_csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert "feed_impression_count" in (reader.fieldnames or [])
    assert "follow_edge_delta" in (reader.fieldnames or [])
    assert "exposure_diversity" in (reader.fieldnames or [])
    assert rows[0]["feed_impression_count"] == "192"
    assert rows[0]["follow_edge_delta"] == "3"
    assert rows[0]["exposure_diversity"] == "2.75"

    summary = json.loads(paths.summary_json_path.read_text(encoding="utf-8"))
    assert summary["metric_means"]["feed_impression_count"] == 192.0
    assert summary["metric_means"]["action_count"] == 48.0
    assert summary["metric_means"]["follow_edge_delta"] == 3.0
    assert summary["metric_means"]["final_stance_variance"] == 0.09
    assert summary["metric_means"]["exposure_diversity"] == 2.75


def test_write_sweep_artifacts_includes_instagram_ad_metrics(
    tmp_path: Path,
) -> None:
    sweep = parse_sweep_config(valid_sweep_dict(tmp_path))
    planned_runs = expand_sweep(sweep)
    records = (
        SweepRunRecord(
            run_id=planned_runs[0].run_id,
            labels=planned_runs[0].labels,
            experiment_name="instagram_social_dynamics",
            output_dir=planned_runs[0].config["output_dir"],
            status="completed",
            error=None,
            metrics={
                "paid_impression_count": 20,
                "unique_paid_reach": 12,
                "organic_ad_impression_count": 4,
                "unique_organic_ad_reach": 4,
                "unique_total_ad_reach": 14,
                "relevant_paid_reach": 6,
                "relevant_total_reach": 6,
                "mean_ad_frequency": 1.666667,
                "max_ad_frequency": 2,
                "frequency_cap_hit_count": 4,
                "ad_like_count": 3,
                "advertiser_follow_count": 1,
                "ad_dm_count": 1,
                "ad_generated_post_count": 1,
                "ad_negative_action_count": 0,
                "paid_to_organic_spillover_rate": 0.2,
                "ad_delivery_exhausted_budget": True,
                "ad_delivery_remaining_budget": 0,
                "burn_in_action_mean": 5.0,
                "burn_in_follow_churn": 1,
                "burn_in_exposure_diversity": 2.5,
            },
        ),
    )

    paths = write_sweep_artifacts(sweep, planned_runs, records)

    manifest_row = json.loads(paths.manifest_path.read_text(encoding="utf-8").splitlines()[0])
    assert manifest_row["paid_impression_count"] == 20
    assert manifest_row["unique_total_ad_reach"] == 14
    assert manifest_row["ad_delivery_exhausted_budget"] is True

    with paths.summary_csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert "paid_impression_count" in (reader.fieldnames or [])
    assert "unique_total_ad_reach" in (reader.fieldnames or [])
    assert "burn_in_exposure_diversity" in (reader.fieldnames or [])
    assert rows[0]["paid_impression_count"] == "20"
    assert rows[0]["ad_delivery_exhausted_budget"] == "True"

    summary = json.loads(paths.summary_json_path.read_text(encoding="utf-8"))
    assert summary["metric_means"]["paid_impression_count"] == 20.0
    assert summary["metric_means"]["unique_total_ad_reach"] == 14.0
    assert summary["metric_means"]["paid_to_organic_spillover_rate"] == 0.2
    assert "ad_delivery_exhausted_budget" not in summary["metric_means"]
