# Sweep Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic `analyze` command that turns sweep output artifacts into a compact research report and grouped metric summaries.

**Architecture:** Add a read-only analysis layer beside the existing sweep runner. `sweep_analysis.py` loads and validates sweep artifacts, computes per-factor grouped summaries, and derives simple topline rankings. `sweep_analysis_artifacts.py` writes Markdown, CSV, and JSON outputs. The CLI delegates to these modules without changing existing `run` or `sweep` output.

**Tech Stack:** Python 3.11+, standard library only at runtime, `pytest` for tests, existing `society_simulation` package structure.

---

## File Structure

- Create `src/society_simulation/sweep_analysis.py`
  - Load `sweep_config.json`, `summary.csv`, `summary.json`, and `manifest.jsonl`.
  - Validate required artifact files and columns.
  - Compute grouped summaries for every factor in sweep config order.
  - Compute deterministic topline rankings.

- Create `src/society_simulation/sweep_analysis_artifacts.py`
  - Write `analysis/report.md`.
  - Write `analysis/group_summary.csv`.
  - Write `analysis/group_summary.json`.
  - Write `analysis/failure_summary.csv`.

- Modify `src/society_simulation/cli.py`
  - Add `analyze <sweep_output_dir>`.
  - Keep `run` and `sweep` output unchanged.

- Modify `tests/test_cli.py`
  - Add CLI analyze success and clean error tests.

- Create `tests/test_sweep_analysis.py`
  - Unit tests for artifact loading, grouped summaries, ordering, failed/incomplete rows, and validation.

- Create `tests/test_sweep_analysis_artifacts.py`
  - Unit tests for Markdown, CSV, JSON, and failure summary writers.

- Modify `README.md`
  - Document the `analyze` command and output artifacts.

---

### Task 1: Sweep Analysis Computation

**Files:**
- Create: `src/society_simulation/sweep_analysis.py`
- Create: `tests/test_sweep_analysis.py`

- [ ] **Step 1: Write failing analysis computation tests**

Create `tests/test_sweep_analysis.py`:

```python
import csv
import json
from pathlib import Path

import pytest

from society_simulation.sweep_analysis import analyze_sweep


SUMMARY_FIELDS = [
    "run_id",
    "seed",
    "topology",
    "threshold",
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
]


def write_analysis_fixture(tmp_path: Path) -> Path:
    sweep_dir = tmp_path / "network_topology_sweep"
    sweep_dir.mkdir()
    (sweep_dir / "sweep_config.json").write_text(
        json.dumps(
            {
                "sweep_name": "network_topology_sweep",
                "base_config": {"experiment_name": "network_herding"},
                "factors": [
                    {"name": "seed", "path": "seed", "values": [1, 2]},
                    {
                        "name": "topology",
                        "values": [
                            {"label": "complete", "overrides": {"topology": {"type": "complete"}}},
                            {"label": "cycle", "overrides": {"topology": {"type": "cycle"}}},
                        ],
                    },
                    {
                        "name": "threshold",
                        "path": "update_policy.adoption_threshold",
                        "values": [0.55, 0.7],
                    },
                ],
                "output_dir": str(sweep_dir),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    rows = [
        {
            "run_id": "seed-1__topology-complete__threshold-0_55",
            "seed": "1",
            "topology": "complete",
            "threshold": "0_55",
            "experiment_name": "network_herding",
            "output_dir": str(sweep_dir / "runs" / "run-1"),
            "status": "completed",
            "error": "",
            "final_action_counts_A": "30",
            "final_action_counts_B": "0",
            "final_a_fraction": "1.0",
            "consensus_reached": "True",
            "consensus_action": "A",
            "time_to_consensus": "1",
            "polarization_index": "0.0",
            "opinion_variance": "0.0",
            "mean_belief": "1.0",
            "edge_disagreement_rate": "0.0",
            "component_count": "1",
        },
        {
            "run_id": "seed-2__topology-complete__threshold-0_7",
            "seed": "2",
            "topology": "complete",
            "threshold": "0_7",
            "experiment_name": "network_herding",
            "output_dir": str(sweep_dir / "runs" / "run-2"),
            "status": "completed",
            "error": "",
            "final_action_counts_A": "27",
            "final_action_counts_B": "3",
            "final_a_fraction": "0.9",
            "consensus_reached": "True",
            "consensus_action": "A",
            "time_to_consensus": "2",
            "polarization_index": "0.1",
            "opinion_variance": "0.02",
            "mean_belief": "0.9",
            "edge_disagreement_rate": "0.05",
            "component_count": "1",
        },
        {
            "run_id": "seed-1__topology-cycle__threshold-0_55",
            "seed": "1",
            "topology": "cycle",
            "threshold": "0_55",
            "experiment_name": "network_herding",
            "output_dir": str(sweep_dir / "runs" / "run-3"),
            "status": "completed",
            "error": "",
            "final_action_counts_A": "15",
            "final_action_counts_B": "15",
            "final_a_fraction": "0.5",
            "consensus_reached": "False",
            "consensus_action": "",
            "time_to_consensus": "",
            "polarization_index": "0.6",
            "opinion_variance": "0.25",
            "mean_belief": "0.5",
            "edge_disagreement_rate": "0.4",
            "component_count": "1",
        },
        {
            "run_id": "seed-2__topology-cycle__threshold-0_7",
            "seed": "2",
            "topology": "cycle",
            "threshold": "0_7",
            "experiment_name": "network_herding",
            "output_dir": str(sweep_dir / "runs" / "run-4"),
            "status": "failed",
            "error": "boom",
            "final_action_counts_A": "",
            "final_action_counts_B": "",
            "final_a_fraction": "",
            "consensus_reached": "",
            "consensus_action": "",
            "time_to_consensus": "",
            "polarization_index": "",
            "opinion_variance": "",
            "mean_belief": "",
            "edge_disagreement_rate": "",
            "component_count": "",
        },
        {
            "run_id": "seed-1__topology-cycle__threshold-0_7",
            "seed": "1",
            "topology": "cycle",
            "threshold": "0_7",
            "experiment_name": "network_herding",
            "output_dir": str(sweep_dir / "runs" / "run-5"),
            "status": "pending",
            "error": "",
            "final_action_counts_A": "",
            "final_action_counts_B": "",
            "final_a_fraction": "",
            "consensus_reached": "",
            "consensus_action": "",
            "time_to_consensus": "",
            "polarization_index": "",
            "opinion_variance": "",
            "mean_belief": "",
            "edge_disagreement_rate": "",
            "component_count": "",
        },
    ]
    with (sweep_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (sweep_dir / "summary.json").write_text(
        json.dumps(
            {
                "sweep_name": "network_topology_sweep",
                "runs": 5,
                "completed": 3,
                "failed": 1,
                "metric_means": {},
                "groups": {},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (sweep_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return sweep_dir


def group_by(result, factor_name: str, value: str):
    for group in result.group_summaries:
        if group.factor_name == factor_name and group.value == value:
            return group
    raise AssertionError(f"group not found: {factor_name}={value}")


def test_analyze_sweep_computes_grouped_metrics_and_toplines(tmp_path: Path) -> None:
    sweep_dir = write_analysis_fixture(tmp_path)

    result = analyze_sweep(sweep_dir)

    assert result.sweep_name == "network_topology_sweep"
    assert result.runs == 5
    assert result.completed == 3
    assert result.failed == 1
    assert result.analysis_dir == sweep_dir / "analysis"

    complete = group_by(result, "topology", "complete")
    assert complete.runs == 2
    assert complete.completed == 2
    assert complete.failed == 0
    assert complete.consensus_rate == 1.0
    assert complete.mean_final_a_fraction == pytest.approx(0.95)
    assert complete.mean_polarization_index == pytest.approx(0.05)
    assert complete.mean_edge_disagreement_rate == pytest.approx(0.025)
    assert complete.mean_time_to_consensus == pytest.approx(1.5)

    cycle = group_by(result, "topology", "cycle")
    assert cycle.runs == 3
    assert cycle.completed == 1
    assert cycle.failed == 1
    assert cycle.consensus_rate == 0.0
    assert cycle.mean_final_a_fraction == 0.5
    assert cycle.mean_polarization_index == 0.6
    assert cycle.mean_time_to_consensus is None

    assert [group.factor_name for group in result.group_summaries[:2]] == [
        "seed",
        "seed",
    ]
    assert [group.value for group in result.group_summaries[:2]] == ["1", "2"]
    assert result.toplines["highest_consensus_rate"].factor_name == "topology"
    assert result.toplines["highest_consensus_rate"].value == "complete"
    assert result.toplines["highest_polarization"].factor_name == "topology"
    assert result.toplines["highest_polarization"].value == "cycle"
    assert [run.status for run in result.incomplete_runs] == ["failed", "pending"]


def test_analyze_sweep_rejects_missing_required_file(tmp_path: Path) -> None:
    sweep_dir = write_analysis_fixture(tmp_path)
    (sweep_dir / "summary.csv").unlink()

    with pytest.raises(ValueError, match="missing required sweep artifact: summary.csv"):
        analyze_sweep(sweep_dir)


def test_analyze_sweep_rejects_summary_count_mismatch(tmp_path: Path) -> None:
    sweep_dir = write_analysis_fixture(tmp_path)
    (sweep_dir / "summary.json").write_text(
        json.dumps({"sweep_name": "network_topology_sweep", "runs": 99, "completed": 3, "failed": 1}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="summary.csv row count 5 does not match summary.json runs 99"):
        analyze_sweep(sweep_dir)


def test_analyze_sweep_rejects_missing_factor_column(tmp_path: Path) -> None:
    sweep_dir = write_analysis_fixture(tmp_path)
    rows = list(csv.DictReader((sweep_dir / "summary.csv").open(newline="", encoding="utf-8")))
    fieldnames = [field for field in SUMMARY_FIELDS if field != "threshold"]
    with (sweep_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key != "threshold"} for row in rows)

    with pytest.raises(ValueError, match="summary.csv is missing required column: threshold"):
        analyze_sweep(sweep_dir)
```

- [ ] **Step 2: Run analysis tests and verify they fail**

Run:

```bash
./.venv/bin/python -m pytest tests/test_sweep_analysis.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'society_simulation.sweep_analysis'`.

- [ ] **Step 3: Implement `sweep_analysis.py`**

Create `src/society_simulation/sweep_analysis.py` with these public dataclasses and functions:

```python
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


MEAN_FIELDS = (
    "final_a_fraction",
    "polarization_index",
    "edge_disagreement_rate",
    "time_to_consensus",
    "opinion_variance",
    "component_count",
)
REQUIRED_BASE_COLUMNS = (
    "run_id",
    "experiment_name",
    "output_dir",
    "status",
    "error",
    "consensus_reached",
    *MEAN_FIELDS,
)
TOPLINE_KEYS = (
    ("highest_consensus_rate", "consensus_rate"),
    ("highest_polarization", "mean_polarization_index"),
    ("highest_edge_disagreement", "mean_edge_disagreement_rate"),
)


@dataclass(frozen=True)
class GroupSummary:
    factor_name: str
    value: str
    runs: int
    completed: int
    failed: int
    consensus_rate: float | None
    mean_final_a_fraction: float | None
    mean_polarization_index: float | None
    mean_edge_disagreement_rate: float | None
    mean_time_to_consensus: float | None
    mean_opinion_variance: float | None
    mean_component_count: float | None

    def metric(self, name: str) -> float | None:
        return getattr(self, name)

    def to_dict(self) -> dict[str, object]:
        return {
            "factor_name": self.factor_name,
            "value": self.value,
            "runs": self.runs,
            "completed": self.completed,
            "failed": self.failed,
            "consensus_rate": self.consensus_rate,
            "mean_final_a_fraction": self.mean_final_a_fraction,
            "mean_polarization_index": self.mean_polarization_index,
            "mean_edge_disagreement_rate": self.mean_edge_disagreement_rate,
            "mean_time_to_consensus": self.mean_time_to_consensus,
            "mean_opinion_variance": self.mean_opinion_variance,
            "mean_component_count": self.mean_component_count,
        }


@dataclass(frozen=True)
class IncompleteRun:
    run_id: str
    status: str
    error: str
    output_dir: str

    def to_dict(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "error": self.error,
            "output_dir": self.output_dir,
        }


@dataclass(frozen=True)
class ToplineEntry:
    name: str
    factor_name: str
    value: str
    metric_value: float

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "factor_name": self.factor_name,
            "value": self.value,
            "metric_value": self.metric_value,
        }


@dataclass(frozen=True)
class SweepAnalysisResult:
    sweep_name: str
    source_dir: Path
    analysis_dir: Path
    runs: int
    completed: int
    failed: int
    factor_names: tuple[str, ...]
    group_summaries: tuple[GroupSummary, ...]
    toplines: dict[str, ToplineEntry]
    incomplete_runs: tuple[IncompleteRun, ...]


def analyze_sweep(output_dir: str | Path) -> SweepAnalysisResult:
    source_dir = Path(output_dir)
    if not source_dir.is_dir():
        raise ValueError(f"sweep output directory does not exist: {source_dir}")

    sweep_config = _read_json(source_dir / "sweep_config.json")
    summary_json = _read_json(source_dir / "summary.json")
    _read_manifest(source_dir / "manifest.jsonl")
    rows = _read_summary_csv(source_dir / "summary.csv")
    factor_names = _factor_names(sweep_config)
    _validate_columns(rows, factor_names)
    _validate_counts(rows, summary_json)

    group_summaries = _group_summaries(rows, factor_names)
    return SweepAnalysisResult(
        sweep_name=_require_str(sweep_config.get("sweep_name"), "sweep_name"),
        source_dir=source_dir,
        analysis_dir=source_dir / "analysis",
        runs=len(rows),
        completed=_count_status(rows, "completed"),
        failed=_count_status(rows, "failed"),
        factor_names=factor_names,
        group_summaries=group_summaries,
        toplines=_toplines(group_summaries),
        incomplete_runs=tuple(
            IncompleteRun(
                run_id=row["run_id"],
                status=row["status"],
                error=row.get("error", ""),
                output_dir=row.get("output_dir", ""),
            )
            for row in rows
            if row["status"] != "completed"
        ),
    )
```

Then implement private helpers in the same file:

```python
def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"missing required sweep artifact: {path.name}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON artifact {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return data


def _read_manifest(path: Path) -> None:
    if not path.exists():
        raise ValueError(f"missing required sweep artifact: {path.name}")
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON artifact {path.name}: {exc}") from exc


def _read_summary_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise ValueError(f"missing required sweep artifact: {path.name}")
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except csv.Error as exc:
        raise ValueError(f"malformed CSV artifact {path.name}: {exc}") from exc
    if rows == []:
        raise ValueError("summary.csv must contain at least one row")
    return rows


def _factor_names(sweep_config: dict[str, Any]) -> tuple[str, ...]:
    factors = sweep_config.get("factors")
    if not isinstance(factors, list) or not factors:
        raise ValueError("sweep_config.json factors must be a non-empty list")
    names: list[str] = []
    for factor in factors:
        if not isinstance(factor, dict) or not isinstance(factor.get("name"), str):
            raise ValueError("sweep_config.json factors must include string names")
        names.append(factor["name"])
    return tuple(names)


def _validate_columns(rows: list[dict[str, str]], factor_names: tuple[str, ...]) -> None:
    columns = set(rows[0])
    for column in (*factor_names, *REQUIRED_BASE_COLUMNS):
        if column not in columns:
            raise ValueError(f"summary.csv is missing required column: {column}")


def _validate_counts(rows: list[dict[str, str]], summary: dict[str, Any]) -> None:
    expected_runs = summary.get("runs")
    if expected_runs != len(rows):
        raise ValueError(
            f"summary.csv row count {len(rows)} does not match summary.json runs {expected_runs}"
        )
    for status in ("completed", "failed"):
        expected = summary.get(status)
        actual = _count_status(rows, status)
        if expected != actual:
            raise ValueError(f"summary.csv {status} count {actual} does not match summary.json {status} {expected}")


def _group_summaries(
    rows: list[dict[str, str]],
    factor_names: tuple[str, ...],
) -> tuple[GroupSummary, ...]:
    summaries: list[GroupSummary] = []
    for factor_name in factor_names:
        seen_values = dict.fromkeys(row[factor_name] for row in rows)
        for value in seen_values:
            group_rows = [row for row in rows if row[factor_name] == value]
            completed_rows = [row for row in group_rows if row["status"] == "completed"]
            summaries.append(
                GroupSummary(
                    factor_name=factor_name,
                    value=value,
                    runs=len(group_rows),
                    completed=len(completed_rows),
                    failed=_count_status(group_rows, "failed"),
                    consensus_rate=_consensus_rate(completed_rows),
                    mean_final_a_fraction=_mean(completed_rows, "final_a_fraction"),
                    mean_polarization_index=_mean(completed_rows, "polarization_index"),
                    mean_edge_disagreement_rate=_mean(completed_rows, "edge_disagreement_rate"),
                    mean_time_to_consensus=_mean(completed_rows, "time_to_consensus"),
                    mean_opinion_variance=_mean(completed_rows, "opinion_variance"),
                    mean_component_count=_mean(completed_rows, "component_count"),
                )
            )
    return tuple(summaries)


def _count_status(rows: list[dict[str, str]], status: str) -> int:
    return sum(1 for row in rows if row["status"] == status)


def _consensus_rate(rows: list[dict[str, str]]) -> float | None:
    values = [_parse_bool(row["consensus_reached"]) for row in rows]
    parsed = [value for value in values if value is not None]
    if not parsed:
        return None
    return sum(1 for value in parsed if value) / len(parsed)


def _parse_bool(value: str) -> bool | None:
    if value == "True":
        return True
    if value == "False":
        return False
    return None


def _mean(rows: list[dict[str, str]], field: str) -> float | None:
    values = [_parse_float(row[field]) for row in rows]
    parsed = [value for value in values if value is not None]
    if not parsed:
        return None
    return sum(parsed) / len(parsed)


def _parse_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _toplines(group_summaries: tuple[GroupSummary, ...]) -> dict[str, ToplineEntry]:
    toplines: dict[str, ToplineEntry] = {}
    for topline_name, metric_name in TOPLINE_KEYS:
        candidates = [
            group for group in group_summaries if group.metric(metric_name) is not None
        ]
        treatment_candidates = [
            group for group in candidates if group.factor_name != "seed"
        ]
        if treatment_candidates:
            candidates = treatment_candidates
        if not candidates:
            continue
        best = max(candidates, key=lambda group: group.metric(metric_name) or float("-inf"))
        value = best.metric(metric_name)
        assert value is not None
        toplines[topline_name] = ToplineEntry(
            name=topline_name,
            factor_name=best.factor_name,
            value=best.value,
            metric_value=value,
        )
    return toplines


def _require_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value
```

- [ ] **Step 4: Run Task 1 tests and commit**

Run:

```bash
./.venv/bin/python -m pytest tests/test_sweep_analysis.py -v
```

Expected: PASS.

Commit:

```bash
git add src/society_simulation/sweep_analysis.py tests/test_sweep_analysis.py
git commit -m "feat: add sweep analysis computation"
```

---

### Task 2: Sweep Analysis Artifact Writers

**Files:**
- Create: `src/society_simulation/sweep_analysis_artifacts.py`
- Create: `tests/test_sweep_analysis_artifacts.py`

- [ ] **Step 1: Write failing artifact writer tests**

Create `tests/test_sweep_analysis_artifacts.py`:

```python
import csv
import json
from pathlib import Path

from society_simulation.sweep_analysis import analyze_sweep
from society_simulation.sweep_analysis_artifacts import write_analysis_artifacts
from tests.test_sweep_analysis import write_analysis_fixture


def test_write_analysis_artifacts_writes_report_csv_json_and_failures(
    tmp_path: Path,
) -> None:
    result = analyze_sweep(write_analysis_fixture(tmp_path))

    paths = write_analysis_artifacts(result)

    assert paths.output_dir == result.analysis_dir
    assert paths.report_path.exists()
    assert paths.group_summary_csv_path.exists()
    assert paths.group_summary_json_path.exists()
    assert paths.failure_summary_csv_path.exists()

    report = paths.report_path.read_text(encoding="utf-8")
    assert "# Sweep Analysis: network_topology_sweep" in report
    assert "- Runs: 5" in report
    assert "- Completed: 3" in report
    assert "- Failed: 1" in report
    assert "Highest consensus rate: topology=complete" in report
    assert "Highest polarization: topology=cycle" in report
    assert "## Factor Summaries" in report
    assert "### topology" in report
    assert "| complete | 2 | 2 | 0 | 1.0000 | 0.9500 | 0.0500 | 0.0250 |" in report
    assert "## Failures" in report
    assert "| seed-2__topology-cycle__threshold-0_7 | failed | boom |" in report

    with paths.group_summary_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["factor"] == "seed"
    assert rows[0]["value"] == "1"
    assert rows[2]["factor"] == "topology"
    assert rows[2]["value"] == "complete"
    assert rows[2]["consensus_rate"] == "1.0000"
    assert rows[3]["value"] == "cycle"
    assert rows[3]["mean_time_to_consensus"] == ""

    summary = json.loads(paths.group_summary_json_path.read_text(encoding="utf-8"))
    assert summary["sweep_name"] == "network_topology_sweep"
    assert summary["runs"] == 5
    assert summary["groups"]["topology"]["complete"]["consensus_rate"] == 1.0
    assert summary["toplines"]["highest_edge_disagreement"]["value"] == "cycle"

    with paths.failure_summary_csv_path.open(newline="", encoding="utf-8") as handle:
        failure_rows = list(csv.DictReader(handle))
    assert [row["status"] for row in failure_rows] == ["failed", "pending"]
    assert failure_rows[0]["error"] == "boom"


def test_write_analysis_artifacts_is_deterministic(tmp_path: Path) -> None:
    result = analyze_sweep(write_analysis_fixture(tmp_path))

    first = write_analysis_artifacts(result)
    first_report = first.report_path.read_text(encoding="utf-8")
    first_json = first.group_summary_json_path.read_text(encoding="utf-8")
    second = write_analysis_artifacts(result)

    assert second.report_path.read_text(encoding="utf-8") == first_report
    assert second.group_summary_json_path.read_text(encoding="utf-8") == first_json
```

- [ ] **Step 2: Run artifact writer tests and verify they fail**

Run:

```bash
./.venv/bin/python -m pytest tests/test_sweep_analysis_artifacts.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'society_simulation.sweep_analysis_artifacts'`.

- [ ] **Step 3: Implement `sweep_analysis_artifacts.py`**

Create `src/society_simulation/sweep_analysis_artifacts.py`:

```python
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path

from society_simulation.sweep_analysis import (
    GroupSummary,
    IncompleteRun,
    SweepAnalysisResult,
)


GROUP_SUMMARY_FIELDS = (
    "factor",
    "value",
    "runs",
    "completed",
    "failed",
    "consensus_rate",
    "mean_final_a_fraction",
    "mean_polarization_index",
    "mean_edge_disagreement_rate",
    "mean_time_to_consensus",
    "mean_opinion_variance",
    "mean_component_count",
)
FAILURE_FIELDS = ("run_id", "status", "error", "output_dir")


@dataclass(frozen=True)
class SweepAnalysisArtifactPaths:
    output_dir: Path
    report_path: Path
    group_summary_csv_path: Path
    group_summary_json_path: Path
    failure_summary_csv_path: Path


def write_analysis_artifacts(result: SweepAnalysisResult) -> SweepAnalysisArtifactPaths:
    output_dir = result.analysis_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = SweepAnalysisArtifactPaths(
        output_dir=output_dir,
        report_path=output_dir / "report.md",
        group_summary_csv_path=output_dir / "group_summary.csv",
        group_summary_json_path=output_dir / "group_summary.json",
        failure_summary_csv_path=output_dir / "failure_summary.csv",
    )
    paths.report_path.write_text(_report_markdown(result), encoding="utf-8")
    _write_group_summary_csv(paths.group_summary_csv_path, result.group_summaries)
    paths.group_summary_json_path.write_text(
        json.dumps(_summary_json(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_failure_summary_csv(paths.failure_summary_csv_path, result.incomplete_runs)
    return paths
```

Then implement private helpers in the same file:

```python
def _write_group_summary_csv(path: Path, groups: tuple[GroupSummary, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GROUP_SUMMARY_FIELDS)
        writer.writeheader()
        for group in groups:
            writer.writerow(_group_csv_row(group))


def _write_failure_summary_csv(
    path: Path,
    incomplete_runs: tuple[IncompleteRun, ...],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILURE_FIELDS)
        writer.writeheader()
        for run in incomplete_runs:
            writer.writerow(run.to_dict())


def _summary_json(result: SweepAnalysisResult) -> dict[str, object]:
    groups: dict[str, dict[str, dict[str, object]]] = {}
    for group in result.group_summaries:
        groups.setdefault(group.factor_name, {})[group.value] = group.to_dict()
    return {
        "sweep_name": result.sweep_name,
        "runs": result.runs,
        "completed": result.completed,
        "failed": result.failed,
        "groups": groups,
        "toplines": {
            name: topline.to_dict()
            for name, topline in result.toplines.items()
        },
        "incomplete_runs": [run.to_dict() for run in result.incomplete_runs],
    }


def _report_markdown(result: SweepAnalysisResult) -> str:
    lines = [
        f"# Sweep Analysis: {result.sweep_name}",
        "",
        "## Overview",
        "",
        f"- Runs: {result.runs}",
        f"- Completed: {result.completed}",
        f"- Failed: {result.failed}",
        "",
        "## Topline",
        "",
    ]
    lines.extend(_topline_lines(result))
    lines.extend(["", "## Factor Summaries", ""])
    for factor_name in result.factor_names:
        lines.extend(_factor_table(factor_name, result.group_summaries))
    lines.extend(["## Failures", ""])
    lines.extend(_failure_lines(result.incomplete_runs))
    return "\n".join(lines) + "\n"


def _topline_lines(result: SweepAnalysisResult) -> list[str]:
    labels = {
        "highest_consensus_rate": "Highest consensus rate",
        "highest_polarization": "Highest polarization",
        "highest_edge_disagreement": "Highest edge disagreement",
    }
    lines: list[str] = []
    for key, label in labels.items():
        topline = result.toplines.get(key)
        if topline is None:
            lines.append(f"- {label}: no completed metric data")
        else:
            lines.append(
                f"- {label}: {topline.factor_name}={topline.value} ({_format_number(topline.metric_value)})"
            )
    return lines


def _factor_table(
    factor_name: str,
    groups: tuple[GroupSummary, ...],
) -> list[str]:
    lines = [
        f"### {factor_name}",
        "",
        "| value | runs | completed | failed | consensus_rate | mean_final_a_fraction | mean_polarization_index | mean_edge_disagreement_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in groups:
        if group.factor_name != factor_name:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    group.value,
                    str(group.runs),
                    str(group.completed),
                    str(group.failed),
                    _format_optional(group.consensus_rate),
                    _format_optional(group.mean_final_a_fraction),
                    _format_optional(group.mean_polarization_index),
                    _format_optional(group.mean_edge_disagreement_rate),
                ]
            )
            + " |"
        )
    lines.extend([""])
    return lines


def _failure_lines(incomplete_runs: tuple[IncompleteRun, ...]) -> list[str]:
    if not incomplete_runs:
        return ["No failed or incomplete runs."]
    lines = [
        "| run_id | status | error | output_dir |",
        "| --- | --- | --- | --- |",
    ]
    for run in incomplete_runs:
        lines.append(f"| {run.run_id} | {run.status} | {run.error} | {run.output_dir} |")
    return lines


def _group_csv_row(group: GroupSummary) -> dict[str, object]:
    return {
        "factor": group.factor_name,
        "value": group.value,
        "runs": group.runs,
        "completed": group.completed,
        "failed": group.failed,
        "consensus_rate": _format_optional(group.consensus_rate),
        "mean_final_a_fraction": _format_optional(group.mean_final_a_fraction),
        "mean_polarization_index": _format_optional(group.mean_polarization_index),
        "mean_edge_disagreement_rate": _format_optional(group.mean_edge_disagreement_rate),
        "mean_time_to_consensus": _format_optional(group.mean_time_to_consensus),
        "mean_opinion_variance": _format_optional(group.mean_opinion_variance),
        "mean_component_count": _format_optional(group.mean_component_count),
    }


def _format_optional(value: float | None) -> str:
    if value is None:
        return ""
    return _format_number(value)


def _format_number(value: float) -> str:
    return f"{value:.4f}"
```

- [ ] **Step 4: Run artifact writer tests and commit**

Run:

```bash
./.venv/bin/python -m pytest tests/test_sweep_analysis_artifacts.py tests/test_sweep_analysis.py -v
```

Expected: PASS.

Commit:

```bash
git add src/society_simulation/sweep_analysis_artifacts.py tests/test_sweep_analysis_artifacts.py
git commit -m "feat: add sweep analysis artifacts"
```

---

### Task 3: CLI Analyze Command

**Files:**
- Modify: `src/society_simulation/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI analyze tests**

Append to `tests/test_cli.py`:

```python
def test_cli_analyze_writes_report_and_prints_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.test_sweep_analysis import write_analysis_fixture

    sweep_dir = write_analysis_fixture(tmp_path)

    exit_code = cli.main(["analyze", str(sweep_dir)])

    assert exit_code == 0
    output = capsys.readouterr().out.splitlines()
    assert output == [
        "analysis=network_topology_sweep",
        "runs=5",
        "completed=3",
        "failed=1",
        f"output_dir={sweep_dir / 'analysis'}",
        f"report={sweep_dir / 'analysis' / 'report.md'}",
    ]
    assert (sweep_dir / "analysis" / "report.md").exists()
    assert (sweep_dir / "analysis" / "group_summary.csv").exists()
    assert (sweep_dir / "analysis" / "group_summary.json").exists()
    assert (sweep_dir / "analysis" / "failure_summary.csv").exists()


def test_cli_analyze_invalid_output_dir_reports_clean_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing-sweep-output"

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["analyze", str(missing)])

    assert exc_info.value.code == 2
    captured = capsys.readouterr().err
    assert "Analyze failed for" in captured
    assert "sweep output directory does not exist" in captured
    assert "Traceback" not in captured


def test_cli_analyze_missing_artifact_reports_clean_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.test_sweep_analysis import write_analysis_fixture

    sweep_dir = write_analysis_fixture(tmp_path)
    (sweep_dir / "manifest.jsonl").unlink()

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["analyze", str(sweep_dir)])

    assert exc_info.value.code == 2
    captured = capsys.readouterr().err
    assert "Analyze failed for" in captured
    assert "missing required sweep artifact: manifest.jsonl" in captured
    assert "Traceback" not in captured
```

- [ ] **Step 2: Run CLI tests and verify they fail**

Run:

```bash
./.venv/bin/python -m pytest tests/test_cli.py::test_cli_analyze_writes_report_and_prints_summary -v
```

Expected: FAIL because parser choices do not include `analyze`.

- [ ] **Step 3: Implement CLI analyze branch**

Modify imports in `src/society_simulation/cli.py`:

```python
from society_simulation.sweep_analysis import analyze_sweep
from society_simulation.sweep_analysis_artifacts import write_analysis_artifacts
```

Update `build_parser`:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="society-sim")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run").add_argument("config")
    subparsers.add_parser("sweep").add_argument("config")
    subparsers.add_parser("analyze").add_argument("sweep_output_dir")

    return parser
```

Add `_run_analyze_config`:

```python
def _run_analyze_config(parser: argparse.ArgumentParser, sweep_output_dir: str) -> int:
    try:
        result = analyze_sweep(sweep_output_dir)
        paths = write_analysis_artifacts(result)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"Analyze failed for '{sweep_output_dir}': {exc}")

    print(f"analysis={result.sweep_name}")
    print(f"runs={result.runs}")
    print(f"completed={result.completed}")
    print(f"failed={result.failed}")
    print(f"output_dir={paths.output_dir}")
    print(f"report={paths.report_path}")
    return 0
```

Update `main` dispatch:

```python
def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "run":
        return _run_single_config(parser, args.config)
    if args.command == "sweep":
        return _run_sweep_config(parser, args.config)
    if args.command == "analyze":
        return _run_analyze_config(parser, args.sweep_output_dir)
    return 1
```

- [ ] **Step 4: Run CLI and focused tests, then commit**

Run:

```bash
./.venv/bin/python -m pytest tests/test_cli.py tests/test_sweep_analysis.py tests/test_sweep_analysis_artifacts.py -v
```

Expected: PASS.

Commit:

```bash
git add src/society_simulation/cli.py tests/test_cli.py
git commit -m "feat: add sweep analyze CLI command"
```

---

### Task 4: Documentation and Real Example Smoke Test

**Files:**
- Modify: `README.md`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write real example smoke test**

Append to `tests/test_cli.py`:

```python
def test_analyze_command_accepts_real_network_topology_sweep(
    tmp_path: Path,
) -> None:
    from society_simulation.sweep_config import expand_sweep, load_sweep_config
    from society_simulation.sweep_runner import run_sweep
    from society_simulation.sweep_analysis import analyze_sweep
    from society_simulation.sweep_analysis_artifacts import write_analysis_artifacts

    sweep = load_sweep_config("experiments/network_topology_sweep.json")
    sweep = type(sweep)(
        sweep_name=sweep.sweep_name,
        base_config=sweep.base_config,
        factors=sweep.factors,
        output_dir=str(tmp_path / "network_topology_sweep"),
    )
    assert len(expand_sweep(sweep)) == 48

    sweep_result = run_sweep(sweep)
    analysis_result = analyze_sweep(sweep_result.output_dir)
    paths = write_analysis_artifacts(analysis_result)

    assert analysis_result.runs == 48
    assert analysis_result.completed == 48
    assert analysis_result.failed == 0
    assert paths.report_path.exists()
    assert paths.group_summary_csv_path.exists()
```

- [ ] **Step 2: Run smoke test and verify it passes**

Run:

```bash
./.venv/bin/python -m pytest tests/test_cli.py::test_analyze_command_accepts_real_network_topology_sweep -v
```

Expected: PASS and confirms the real example can be analyzed.

- [ ] **Step 3: Update README**

Modify `README.md` so the `Run` section includes:

````markdown
Analyze a completed sweep:

```bash
python -m society_simulation analyze runs/sweeps/network_topology_sweep
```
````

Modify the artifact section to add:

```markdown
Sweep analysis writes:

- `analysis/report.md`
- `analysis/group_summary.csv`
- `analysis/group_summary.json`
- `analysis/failure_summary.csv`
```

- [ ] **Step 4: Run example workflow**

Run:

```bash
./.venv/bin/python -m society_simulation sweep experiments/network_topology_sweep.json
./.venv/bin/python -m society_simulation analyze runs/sweeps/network_topology_sweep
```

Expected analyze output:

```text
analysis=network_topology_sweep
runs=48
completed=48
failed=0
output_dir=runs/sweeps/network_topology_sweep/analysis
report=runs/sweeps/network_topology_sweep/analysis/report.md
```

- [ ] **Step 5: Run focused tests and commit**

Run:

```bash
./.venv/bin/python -m pytest tests/test_cli.py tests/test_sweep_analysis.py tests/test_sweep_analysis_artifacts.py -v
```

Expected: PASS.

Commit:

```bash
git add README.md tests/test_cli.py
git commit -m "docs: document sweep analysis workflow"
```

Generated ignored artifacts under `runs/` should not be committed.

---

### Task 5: Final Verification and Integration

**Files:**
- No new source files.
- Verify all changed behavior.

- [ ] **Step 1: Run focused analyzer tests**

Run:

```bash
./.venv/bin/python -m pytest tests/test_sweep_analysis.py tests/test_sweep_analysis_artifacts.py tests/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 2: Run full test suite**

Run:

```bash
./.venv/bin/python -m pytest -v
```

Expected: PASS.

- [ ] **Step 3: Run existing single-run examples**

Run:

```bash
./.venv/bin/python -m society_simulation run examples/sequential_cascade.json
./.venv/bin/python -m society_simulation run examples/network_herding.json
```

Expected: both commands exit 0 and print a line beginning with `output_dir=`.

- [ ] **Step 4: Run sweep and analyze example**

Run:

```bash
./.venv/bin/python -m society_simulation sweep experiments/network_topology_sweep.json
./.venv/bin/python -m society_simulation analyze runs/sweeps/network_topology_sweep
```

Expected: analyze command exits 0 and prints `runs=48`, `completed=48`, `failed=0`.

- [ ] **Step 5: Inspect generated analysis artifacts**

Run:

```bash
test -f runs/sweeps/network_topology_sweep/analysis/report.md
test -f runs/sweeps/network_topology_sweep/analysis/group_summary.csv
test -f runs/sweeps/network_topology_sweep/analysis/group_summary.json
test -f runs/sweeps/network_topology_sweep/analysis/failure_summary.csv
head -n 20 runs/sweeps/network_topology_sweep/analysis/report.md
head -n 5 runs/sweeps/network_topology_sweep/analysis/group_summary.csv
```

Expected: all `test -f` commands exit 0. Report starts with `# Sweep Analysis: network_topology_sweep`. CSV header starts with `factor,value,runs,completed,failed,consensus_rate`.

- [ ] **Step 6: Confirm git status**

Run:

```bash
git status --short
```

Expected: no tracked source, test, or docs files modified. Ignored `runs/` artifacts may exist and should not be committed.

- [ ] **Step 7: Final code review**

Dispatch a reviewer with this brief:

```text
Review the sweep analyzer implementation against docs/superpowers/specs/2026-06-18-sweep-analyzer-design.md and docs/superpowers/plans/2026-06-18-sweep-analyzer.md. Focus on correctness, deterministic output, clean CLI errors, artifact completeness, and preservation of existing `run` and `sweep` CLI behavior. Do not edit files. Return Critical/Important/Minor findings with file and line references.
```

Fix any Critical or Important findings, rerun the relevant tests, and commit fixes.

- [ ] **Step 8: Merge/push workflow**

Use `superpowers:finishing-a-development-branch` after review approval.

If merging locally to `main`, verify:

```bash
git status --short --branch
git log --oneline --decorate --max-count=5
git push origin main
```

Expected: `main` pushed to `origin/main` with all sweep analyzer commits.
