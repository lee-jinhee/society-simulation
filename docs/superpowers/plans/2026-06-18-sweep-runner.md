# Sweep Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a no-LLM sweep runner that expands experiment parameter grids, runs existing scenarios repeatedly, and writes comparable batch summaries.

**Architecture:** Add a sweep layer beside the existing single-run path. `sweep_config.py` owns strict parsing, factor expansion, run id generation, and generated config validation. `sweep_runner.py` dispatches materialized configs through the existing `run_experiment` function. `sweep_artifacts.py` writes deterministic manifest, CSV, and JSON outputs.

**Tech Stack:** Python 3.11+, standard library only at runtime, `pytest` for tests, existing `society_simulation` package structure.

---

## File Structure

- Create `src/society_simulation/sweep_config.py`
  - Parse sweep JSON.
  - Represent factors and materialized runs.
  - Apply scalar path replacements and override bundles.
  - Generate filesystem-safe run ids.
  - Validate all generated experiment configs before execution.

- Create `src/society_simulation/sweep_artifacts.py`
  - Represent per-run sweep records.
  - Flatten metrics into CSV columns.
  - Write `sweep_config.json`, `manifest.jsonl`, `summary.csv`, and `summary.json`.

- Create `src/society_simulation/sweep_runner.py`
  - Run every materialized config through `run_experiment`.
  - Continue after runtime failures.
  - Return a summary object for CLI printing.

- Modify `src/society_simulation/cli.py`
  - Add `sweep <config>` subcommand.
  - Keep `run <config>` output unchanged.
  - Print concise sweep summary.

- Create `tests/test_sweep_config.py`
  - Unit tests for parsing, expansion, validation, and run id generation.

- Create `tests/test_sweep_artifacts.py`
  - Unit tests for manifest, CSV, aggregate JSON, and failed-run rows.

- Create `tests/test_sweep_runner.py`
  - Unit tests for successful sweep execution and runtime failure capture.

- Modify `tests/test_cli.py`
  - Add CLI sweep success and invalid config tests.
  - Keep existing run tests passing.

- Create `experiments/network_topology_sweep.json`
  - 48-run no-LLM example sweep.

- Modify `README.md`
  - Document `sweep` command and output artifacts.

---

### Task 1: Sweep Config Parsing and Expansion

**Files:**
- Create: `src/society_simulation/sweep_config.py`
- Create: `tests/test_sweep_config.py`

- [ ] **Step 1: Write failing tests for valid sweep parsing and deterministic expansion**

Create `tests/test_sweep_config.py` with:

```python
import json
from pathlib import Path

import pytest

from society_simulation.sweep_config import (
    SweepConfig,
    apply_path_override,
    expand_sweep,
    load_sweep_config,
    safe_label,
)


def valid_base_config(tmp_path: Path) -> dict[str, object]:
    return {
        "experiment_name": "network_herding",
        "seed": 1,
        "num_agents": 6,
        "initial_opinion": {"type": "bernoulli", "probability_a": 0.5},
        "topology": {"type": "cycle"},
        "scheduler": {"type": "synchronous_rounds", "rounds": 2},
        "observation_policy": {"type": "neighbor_actions"},
        "update_policy": {"type": "threshold", "adoption_threshold": 0.6},
        "output_dir": str(tmp_path / "ignored"),
    }


def valid_sweep_dict(tmp_path: Path) -> dict[str, object]:
    return {
        "sweep_name": "network_topology_sweep",
        "base_config": valid_base_config(tmp_path),
        "factors": [
            {"name": "seed", "path": "seed", "values": [1, 2]},
            {
                "name": "initial_a",
                "path": "initial_opinion.probability_a",
                "values": [0.45, 0.55],
            },
            {
                "name": "topology",
                "values": [
                    {"label": "cycle", "overrides": {"topology": {"type": "cycle"}}},
                    {
                        "label": "complete",
                        "overrides": {"topology": {"type": "complete"}},
                    },
                ],
            },
        ],
        "output_dir": str(tmp_path / "sweep"),
    }


def test_load_sweep_config_parses_valid_config(tmp_path: Path) -> None:
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(valid_sweep_dict(tmp_path)), encoding="utf-8")

    sweep = load_sweep_config(path)

    assert isinstance(sweep, SweepConfig)
    assert sweep.sweep_name == "network_topology_sweep"
    assert sweep.output_dir == str(tmp_path / "sweep")
    assert [factor.name for factor in sweep.factors] == [
        "seed",
        "initial_a",
        "topology",
    ]


def test_expand_sweep_creates_deterministic_run_ids_and_output_dirs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(valid_sweep_dict(tmp_path)), encoding="utf-8")
    sweep = load_sweep_config(path)

    runs = expand_sweep(sweep)

    assert len(runs) == 8
    assert runs[0].run_id == "seed-1__initial_a-0_45__topology-cycle"
    assert runs[0].labels == {
        "seed": "1",
        "initial_a": "0_45",
        "topology": "cycle",
    }
    assert runs[0].config["seed"] == 1
    assert runs[0].config["initial_opinion"] == {
        "type": "bernoulli",
        "probability_a": 0.45,
    }
    assert runs[0].config["topology"] == {"type": "cycle"}
    assert runs[0].config["output_dir"] == str(
        tmp_path
        / "sweep"
        / "runs"
        / "seed-1__initial_a-0_45__topology-cycle"
    )
    assert runs[-1].run_id == "seed-2__initial_a-0_55__topology-complete"


def test_override_bundle_replaces_entire_subtree(tmp_path: Path) -> None:
    data = valid_sweep_dict(tmp_path)
    base_config = data["base_config"]
    assert isinstance(base_config, dict)
    base_config["topology"] = {
        "type": "small_world",
        "degree": 4,
        "rewiring_probability": 0.1,
    }
    data["factors"] = [
        {
            "name": "topology",
            "values": [
                {"label": "complete", "overrides": {"topology": {"type": "complete"}}}
            ],
        }
    ]
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    run = expand_sweep(load_sweep_config(path))[0]

    assert run.config["topology"] == {"type": "complete"}


def test_apply_path_override_requires_existing_object_path(tmp_path: Path) -> None:
    config = valid_base_config(tmp_path)

    apply_path_override(config, "initial_opinion.probability_a", 0.7)

    assert config["initial_opinion"] == {
        "type": "bernoulli",
        "probability_a": 0.7,
    }


def test_safe_label_converts_values_to_filesystem_safe_labels() -> None:
    assert safe_label(0.45) == "0_45"
    assert safe_label("small/world") == "small_world"
    assert safe_label(True) == "true"
    assert safe_label(None) == "null"
```

- [ ] **Step 2: Run config tests and verify they fail**

Run:

```bash
./.venv/bin/python -m pytest tests/test_sweep_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'society_simulation.sweep_config'`.

- [ ] **Step 3: Implement sweep config parser and expander**

Create `src/society_simulation/sweep_config.py`:

```python
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
import re
from typing import Any

from society_simulation.config import ExperimentConfig, NetworkHerdingConfig


JSONDict = dict[str, Any]


@dataclass(frozen=True)
class SweepFactorValue:
    label: str
    value: Any = None
    overrides: JSONDict | None = None


@dataclass(frozen=True)
class SweepFactor:
    name: str
    path: str | None
    values: tuple[SweepFactorValue, ...]


@dataclass(frozen=True)
class SweepConfig:
    sweep_name: str
    base_config: JSONDict
    factors: tuple[SweepFactor, ...]
    output_dir: str

    def to_dict(self) -> JSONDict:
        return {
            "sweep_name": self.sweep_name,
            "base_config": deepcopy(self.base_config),
            "factors": [
                _factor_to_dict(factor)
                for factor in self.factors
            ],
            "output_dir": self.output_dir,
        }


@dataclass(frozen=True)
class MaterializedRun:
    run_id: str
    labels: dict[str, str]
    config: JSONDict


def load_sweep_config(path: str | Path) -> SweepConfig:
    sweep_path = Path(path)
    data = json.loads(sweep_path.read_text(encoding="utf-8"))
    return parse_sweep_config(data)


def parse_sweep_config(data: object) -> SweepConfig:
    if not isinstance(data, dict):
        raise ValueError("sweep config root must be an object")
    sweep_name = _require_non_empty_str(data.get("sweep_name"), "sweep_name")
    base_config = _require_mapping(data.get("base_config"), "base_config")
    output_dir = _require_non_empty_str(data.get("output_dir"), "output_dir")
    raw_factors = data.get("factors")
    if not isinstance(raw_factors, list) or not raw_factors:
        raise ValueError("factors must be a non-empty list")

    factors = tuple(_parse_factor(raw_factor) for raw_factor in raw_factors)
    factor_names = [factor.name for factor in factors]
    if len(factor_names) != len(set(factor_names)):
        raise ValueError("factor names must be unique")

    sweep = SweepConfig(
        sweep_name=sweep_name,
        base_config=deepcopy(base_config),
        factors=factors,
        output_dir=output_dir,
    )
    _validate_materialized_runs(sweep)
    return sweep


def _parse_factor(raw_factor: object) -> SweepFactor:
    factor = _require_mapping(raw_factor, "factor")
    name = _require_non_empty_str(factor.get("name"), "factor.name")
    values = factor.get("values")
    if not isinstance(values, list) or not values:
        raise ValueError(f"factor {name} values must be a non-empty list")

    if "path" in factor:
        path = _require_non_empty_str(factor.get("path"), f"factor {name} path")
        parsed_values = tuple(
            _parse_path_value(name, path, value)
            for value in values
        )
        parsed_factor = SweepFactor(name=name, path=path, values=parsed_values)
    else:
        parsed_values = tuple(_parse_override_value(name, value) for value in values)
        parsed_factor = SweepFactor(name=name, path=None, values=parsed_values)

    labels = [value.label for value in parsed_values]
    if len(labels) != len(set(labels)):
        raise ValueError(f"factor {name} labels must be unique")
    return parsed_factor


def _parse_path_value(factor_name: str, path: str, value: object) -> SweepFactorValue:
    if isinstance(value, (dict, list)):
        raise ValueError(f"factor {factor_name} path values must be scalar")
    return SweepFactorValue(
        label=safe_label(value),
        value=deepcopy(value),
        overrides=None,
    )


def _parse_override_value(factor_name: str, value: object) -> SweepFactorValue:
    value_dict = _require_mapping(value, f"factor {factor_name} value")
    label = _require_non_empty_str(value_dict.get("label"), f"factor {factor_name} label")
    overrides = _require_mapping(
        value_dict.get("overrides"),
        f"factor {factor_name} overrides",
    )
    return SweepFactorValue(label=safe_label(label), value=None, overrides=deepcopy(overrides))


def apply_path_override(
    config: JSONDict,
    path: str,
    value: object,
    *,
    create_missing: bool = False,
) -> None:
    parts = path.split(".")
    if not parts or any(part == "" for part in parts):
        raise ValueError("factor path must contain non-empty segments")

    current = config
    for part in parts[:-1]:
        if part not in current:
            if not create_missing:
                raise ValueError(f"factor path {path} does not exist")
            current[part] = {}
        next_value = current[part]
        if not isinstance(next_value, dict):
            raise ValueError(f"factor path {path} traverses non-object field {part}")
        current = next_value
    current[parts[-1]] = value


def expand_sweep(sweep: SweepConfig) -> tuple[MaterializedRun, ...]:
    runs: list[MaterializedRun] = []
    for combination in product(*(factor.values for factor in sweep.factors)):
        labels = {
            factor.name: value.label
            for factor, value in zip(sweep.factors, combination, strict=True)
        }
        config = deepcopy(sweep.base_config)
        for factor, value in zip(sweep.factors, combination, strict=True):
            if factor.path is not None:
                apply_path_override(config, factor.path, deepcopy(value.value))
            else:
                _apply_overrides(config, value.overrides or {})
        run_id = "__".join(f"{factor.name}-{labels[factor.name]}" for factor in sweep.factors)
        output_dir = str(Path(sweep.output_dir) / "runs" / run_id)
        config["output_dir"] = output_dir
        runs.append(MaterializedRun(run_id=run_id, labels=labels, config=config))
    return tuple(runs)


def build_experiment_config(data: JSONDict) -> ExperimentConfig | NetworkHerdingConfig:
    config_data = deepcopy(data)
    if config_data.get("experiment_name") == "network_herding":
        config = NetworkHerdingConfig.from_dict(config_data)
    else:
        config = ExperimentConfig(**config_data)
    config.validate()
    return config


def safe_label(value: object) -> str:
    if value is True:
        text = "true"
    elif value is False:
        text = "false"
    elif value is None:
        text = "null"
    else:
        text = str(value)
    text = text.strip().replace(".", "_")
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    text = text.strip("_")
    if not text:
        raise ValueError("factor label must not be empty after sanitization")
    return text


def _apply_overrides(config: JSONDict, overrides: JSONDict) -> None:
    for key, value in overrides.items():
        config[key] = deepcopy(value)


def _factor_to_dict(factor: SweepFactor) -> JSONDict:
    if factor.path is not None:
        return {
            "name": factor.name,
            "path": factor.path,
            "values": [deepcopy(value.value) for value in factor.values],
        }
    return {
        "name": factor.name,
        "values": [
            {"label": value.label, "overrides": deepcopy(value.overrides or {})}
            for value in factor.values
        ],
    }


def _validate_materialized_runs(sweep: SweepConfig) -> None:
    runs = expand_sweep(sweep)
    run_ids = [run.run_id for run in runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("generated run ids must be unique")
    output_root = Path(sweep.output_dir)
    for run in runs:
        output_dir = Path(str(run.config["output_dir"]))
        if output_root not in (output_dir, *output_dir.parents):
            raise ValueError("generated output_dir must live under sweep output_dir")
        build_experiment_config(run.config)


def _require_mapping(value: object, field: str) -> JSONDict:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _require_non_empty_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value
```

- [ ] **Step 4: Run Task 1 tests and verify they pass**

Run:

```bash
./.venv/bin/python -m pytest tests/test_sweep_config.py -v
```

Expected: PASS.

- [ ] **Step 5: Add invalid sweep config tests**

Append to `tests/test_sweep_config.py`:

```python
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.update({"sweep_name": ""}), "sweep_name must be a non-empty string"),
        (lambda data: data.update({"base_config": []}), "base_config must be an object"),
        (lambda data: data.update({"factors": []}), "factors must be a non-empty list"),
        (
            lambda data: data.update(
                {
                    "factors": [
                        {"name": "seed", "path": "seed", "values": [1]},
                        {"name": "seed", "path": "seed", "values": [2]},
                    ]
                }
            ),
            "factor names must be unique",
        ),
        (
            lambda data: data.update(
                {"factors": [{"name": "seed", "path": "seed", "values": [1, 1]}]}
            ),
            "factor seed labels must be unique",
        ),
        (
            lambda data: data.update(
                {"factors": [{"name": "bad", "path": "seed.bad", "values": [1]}]}
            ),
            "traverses non-object field seed",
        ),
        (
            lambda data: data.update(
                {
                    "factors": [
                        {
                            "name": "bad_value",
                            "path": "seed",
                            "values": [{"label": "one", "overrides": {"seed": 1}}],
                        }
                    ]
                }
            ),
            "factor bad_value path values must be scalar",
        ),
        (
            lambda data: data.update(
                {"factors": [{"name": "topology", "values": [{"label": "cycle"}]}]}
            ),
            "factor topology overrides must be an object",
        ),
    ],
)
def test_load_sweep_config_rejects_invalid_shapes(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    data = valid_sweep_dict(tmp_path)
    mutation(data)
    path = tmp_path / "bad_sweep.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_sweep_config(path)


def test_load_sweep_config_rejects_generated_invalid_experiment_config(
    tmp_path: Path,
) -> None:
    data = valid_sweep_dict(tmp_path)
    data["factors"] = [
        {
            "name": "bad_threshold",
            "path": "update_policy.adoption_threshold",
            "values": [0.1],
        }
    ]
    path = tmp_path / "bad_generated.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="adoption_threshold must be between 0.5 and 1.0"):
        load_sweep_config(path)
```

- [ ] **Step 6: Run Task 1 full tests and commit**

Run:

```bash
./.venv/bin/python -m pytest tests/test_sweep_config.py -v
```

Expected: PASS.

Commit:

```bash
git add src/society_simulation/sweep_config.py tests/test_sweep_config.py
git commit -m "feat: add sweep config expansion"
```

---

### Task 2: Sweep Artifact Writers

**Files:**
- Create: `src/society_simulation/sweep_artifacts.py`
- Create: `tests/test_sweep_artifacts.py`

- [ ] **Step 1: Write failing artifact writer tests**

Create `tests/test_sweep_artifacts.py`:

```python
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
        rows = list(csv.DictReader(handle))
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
```

- [ ] **Step 2: Run artifact tests and verify they fail**

Run:

```bash
./.venv/bin/python -m pytest tests/test_sweep_artifacts.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'society_simulation.sweep_artifacts'`.

- [ ] **Step 3: Implement artifact writer**

Create `src/society_simulation/sweep_artifacts.py`:

```python
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Any

from society_simulation.sweep_config import MaterializedRun, SweepConfig


SUMMARY_METRIC_FIELDS = (
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
)

NUMERIC_SUMMARY_FIELDS = (
    "final_a_fraction",
    "time_to_consensus",
    "polarization_index",
    "opinion_variance",
    "mean_belief",
    "edge_disagreement_rate",
    "component_count",
)


@dataclass(frozen=True)
class SweepRunRecord:
    run_id: str
    labels: dict[str, str]
    experiment_name: str
    output_dir: object
    status: str
    error: str | None
    metrics: dict[str, Any]


@dataclass(frozen=True)
class SweepArtifactPaths:
    output_dir: Path
    sweep_config_path: Path
    manifest_path: Path
    summary_csv_path: Path
    summary_json_path: Path


def write_sweep_artifacts(
    sweep: SweepConfig,
    planned_runs: tuple[MaterializedRun, ...],
    records: tuple[SweepRunRecord, ...],
) -> SweepArtifactPaths:
    output_dir = Path(sweep.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = SweepArtifactPaths(
        output_dir=output_dir,
        sweep_config_path=output_dir / "sweep_config.json",
        manifest_path=output_dir / "manifest.jsonl",
        summary_csv_path=output_dir / "summary.csv",
        summary_json_path=output_dir / "summary.json",
    )

    record_by_id = {record.run_id: record for record in records}
    rows = [
        _row_for_run(sweep, run, record_by_id.get(run.run_id))
        for run in planned_runs
    ]

    paths.sweep_config_path.write_text(
        json.dumps(sweep.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_manifest(paths.manifest_path, rows)
    _write_summary_csv(paths.summary_csv_path, sweep, rows)
    paths.summary_json_path.write_text(
        json.dumps(_summary_json(sweep, rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def _row_for_run(
    sweep: SweepConfig,
    run: MaterializedRun,
    record: SweepRunRecord | None,
) -> dict[str, Any]:
    metrics = {} if record is None else record.metrics
    flattened = _flatten_metrics(metrics)
    row: dict[str, Any] = {
        "run_id": run.run_id,
        "experiment_name": run.config["experiment_name"],
        "output_dir": run.config["output_dir"],
        "status": "pending" if record is None else record.status,
        "error": "" if record is None or record.error is None else record.error,
    }
    for factor in sweep.factors:
        row[factor.name] = run.labels[factor.name]
    for field in SUMMARY_METRIC_FIELDS:
        row[field] = flattened.get(field, "")
    return row


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    counts = metrics.get("final_action_counts") or metrics.get("action_counts")
    if isinstance(counts, dict):
        flattened["final_action_counts_A"] = counts.get("A", "")
        flattened["final_action_counts_B"] = counts.get("B", "")
    for field in SUMMARY_METRIC_FIELDS:
        if field in ("final_action_counts_A", "final_action_counts_B"):
            continue
        if field in metrics:
            flattened[field] = metrics[field]
    return flattened


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_summary_csv(
    path: Path,
    sweep: SweepConfig,
    rows: list[dict[str, Any]],
) -> None:
    fieldnames = (
        ["run_id"]
        + [factor.name for factor in sweep.factors]
        + ["experiment_name", "output_dir", "status", "error"]
        + list(SUMMARY_METRIC_FIELDS)
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_json(sweep: SweepConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed_rows = [row for row in rows if row["status"] == "completed"]
    failed_rows = [row for row in rows if row["status"] == "failed"]
    return {
        "sweep_name": sweep.sweep_name,
        "runs": len(rows),
        "completed": len(completed_rows),
        "failed": len(failed_rows),
        "metric_means": _metric_means(completed_rows),
        "groups": _group_summaries(sweep, rows),
    }


def _group_summaries(
    sweep: SweepConfig,
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for factor in sweep.factors:
        factor_groups: dict[str, dict[str, Any]] = {}
        values = sorted({str(row[factor.name]) for row in rows})
        for value in values:
            group_rows = [row for row in rows if row[factor.name] == value]
            completed_rows = [row for row in group_rows if row["status"] == "completed"]
            failed_rows = [row for row in group_rows if row["status"] == "failed"]
            factor_groups[value] = {
                "runs": len(group_rows),
                "completed": len(completed_rows),
                "failed": len(failed_rows),
                "metric_means": _metric_means(completed_rows),
            }
        groups[factor.name] = factor_groups
    return groups


def _metric_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    means: dict[str, float] = {}
    for field in NUMERIC_SUMMARY_FIELDS:
        values = [_to_float(row[field]) for row in rows if row[field] != ""]
        numeric_values = [value for value in values if value is not None]
        if numeric_values:
            means[field] = mean(numeric_values)
    return means


def _to_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
```

- [ ] **Step 4: Run artifact tests and commit**

Run:

```bash
./.venv/bin/python -m pytest tests/test_sweep_artifacts.py tests/test_sweep_config.py -v
```

Expected: PASS.

Commit:

```bash
git add src/society_simulation/sweep_artifacts.py tests/test_sweep_artifacts.py
git commit -m "feat: add sweep artifact writers"
```

---

### Task 3: Sweep Runner Execution

**Files:**
- Create: `src/society_simulation/sweep_runner.py`
- Create: `tests/test_sweep_runner.py`

- [ ] **Step 1: Write failing sweep runner tests**

Create `tests/test_sweep_runner.py`:

```python
import csv
from pathlib import Path
from types import SimpleNamespace

from society_simulation.sweep_config import parse_sweep_config
from society_simulation.sweep_runner import run_sweep
from tests.test_sweep_config import valid_sweep_dict


def test_run_sweep_executes_all_materialized_runs(tmp_path: Path) -> None:
    data = valid_sweep_dict(tmp_path)
    data["factors"] = [
        {"name": "seed", "path": "seed", "values": [1, 2]},
        {
            "name": "topology",
            "values": [
                {"label": "cycle", "overrides": {"topology": {"type": "cycle"}}}
            ],
        },
    ]
    sweep = parse_sweep_config(data)

    result = run_sweep(sweep)

    assert result.sweep_name == "network_topology_sweep"
    assert result.runs == 2
    assert result.completed == 2
    assert result.failed == 0
    assert result.summary_csv_path.exists()
    assert result.manifest_path.exists()
    for record in result.records:
        assert Path(str(record.output_dir)).exists()
        assert (Path(str(record.output_dir)) / "metrics.json").exists()

    with result.summary_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["status"] == "completed"


def test_run_sweep_records_runtime_failures_and_continues(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data = valid_sweep_dict(tmp_path)
    data["factors"] = [{"name": "seed", "path": "seed", "values": [1, 2]}]
    sweep = parse_sweep_config(data)
    calls: list[int] = []

    def fake_run_experiment(config: object) -> object:
        seed = config.seed
        calls.append(seed)
        if seed == 1:
            raise ValueError("seed one failed")
        return SimpleNamespace(
            metrics={
                "final_action_counts": {"A": 1, "B": 5},
                "final_a_fraction": 1 / 6,
                "consensus_reached": False,
                "consensus_action": None,
                "time_to_consensus": None,
                "polarization_index": 0.0,
                "opinion_variance": 0.0,
                "mean_belief": 0.4,
                "edge_disagreement_rate": 0.0,
                "component_count": 1,
            },
            output_dir=Path(config.output_dir),
        )

    monkeypatch.setattr("society_simulation.sweep_runner.run_experiment", fake_run_experiment)

    result = run_sweep(sweep)

    assert calls == [1, 2]
    assert result.runs == 2
    assert result.completed == 1
    assert result.failed == 1
    assert result.records[0].status == "failed"
    assert result.records[0].error == "seed one failed"
    assert result.records[1].status == "completed"
```

- [ ] **Step 2: Run runner tests and verify they fail**

Run:

```bash
./.venv/bin/python -m pytest tests/test_sweep_runner.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'society_simulation.sweep_runner'`.

- [ ] **Step 3: Implement sweep runner**

Create `src/society_simulation/sweep_runner.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from society_simulation.runner import run_experiment
from society_simulation.sweep_artifacts import (
    SweepArtifactPaths,
    SweepRunRecord,
    write_sweep_artifacts,
)
from society_simulation.sweep_config import (
    SweepConfig,
    build_experiment_config,
    expand_sweep,
)


@dataclass(frozen=True)
class SweepRunResult:
    sweep_name: str
    runs: int
    completed: int
    failed: int
    output_dir: Path
    manifest_path: Path
    summary_csv_path: Path
    summary_json_path: Path
    records: tuple[SweepRunRecord, ...]


def run_sweep(sweep: SweepConfig) -> SweepRunResult:
    planned_runs = expand_sweep(sweep)
    records: list[SweepRunRecord] = []

    for planned_run in planned_runs:
        try:
            config = build_experiment_config(planned_run.config)
            result = run_experiment(config)
        except Exception as exc:
            records.append(
                SweepRunRecord(
                    run_id=planned_run.run_id,
                    labels=planned_run.labels,
                    experiment_name=str(planned_run.config["experiment_name"]),
                    output_dir=planned_run.config["output_dir"],
                    status="failed",
                    error=str(exc),
                    metrics={},
                )
            )
            continue

        records.append(
            SweepRunRecord(
                run_id=planned_run.run_id,
                labels=planned_run.labels,
                experiment_name=config.experiment_name,
                output_dir=result.output_dir,
                status="completed",
                error=None,
                metrics=result.metrics,
            )
        )

    artifact_paths = write_sweep_artifacts(sweep, planned_runs, tuple(records))
    return _build_result(sweep, records, artifact_paths)


def _build_result(
    sweep: SweepConfig,
    records: list[SweepRunRecord],
    artifact_paths: SweepArtifactPaths,
) -> SweepRunResult:
    failed = sum(1 for record in records if record.status == "failed")
    completed = sum(1 for record in records if record.status == "completed")
    return SweepRunResult(
        sweep_name=sweep.sweep_name,
        runs=len(records),
        completed=completed,
        failed=failed,
        output_dir=artifact_paths.output_dir,
        manifest_path=artifact_paths.manifest_path,
        summary_csv_path=artifact_paths.summary_csv_path,
        summary_json_path=artifact_paths.summary_json_path,
        records=tuple(records),
    )
```

- [ ] **Step 4: Run runner tests and commit**

Run:

```bash
./.venv/bin/python -m pytest tests/test_sweep_runner.py tests/test_sweep_artifacts.py tests/test_sweep_config.py -v
```

Expected: PASS.

Commit:

```bash
git add src/society_simulation/sweep_runner.py tests/test_sweep_runner.py
git commit -m "feat: add sweep runner execution"
```

---

### Task 4: CLI Sweep Command

**Files:**
- Modify: `src/society_simulation/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI sweep tests**

Append to `tests/test_cli.py`:

```python
def test_cli_sweep_runs_config_and_prints_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sweep_path = tmp_path / "sweep.json"
    output_dir = tmp_path / "sweep-output"
    sweep_path.write_text(
        json.dumps(
            {
                "sweep_name": "cli_sweep",
                "base_config": {
                    "experiment_name": "network_herding",
                    "seed": 1,
                    "num_agents": 6,
                    "initial_opinion": {"type": "bernoulli", "probability_a": 0.5},
                    "topology": {"type": "cycle"},
                    "scheduler": {"type": "synchronous_rounds", "rounds": 2},
                    "observation_policy": {"type": "neighbor_actions"},
                    "update_policy": {"type": "majority_rule"},
                    "output_dir": str(tmp_path / "ignored"),
                },
                "factors": [{"name": "seed", "path": "seed", "values": [1, 2]}],
                "output_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["sweep", str(sweep_path)])

    assert exit_code == 0
    output = capsys.readouterr().out.splitlines()
    assert output == [
        "sweep=cli_sweep",
        "runs=2",
        "completed=2",
        "failed=0",
        f"output_dir={output_dir}",
        f"summary_csv={output_dir / 'summary.csv'}",
    ]
    assert (output_dir / "manifest.jsonl").exists()
    assert (output_dir / "summary.csv").exists()


def test_cli_sweep_invalid_config_reports_clean_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sweep_path = tmp_path / "bad_sweep.json"
    sweep_path.write_text(json.dumps({"sweep_name": ""}), encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["sweep", str(sweep_path)])

    assert exc_info.value.code == 2
    captured = capsys.readouterr().err
    assert "Invalid sweep config file" in captured
    assert "sweep_name must be a non-empty string" in captured
    assert "Traceback" not in captured
```

- [ ] **Step 2: Run CLI tests and verify they fail**

Run:

```bash
./.venv/bin/python -m pytest tests/test_cli.py -v
```

Expected: FAIL because parser choices do not include `sweep`.

- [ ] **Step 3: Implement CLI sweep branch**

Modify `src/society_simulation/cli.py`:

```python
from society_simulation.sweep_config import load_sweep_config
from society_simulation.sweep_runner import run_sweep
```

Update `build_parser`:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="society-sim")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run").add_argument("config")
    subparsers.add_parser("sweep").add_argument("config")

    return parser
```

Update `main` command dispatch so the start of the function is:

```python
def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "run":
        return _run_single_config(parser, args.config)
    if args.command == "sweep":
        return _run_sweep_config(parser, args.config)
    return 1
```

Move the existing run logic into `_run_single_config` without changing its printed lines:

```python
def _run_single_config(parser: argparse.ArgumentParser, config_path: str) -> int:
    try:
        config = load_config(config_path)
    except OSError as exc:
        parser.error(f"Unable to read config file '{config_path}': {exc}")
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(f"Invalid config file '{config_path}': {exc}")

    try:
        result = run_experiment(config)
        metrics = result.metrics
        action_counts = _require_action_counts(metrics)
    except (OSError, ValueError) as exc:
        parser.error(f"Experiment run failed for '{config_path}': {exc}")

    print(f"experiment={config.experiment_name}")
    if hasattr(result, "true_state"):
        print(f"true_state={result.true_state}")
    print(f"action_counts={action_counts}")
    if "correct_cascade" in metrics:
        print(f"correct_cascade={metrics['correct_cascade']}")
    if "wrong_cascade" in metrics:
        print(f"wrong_cascade={metrics['wrong_cascade']}")
    if "consensus_reached" in metrics:
        print(f"consensus_reached={metrics['consensus_reached']}")
    if "edge_disagreement_rate" in metrics:
        print(f"edge_disagreement_rate={metrics['edge_disagreement_rate']}")
    print(f"output_dir={result.output_dir}")

    return 0
```

Add `_run_sweep_config`:

```python
def _run_sweep_config(parser: argparse.ArgumentParser, config_path: str) -> int:
    try:
        sweep = load_sweep_config(config_path)
    except OSError as exc:
        parser.error(f"Unable to read sweep config file '{config_path}': {exc}")
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(f"Invalid sweep config file '{config_path}': {exc}")

    result = run_sweep(sweep)
    print(f"sweep={result.sweep_name}")
    print(f"runs={result.runs}")
    print(f"completed={result.completed}")
    print(f"failed={result.failed}")
    print(f"output_dir={result.output_dir}")
    print(f"summary_csv={result.summary_csv_path}")
    if result.failed:
        return 1
    return 0
```

- [ ] **Step 4: Run CLI tests and commit**

Run:

```bash
./.venv/bin/python -m pytest tests/test_cli.py tests/test_sweep_runner.py tests/test_sweep_artifacts.py tests/test_sweep_config.py -v
```

Expected: PASS.

Commit:

```bash
git add src/society_simulation/cli.py tests/test_cli.py
git commit -m "feat: add sweep CLI command"
```

---

### Task 5: Example Sweep and Documentation

**Files:**
- Create: `experiments/network_topology_sweep.json`
- Modify: `README.md`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing example config test**

Append to `tests/test_cli.py`:

```python
def test_example_network_topology_sweep_exists_and_is_valid() -> None:
    from society_simulation.sweep_config import expand_sweep, load_sweep_config

    sweep = load_sweep_config("experiments/network_topology_sweep.json")
    runs = expand_sweep(sweep)

    assert sweep.sweep_name == "network_topology_sweep"
    assert len(runs) == 48
```

- [ ] **Step 2: Run the example test and verify it fails**

Run:

```bash
./.venv/bin/python -m pytest tests/test_cli.py::test_example_network_topology_sweep_exists_and_is_valid -v
```

Expected: FAIL because `experiments/network_topology_sweep.json` does not exist.

- [ ] **Step 3: Add example sweep config**

Create `experiments/network_topology_sweep.json`:

```json
{
  "sweep_name": "network_topology_sweep",
  "base_config": {
    "experiment_name": "network_herding",
    "seed": 1,
    "num_agents": 30,
    "initial_opinion": {
      "type": "bernoulli",
      "probability_a": 0.5
    },
    "topology": {
      "type": "small_world",
      "degree": 4,
      "rewiring_probability": 0.1
    },
    "scheduler": {
      "type": "synchronous_rounds",
      "rounds": 12
    },
    "observation_policy": {
      "type": "neighbor_actions"
    },
    "update_policy": {
      "type": "threshold",
      "adoption_threshold": 0.6
    },
    "output_dir": "runs/ignored-by-sweep"
  },
  "factors": [
    {
      "name": "seed",
      "path": "seed",
      "values": [1, 2, 3]
    },
    {
      "name": "initial_a",
      "path": "initial_opinion.probability_a",
      "values": [0.45, 0.55]
    },
    {
      "name": "topology",
      "values": [
        {
          "label": "complete",
          "overrides": {
            "topology": {
              "type": "complete"
            }
          }
        },
        {
          "label": "cycle",
          "overrides": {
            "topology": {
              "type": "cycle"
            }
          }
        },
        {
          "label": "small_world_k4_p01",
          "overrides": {
            "topology": {
              "type": "small_world",
              "degree": 4,
              "rewiring_probability": 0.1
            }
          }
        },
        {
          "label": "erdos_renyi_p02",
          "overrides": {
            "topology": {
              "type": "erdos_renyi",
              "edge_probability": 0.2
            }
          }
        }
      ]
    },
    {
      "name": "threshold",
      "path": "update_policy.adoption_threshold",
      "values": [0.55, 0.7]
    }
  ],
  "output_dir": "runs/sweeps/network_topology_sweep"
}
```

- [ ] **Step 4: Update README**

Modify `README.md` so the `Run` section includes:

````markdown
Network topology sweep:

```bash
python -m society_simulation sweep experiments/network_topology_sweep.json
```
````

Modify the artifact section to add:

```markdown
Sweep runs write:

- `sweep_config.json`
- `manifest.jsonl`
- `summary.csv`
- `summary.json`
- `runs/<run_id>/...`
```

- [ ] **Step 5: Run example sweep and docs-related tests**

Run:

```bash
./.venv/bin/python -m pytest tests/test_cli.py::test_example_network_topology_sweep_exists_and_is_valid -v
./.venv/bin/python -m society_simulation sweep experiments/network_topology_sweep.json
```

Expected pytest: PASS.

Expected CLI output:

```text
sweep=network_topology_sweep
runs=48
completed=48
failed=0
output_dir=runs/sweeps/network_topology_sweep
summary_csv=runs/sweeps/network_topology_sweep/summary.csv
```

- [ ] **Step 6: Commit example and docs**

Run:

```bash
git add experiments/network_topology_sweep.json README.md tests/test_cli.py
git commit -m "docs: add network topology sweep example"
```

---

### Task 6: Final Verification and Integration

**Files:**
- No new files.
- Verify all changed behavior.

- [ ] **Step 1: Run focused sweep tests**

Run:

```bash
./.venv/bin/python -m pytest tests/test_sweep_config.py tests/test_sweep_artifacts.py tests/test_sweep_runner.py tests/test_cli.py -v
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

Expected: both commands exit 0 and print `output_dir=...`.

- [ ] **Step 4: Run the new sweep example**

Run:

```bash
./.venv/bin/python -m society_simulation sweep experiments/network_topology_sweep.json
```

Expected: command exits 0 and prints `completed=48` and `failed=0`.

- [ ] **Step 5: Inspect generated sweep artifacts**

Run:

```bash
test -f runs/sweeps/network_topology_sweep/manifest.jsonl
test -f runs/sweeps/network_topology_sweep/summary.csv
test -f runs/sweeps/network_topology_sweep/summary.json
head -n 2 runs/sweeps/network_topology_sweep/summary.csv
```

Expected: all `test -f` commands exit 0, and CSV header includes `run_id`, `topology`, `consensus_reached`, and `edge_disagreement_rate`.

- [ ] **Step 6: Confirm git status and commit any missed documentation fixes**

Run:

```bash
git status --short
```

Expected: no tracked source or test files modified. Ignored `runs/` artifacts may exist and should not be committed.

If README or example corrections were needed during verification:

```bash
git add README.md experiments/network_topology_sweep.json
git commit -m "docs: clarify sweep usage"
```

- [ ] **Step 7: Final code review**

Dispatch a reviewer with this brief:

```text
Review the sweep runner implementation against docs/superpowers/specs/2026-06-17-sweep-runner-design.md and docs/superpowers/plans/2026-06-18-sweep-runner.md. Focus on correctness, deterministic output, config validation, artifact completeness, and preservation of existing `run` CLI behavior. Do not edit files. Return Critical/Important/Minor findings with file and line references.
```

Fix any Critical or Important findings, rerun the relevant tests, and commit fixes.

- [ ] **Step 8: Merge/push workflow**

Use `superpowers:finishing-a-development-branch` after review approval. If implementing directly on `main`, verify:

```bash
git status --short --branch
git log --oneline --decorate --max-count=5
git push origin main
```

Expected: `main` pushed to `origin/main` with all sweep runner commits.
